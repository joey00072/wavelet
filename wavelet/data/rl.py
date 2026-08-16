from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import torch
from torch import Tensor
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from wavelet.configs.rl_config import RLDataConfig
from wavelet.data._stateful import StatefulDatasetMixin
from wavelet.data.sft import (
    IGNORE_INDEX,
    Example,
    Sample,
    build_sample,
    load_data_payloads,
    normalize_record,
)


class RLSample(TypedDict):
    """One tokenized RL example before tensor collation."""

    input_ids: list[int]
    position_ids: list[int]
    target_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: NotRequired[list[float]]
    teacher_logprobs: NotRequired[list[float]]
    temperatures: list[float]
    reward: float | None
    sample_count: NotRequired[int]


class RLBatch(TypedDict):
    """Padded tensors consumed by the RL trainer."""

    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    target_ids: Tensor
    labels: Tensor
    loss_mask: Tensor
    advantages: Tensor
    rewards: Tensor
    has_inference_logprobs: Tensor
    inference_logprobs: Tensor
    has_teacher_logprobs: Tensor
    teacher_logprobs: Tensor
    temperatures: Tensor
    sample_counts: Tensor


@dataclass
class RLExample:
    """Serializable rollout record shared by orchestration and training."""

    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    advantage: float | list[float] | None
    reward: float | None
    input_ids: list[int] | None = None
    target_ids: list[int] | None = None
    loss_mask: list[bool] | None = None
    target_completion: list[dict[str, str]] | None = None
    inference_logprobs: list[float] | None = None
    teacher_logprobs: list[float] | None = None
    temperatures: float | list[float] | None = None
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    source: str = "dataset"


def rl_example_to_payload(record: RLExample) -> dict[str, Any]:
    """Convert one rollout record to its JSON-compatible wire payload."""
    return asdict(record)


def rl_example_from_payload(payload: dict[str, Any]) -> RLExample:
    """Construct one rollout record from its wire payload."""
    return RLExample(
        prompt=payload["prompt"],
        completion=payload["completion"],
        advantage=payload.get("advantage"),
        reward=payload.get("reward"),
        input_ids=payload.get("input_ids"),
        target_ids=payload.get("target_ids"),
        loss_mask=payload.get("loss_mask"),
        target_completion=payload.get("target_completion"),
        inference_logprobs=payload.get("inference_logprobs"),
        teacher_logprobs=payload.get("teacher_logprobs"),
        temperatures=payload.get("temperatures"),
        tools=payload.get("tools"),
        chat_template_kwargs=payload.get("chat_template_kwargs"),
        metadata=payload.get("metadata"),
        source=payload.get("source") or "dataset",
    )


def rl_examples_to_payload(records: list[RLExample]) -> list[dict[str, Any]]:
    """Convert rollout records to wire payloads."""
    return [rl_example_to_payload(record) for record in records]


def rl_examples_from_payload(payloads: list[dict[str, Any]]) -> list[RLExample]:
    """Construct rollout records from wire payloads."""
    return [rl_example_from_payload(payload) for payload in payloads]


def serialize_rl_record(
    record: RLExample,
    config: RLDataConfig,
    *,
    task: str,
    example_id: str,
) -> dict[str, object]:
    """Serialize a rollout into the trainer JSONL contract."""
    payload: dict[str, object] = {
        config.prompt_column: record.prompt,
        config.completion_column: record.completion,
        "target_completion": record.target_completion,
        "source": record.source,
        "env_name": record.source,
        "task": task,
        "example_id": example_id,
        config.advantage_column: record.advantage,
        config.reward_column: record.reward,
        config.temperature_column: record.temperatures,
    }
    optional = {
        "input_ids": record.input_ids,
        "target_ids": record.target_ids,
        "loss_mask": record.loss_mask,
        config.inference_logprobs_column: record.inference_logprobs,
        config.teacher_logprobs_column: record.teacher_logprobs,
        config.tools_column: record.tools,
        config.chat_template_kwargs_column: record.chat_template_kwargs,
        config.metadata_column: record.metadata,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    return payload


def deserialize_rl_record(payload: dict[str, Any], config: RLDataConfig) -> RLExample:
    """Deserialize the trainer JSONL contract into one rollout record."""
    base = normalize_record(payload, config)
    advantage = payload.get(config.advantage_column)
    reward_value = payload.get(config.reward_column)
    reward = None if reward_value is None else float(reward_value)
    temperatures = payload.get(config.temperature_column)
    metadata = payload.get(config.metadata_column)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object when provided.")
    if config.source == "fake":
        if advantage is None and reward is None:
            reward = 1.0
        if temperatures is None:
            temperatures = 1.0
    return RLExample(
        prompt=base.prompt,
        completion=base.completion,
        target_completion=payload.get("target_completion", base.completion),
        tools=base.tools,
        chat_template_kwargs=base.chat_template_kwargs,
        source=str(payload.get("source") or base.source),
        advantage=advantage,
        reward=reward,
        input_ids=payload.get("input_ids"),
        target_ids=payload.get("target_ids"),
        loss_mask=payload.get("loss_mask"),
        inference_logprobs=payload.get(config.inference_logprobs_column),
        teacher_logprobs=payload.get(config.teacher_logprobs_column),
        temperatures=temperatures,
        metadata=metadata,
    )


def trainable_token_count(sample: RLSample) -> int:
    """Return the number of tokens that contribute to the RL loss."""
    return sum(bool(value) for value in sample["loss_mask"])


def trainable_sequence_count(sample: RLSample) -> int:
    """Return the number of packed sequences that contribute to the RL loss."""
    loss_mask = sample["loss_mask"]
    if not any(bool(value) for value in loss_mask):
        return 0

    starts = [
        index
        for index, position_id in enumerate(sample["position_ids"][: len(loss_mask)])
        if position_id == 0
    ]
    if not starts:
        return 1
    if starts[0] != 0:
        starts.insert(0, 0)
    ends = [*starts[1:], len(loss_mask)]
    return sum(
        any(bool(value) for value in loss_mask[start:end])
        for start, end in zip(starts, ends, strict=True)
    )


def pack_samples(
    samples: list[RLSample],
    *,
    seq_len: int,
    pad_to_multiple_of: int,
) -> list[RLSample]:
    """Pack samples with first-fit decreasing bin packing."""
    sorted_samples = sorted(samples, key=lambda sample: -len(sample["input_ids"]))
    bins: list[list[RLSample]] = []
    bin_lengths: list[int] = []
    for sample in sorted_samples:
        sample_len = len(sample["input_ids"])
        for index, current_len in enumerate(bin_lengths):
            if current_len + sample_len <= seq_len:
                bins[index].append(sample)
                bin_lengths[index] += sample_len
                break
        else:
            bins.append([sample])
            bin_lengths.append(sample_len)
    return [
        _merge_samples(items, pad_to_multiple_of=pad_to_multiple_of) for items in bins
    ]


def pad_bins_for_distribution(
    bins: list[RLSample],
    *,
    data_world_size: int,
) -> list[RLSample]:
    """Add zero-loss bins so every data rank receives the same bin count."""
    if data_world_size <= 1 or not bins:
        return bins
    pad_count = (-len(bins)) % data_world_size
    if pad_count == 0:
        return bins
    return [*bins, *(_zero_loss_copy(bins[0]) for _ in range(pad_count))]


def _merge_samples(
    samples: list[RLSample],
    *,
    pad_to_multiple_of: int,
) -> RLSample:
    input_ids: list[int] = []
    target_ids: list[int] = []
    position_ids: list[int] = []
    loss_mask: list[bool] = []
    advantages: list[float] = []
    inference_logprobs: list[float] = []
    teacher_logprobs: list[float] = []
    temperatures: list[float] = []
    rewards = [
        float(sample["reward"]) for sample in samples if sample["reward"] is not None
    ]
    sample_count = sum(int(sample.get("sample_count", 1)) for sample in samples)
    has_inference = all("inference_logprobs" in sample for sample in samples)
    has_teacher = all("teacher_logprobs" in sample for sample in samples)

    for sample in samples:
        input_ids.extend(sample["input_ids"])
        target_ids.extend(sample["target_ids"])
        position_ids.extend(sample["position_ids"])
        loss_mask.extend(sample["loss_mask"])
        advantages.extend(sample["advantages"])
        temperatures.extend(sample["temperatures"])
        if has_inference:
            inference_logprobs.extend(sample["inference_logprobs"])
        if has_teacher:
            teacher_logprobs.extend(sample["teacher_logprobs"])

    _pad_token_streams(
        input_ids,
        target_ids,
        position_ids,
        loss_mask,
        multiple=pad_to_multiple_of,
    )
    packed: RLSample = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "advantages": advantages,
        "temperatures": temperatures,
        "reward": sum(rewards) / len(rewards) if rewards else None,
        "sample_count": sample_count,
    }
    if has_inference:
        packed["inference_logprobs"] = inference_logprobs
    if has_teacher:
        packed["teacher_logprobs"] = teacher_logprobs
    return packed


def _pad_token_streams(
    input_ids: list[int],
    target_ids: list[int],
    position_ids: list[int],
    loss_mask: list[bool],
    *,
    multiple: int,
) -> None:
    if multiple <= 1:
        return
    padding = (-len(input_ids)) % multiple
    if padding == 0:
        return
    input_ids.extend([1] * padding)
    target_ids.extend([0] * padding)
    position_ids.extend(range(padding))
    loss_mask.extend([False] * padding)


def _zero_loss_copy(source: RLSample) -> RLSample:
    dummy: RLSample = {
        "input_ids": list(source["input_ids"]),
        "position_ids": list(source["position_ids"]),
        "target_ids": list(source["target_ids"]),
        "loss_mask": [False] * len(source["loss_mask"]),
        "advantages": [],
        "temperatures": [],
        "reward": None,
        "sample_count": 0,
    }
    if "inference_logprobs" in source:
        dummy["inference_logprobs"] = []
    if "teacher_logprobs" in source:
        dummy["teacher_logprobs"] = []
    return dummy


def _validate_trainable_values(
    values: list[float] | None,
    *,
    field_name: str,
    trainable_tokens: int,
) -> None:
    if values is not None and len(values) != trainable_tokens:
        raise ValueError(
            f"{field_name} must align with the number of trainable tokens in the sample"
        )


def _expand_trainable_values(
    values: list[float] | None,
    *,
    mask: list[bool],
    default: float,
) -> list[float]:
    expanded: list[float] = []
    trainable_index = 0
    for trainable in mask:
        if trainable:
            value = default if values is None else float(values[trainable_index])
            expanded.append(value)
            trainable_index += 1
        else:
            expanded.append(default)
    return expanded


def collate_rl_batch(
    batch: list[RLSample],
    *,
    pad_token_id: int,
) -> RLBatch:
    """Pad tokenized RL samples and align trainable-only value streams."""
    max_len = max(len(item["input_ids"]) for item in batch)
    output: dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "position_ids": [],
        "target_ids": [],
        "labels": [],
        "loss_mask": [],
        "advantages": [],
        "rewards": [],
        "has_inference_logprobs": [],
        "inference_logprobs": [],
        "has_teacher_logprobs": [],
        "teacher_logprobs": [],
        "temperatures": [],
        "sample_counts": [],
    }

    for item in batch:
        _append_sample(output, item, max_len=max_len, pad_token_id=pad_token_id)

    stacked = {key: torch.stack(values) for key, values in output.items()}
    return cast(RLBatch, stacked)


def _append_sample(
    output: dict[str, list[torch.Tensor]],
    item: RLSample,
    *,
    max_len: int,
    pad_token_id: int,
) -> None:
    sequence_length = len(item["input_ids"])
    padding = max_len - sequence_length
    loss_mask = list(item["loss_mask"])
    trainable_tokens = sum(loss_mask)
    inference_logprobs = item.get("inference_logprobs")
    teacher_logprobs = item.get("teacher_logprobs")

    _validate_trainable_values(
        item["advantages"],
        field_name="advantages",
        trainable_tokens=trainable_tokens,
    )
    _validate_trainable_values(
        item["temperatures"],
        field_name="temperatures",
        trainable_tokens=trainable_tokens,
    )
    _validate_trainable_values(
        inference_logprobs,
        field_name="inference_logprobs",
        trainable_tokens=trainable_tokens,
    )
    _validate_trainable_values(
        teacher_logprobs,
        field_name="teacher_logprobs",
        trainable_tokens=trainable_tokens,
    )

    expanded_advantages = _expand_trainable_values(
        item["advantages"], mask=loss_mask, default=0.0
    )
    expanded_inference = _expand_trainable_values(
        inference_logprobs, mask=loss_mask, default=0.0
    )
    expanded_teacher = _expand_trainable_values(
        teacher_logprobs, mask=loss_mask, default=0.0
    )
    expanded_temperatures = _expand_trainable_values(
        item["temperatures"], mask=loss_mask, default=1.0
    )
    labels = [
        target_id if trainable else IGNORE_INDEX
        for target_id, trainable in zip(item["target_ids"], loss_mask, strict=True)
    ]

    output["input_ids"].append(
        torch.tensor(
            item["input_ids"] + [pad_token_id] * padding,
            dtype=torch.long,
        )
    )
    output["attention_mask"].append(
        torch.tensor([1] * sequence_length + [0] * padding, dtype=torch.long)
    )
    output["position_ids"].append(
        torch.tensor(
            item["position_ids"] + list(range(sequence_length, max_len)),
            dtype=torch.long,
        )
    )
    output["target_ids"].append(
        torch.tensor(item["target_ids"] + [0] * padding, dtype=torch.long)
    )
    output["labels"].append(
        torch.tensor(labels + [IGNORE_INDEX] * padding, dtype=torch.long)
    )
    output["loss_mask"].append(
        torch.tensor(loss_mask + [False] * padding, dtype=torch.bool)
    )
    output["advantages"].append(
        torch.tensor(expanded_advantages + [0.0] * padding, dtype=torch.float32)
    )
    output["rewards"].append(
        torch.tensor(
            float("nan") if item["reward"] is None else float(item["reward"]),
            dtype=torch.float32,
        )
    )
    output["sample_counts"].append(
        torch.tensor(int(item.get("sample_count", 1)), dtype=torch.long)
    )
    output["has_inference_logprobs"].append(
        torch.tensor(inference_logprobs is not None, dtype=torch.bool)
    )
    output["inference_logprobs"].append(
        torch.tensor(expanded_inference + [0.0] * padding, dtype=torch.float32)
    )
    output["has_teacher_logprobs"].append(
        torch.tensor(teacher_logprobs is not None, dtype=torch.bool)
    )
    output["teacher_logprobs"].append(
        torch.tensor(expanded_teacher + [0.0] * padding, dtype=torch.float32)
    )
    output["temperatures"].append(
        torch.tensor(expanded_temperatures + [1.0] * padding, dtype=torch.float32)
    )


def count_nonempty_jsonl_rows(
    path: Path,
    *,
    description: str = "JSONL file",
) -> int:
    rows = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    if rows == 0:
        raise ValueError(f"{description} '{path}' contains no rows.")
    return rows


def _coerce_advantages(
    value: float | list[float] | None,
    *,
    fallback_reward: float | None,
    num_trainable_tokens: int,
) -> list[float]:
    if num_trainable_tokens == 0:
        return []
    if value is None:
        if fallback_reward is None:
            raise ValueError("Each RL row must provide either advantage or reward.")
        return [float(fallback_reward)] * num_trainable_tokens
    if isinstance(value, list):
        if len(value) < num_trainable_tokens:
            raise ValueError(
                "Token-level advantages are shorter than the number of trainable tokens "
                f"({len(value)} < {num_trainable_tokens})."
            )
        return [float(item) for item in value[:num_trainable_tokens]]
    return [float(value)] * num_trainable_tokens


def _coerce_optional_sequence(
    value: float | list[float] | None,
    *,
    num_trainable_tokens: int,
    field_name: str,
    default: float | None = None,
) -> list[float] | None:
    if value is None:
        if default is None:
            return None
        return [default] * num_trainable_tokens
    if isinstance(value, list):
        if len(value) < num_trainable_tokens:
            raise ValueError(
                f"{field_name} is shorter than the number of trainable tokens "
                f"({len(value)} < {num_trainable_tokens})."
            )
        return [float(item) for item in value[:num_trainable_tokens]]
    return [float(value)] * num_trainable_tokens


def _trim_loss_mask_to_sequence(
    loss_mask: list[bool],
    sequence: list[float] | None,
) -> list[bool]:
    if sequence is None:
        return loss_mask
    trainable_count = sum(loss_mask)
    if len(sequence) >= trainable_count:
        return loss_mask

    to_mask = trainable_count - len(sequence)
    trimmed = list(loss_mask)
    for index in range(len(trimmed) - 1, -1, -1):
        if not trimmed[index]:
            continue
        trimmed[index] = False
        to_mask -= 1
        if to_mask == 0:
            break
    return trimmed


def _pretokenized_sample(record: RLExample, seq_len: int) -> RLSample | None:
    if (
        record.input_ids is None
        or record.target_ids is None
        or record.loss_mask is None
    ):
        return None

    input_ids = [int(token_id) for token_id in record.input_ids[:seq_len]]
    target_ids = [int(token_id) for token_id in record.target_ids[:seq_len]]
    loss_mask = [bool(value) for value in record.loss_mask[:seq_len]]
    if not (len(input_ids) == len(target_ids) == len(loss_mask)):
        raise ValueError(
            "Pretokenized RL row has mismatched input_ids, target_ids, and loss_mask "
            f"lengths ({len(input_ids)}, {len(target_ids)}, {len(loss_mask)})."
        )
    if sum(loss_mask) == 0:
        metadata = record.metadata or {}
        if not (
            metadata.get("_wavelet_dummy_rollout")
            or metadata.get("_wavelet_filtered_rollout")
        ):
            return None
    return {
        "input_ids": input_ids,
        "position_ids": list(range(len(input_ids))),
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "advantages": [],
        "temperatures": [],
        "reward": record.reward,
    }


def prepare_rl_sample(
    record: RLExample,
    tokenizer: PreTrainedTokenizerBase,
    data_config: RLDataConfig,
    seq_len: int,
) -> RLSample | None:
    base_sample: Sample | RLSample | None = _pretokenized_sample(record, seq_len)
    if base_sample is None:
        base_sample = build_sample(
            Example(
                prompt=record.prompt,
                completion=record.completion,
                tools=record.tools,
                chat_template_kwargs=record.chat_template_kwargs,
                source=record.source,
            ),
            tokenizer,
            seq_len=seq_len,
            loss_mask_config=data_config.loss_mask,
        )
    if base_sample is None:
        return None

    base_sample["loss_mask"] = _trim_loss_mask_to_sequence(
        base_sample["loss_mask"],
        record.inference_logprobs
        if isinstance(record.inference_logprobs, list)
        else None,
    )
    num_trainable_tokens = sum(base_sample["loss_mask"])
    advantages = _coerce_advantages(
        record.advantage,
        fallback_reward=record.reward,
        num_trainable_tokens=num_trainable_tokens,
    )
    inference_logprobs = _coerce_optional_sequence(
        record.inference_logprobs,
        num_trainable_tokens=num_trainable_tokens,
        field_name="inference_logprobs",
    )
    teacher_logprobs = _coerce_optional_sequence(
        record.teacher_logprobs,
        num_trainable_tokens=num_trainable_tokens,
        field_name="teacher_logprobs",
    )
    temperatures = _coerce_optional_sequence(
        record.temperatures,
        num_trainable_tokens=num_trainable_tokens,
        field_name="temperature",
        default=1.0,
    )
    assert temperatures is not None

    sample: RLSample = {
        "input_ids": base_sample["input_ids"],
        "position_ids": base_sample["position_ids"],
        "target_ids": base_sample["target_ids"],
        "loss_mask": base_sample["loss_mask"],
        "advantages": advantages,
        "temperatures": temperatures,
        "reward": record.reward,
    }
    metadata = record.metadata or {}
    if metadata.get("_wavelet_dummy_rollout"):
        sample["sample_count"] = 0
    elif "_wavelet_rollout_count" in metadata:
        sample["sample_count"] = int(metadata["_wavelet_rollout_count"])
    if inference_logprobs is not None:
        sample["inference_logprobs"] = inference_logprobs
    if teacher_logprobs is not None:
        sample["teacher_logprobs"] = teacher_logprobs
    return sample


def _normalize_rl_record(payload: dict[str, Any], config: RLDataConfig) -> RLExample:
    return deserialize_rl_record(payload, config)


def load_rl_records(config: RLDataConfig) -> list[RLExample]:
    payloads = load_data_payloads(config)
    rows = [_normalize_rl_record(payload, config) for payload in payloads]
    if config.max_examples is not None:
        rows = rows[: config.max_examples]
    if not rows:
        raise ValueError("No RL rows found for the configured data source.")
    return rows


@dataclass
class RLDataset(StatefulDatasetMixin[RLExample], IterableDataset[RLSample]):
    records: list[RLExample]
    tokenizer: PreTrainedTokenizerBase
    seq_len: int
    data_config: RLDataConfig
    shuffle: bool = False
    seed: int = 0
    data_rank: int = 0
    data_world_size: int = 1

    def __post_init__(self) -> None:
        self._initialize_iteration_state()

    def loss_scale_for_next_local_batch(
        self,
        local_batch_size: int,
        *,
        normalization: str = "token",
    ) -> int:
        """Count trainable units for the next local optimizer batch."""
        if local_batch_size <= 0:
            return 1

        num_examples = len(self.records)
        data_rank, data_world_size = self._effective_data_partition()
        total = 0
        collected = 0
        offset = 0
        while collected < local_batch_size:
            next_step = self.step + offset + 1
            epoch = (next_step - 1) // num_examples
            sample_index = (next_step - 1) % num_examples
            offset += 1
            if (next_step - 1) % data_world_size != data_rank:
                continue

            record_index = self._order_for_epoch(epoch)[sample_index]
            sample = prepare_rl_sample(
                self.records[record_index],
                self.tokenizer,
                self.data_config,
                self.seq_len,
            )
            if sample is None:
                continue
            if normalization == "sequence":
                total += trainable_sequence_count(sample)
            else:
                total += trainable_token_count(sample)
            collected += 1
        return max(total, 1)

    def __iter__(self) -> Iterator[RLSample]:
        for record_index in self._local_record_indexes():
            record = self.records[record_index]
            sample = prepare_rl_sample(
                record,
                self.tokenizer,
                self.data_config,
                self.seq_len,
            )
            if sample is None:
                self.skipped += 1
                continue

            self._record_sample(record.source, len(sample["input_ids"]))
            yield sample


@dataclass
class PackedRLDataset(StatefulDatasetMixin[RLExample], IterableDataset[RLSample]):
    records: list[RLExample]
    tokenizer: PreTrainedTokenizerBase
    seq_len: int
    data_config: RLDataConfig
    shuffle: bool = False
    seed: int = 0
    data_rank: int = 0
    data_world_size: int = 1

    def __post_init__(self) -> None:
        self._initialize_iteration_state()
        self._epoch_bins: dict[int, list[RLSample]] = {}
        self._epoch_global_bins: dict[int, list[RLSample]] = {}

    def micro_batch_count(self) -> int:
        return max(len(self._bins_for_epoch(self.epoch)), 1)

    def local_real_sample_count(self) -> int:
        return sum(
            int(sample.get("sample_count", 1))
            for sample in self._bins_for_epoch(self.epoch)
        )

    def loss_scale_for_next_local_batch(
        self,
        _local_batch_size: int,
        *,
        normalization: str = "token",
    ) -> float:
        counter = (
            trainable_sequence_count
            if normalization == "sequence"
            else trainable_token_count
        )
        total = sum(counter(sample) for sample in self._bins_for_epoch(self.epoch))
        return max(float(total), 1.0)

    def __iter__(self) -> Iterator[RLSample]:
        while True:
            bins = self._bins_for_epoch(self.epoch)
            if not bins:
                return
            for sample in bins:
                self.step += 1
                yield sample
            self.epoch += 1

    def _bins_for_epoch(self, epoch: int) -> list[RLSample]:
        cached = self._epoch_bins.get(epoch)
        if cached is not None:
            return cached

        global_bins = self._global_bins_for_epoch(epoch)
        data_rank, data_world_size = self._effective_data_partition()
        bins = global_bins[data_rank::data_world_size]
        self._epoch_bins[epoch] = bins
        return bins

    def _global_bins_for_epoch(self, epoch: int) -> list[RLSample]:
        cached = self._epoch_global_bins.get(epoch)
        if cached is not None:
            return cached

        samples: list[RLSample] = []
        for record_index in self._order_for_epoch(epoch):
            record = self.records[record_index]
            sample = prepare_rl_sample(
                record,
                self.tokenizer,
                self.data_config,
                self.seq_len,
            )
            if sample is None:
                self.skipped += 1
                continue
            self._record_sample(record.source, len(sample["input_ids"]))
            samples.append(sample)

        packed = pack_samples(
            samples,
            seq_len=self.seq_len,
            pad_to_multiple_of=self.data_config.pad_to_multiple_of,
        )
        _, data_world_size = self._effective_data_partition()
        packed = pad_bins_for_distribution(
            packed,
            data_world_size=data_world_size,
        )
        self._epoch_global_bins[epoch] = packed
        return packed


class FakeRLDataset(IterableDataset[RLSample]):
    def __init__(
        self,
        *,
        seq_len: int,
        vocab_size: int,
        length_mode: str,
        input_mode: str,
        seed: int,
    ) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.length_mode = length_mode
        self.input_mode = input_mode
        self.seed = seed
        self.step = 0
        self.epoch = 0

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step = int(state_dict["step"])
        self.epoch = int(state_dict["epoch"])

    def __iter__(self) -> Iterator[RLSample]:
        while True:
            self.step += 1
            self.epoch = self.step // max(self.vocab_size, 1)
            rng = random.Random(self.seed + self.step)
            sample_len = self.seq_len
            if self.length_mode == "variable":
                sample_len = rng.randint(max(2, self.seq_len // 4), self.seq_len)

            if self.input_mode == "increasing":
                full_ids = [
                    (self.step + i) % self.vocab_size for i in range(sample_len + 1)
                ]
            else:
                full_ids = [
                    rng.randrange(self.vocab_size) for _ in range(sample_len + 1)
                ]

            yield {
                "input_ids": full_ids[:-1],
                "position_ids": list(range(sample_len)),
                "target_ids": full_ids[1:],
                "loss_mask": [True] * sample_len,
                "advantages": [rng.uniform(-1.0, 1.0) for _ in range(sample_len)],
                "inference_logprobs": [
                    rng.uniform(-5.0, -0.1) for _ in range(sample_len)
                ],
                "temperatures": [1.0] * sample_len,
                "reward": None,
            }


def setup_rl_dataset(
    tokenizer: PreTrainedTokenizerBase,
    config: RLDataConfig,
    *,
    data_rank: int,
    data_world_size: int,
) -> IterableDataset[RLSample]:
    if config.source == "fake":
        return FakeRLDataset(
            seq_len=config.seq_len,
            vocab_size=config.fake_vocab_size,
            length_mode=config.fake_length,
            input_mode=config.fake_input_ids,
            seed=config.seed + data_rank,
        )
    records = load_rl_records(config)
    has_rl_targets = all(
        record.advantage is not None or record.reward is not None for record in records
    )
    if config.pack_sequences and has_rl_targets:
        return PackedRLDataset(
            records=records,
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            data_config=config,
            shuffle=config.shuffle,
            seed=config.seed,
            data_rank=data_rank,
            data_world_size=data_world_size,
        )
    return RLDataset(
        records=records,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        data_config=config,
        shuffle=config.shuffle,
        seed=config.seed,
        data_rank=data_rank,
        data_world_size=data_world_size,
    )


def setup_rl_dataloader(
    dataset: IterableDataset[RLSample],
    config: RLDataConfig,
    pad_token_id: int,
) -> StatefulDataLoader:
    return StatefulDataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        snapshot_every_n_steps=1,
        collate_fn=partial(collate_rl_batch, pad_token_id=pad_token_id),
    )

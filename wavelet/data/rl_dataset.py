from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any, NotRequired, TypedDict

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from wavelet.configs.rl_config import RLDataConfig
from wavelet.data.collation import IGNORE_INDEX
from wavelet.data.loading import Example
from wavelet.data.tokenization import build_sample


class RLSample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    target_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: NotRequired[list[float]]
    teacher_logprobs: NotRequired[list[float]]
    temperatures: list[float]
    reward: float | None


class RLBatch(TypedDict):
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


@dataclass
class RLExample:
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    advantage: float | list[float] | None
    reward: float | None
    target_completion: list[dict[str, str]] | None = None
    inference_logprobs: list[float] | None = None
    teacher_logprobs: list[float] | None = None
    temperatures: float | list[float] | None = None
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    source: str = "dataset"


def _coerce_advantages(
    value: float | list[float] | None,
    *,
    fallback_reward: float | None,
    num_trainable_tokens: int,
) -> list[float]:
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


def _normalize_rl_record(payload: dict[str, Any], config: RLDataConfig) -> RLExample:
    from wavelet.data import loading as loading_mod

    base: Example = loading_mod._normalize_record(payload, config)  # noqa: SLF001
    advantage = payload.get(config.advantage_column)
    reward_value = payload.get(config.reward_column)
    reward = None if reward_value is None else float(reward_value)
    inference_logprobs = payload.get(config.inference_logprobs_column)
    teacher_logprobs = payload.get(config.teacher_logprobs_column)
    temperatures = payload.get(config.temperature_column)

    if config.source == "fake":
        if advantage is None and reward is None:
            reward = 1.0
        if temperatures is None:
            temperatures = 1.0

    return RLExample(
        prompt=base.prompt,
        completion=base.completion,
        target_completion=base.completion,
        tools=base.tools,
        chat_template_kwargs=base.chat_template_kwargs,
        source=base.source,
        advantage=advantage,
        reward=reward,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        temperatures=temperatures,
    )


def load_rl_records(config: RLDataConfig) -> list[RLExample]:
    from wavelet.data import loading as loading_mod

    if config.source == "hf":
        payload_groups = loading_mod._load_hf_payload_groups(config)  # noqa: SLF001
    elif config.source == "fake":
        payload_groups = loading_mod._load_fake_payload_groups(config)  # noqa: SLF001
    else:
        payload_groups = loading_mod._load_local_payload_groups(config)  # noqa: SLF001

    payloads = loading_mod._mix_payload_groups(  # noqa: SLF001
        payload_groups,
        probabilities=config.probabilities,
        stopping_strategy=config.stopping_strategy,
        seed=config.seed,
    )
    rows = [_normalize_rl_record(payload, config) for payload in payloads]
    if config.max_examples is not None:
        rows = rows[: config.max_examples]
    if not rows:
        raise ValueError("No RL rows found for the configured data source.")
    return rows


@dataclass
class RLDataset(IterableDataset[RLSample]):
    records: list[RLExample]
    tokenizer: PreTrainedTokenizerBase
    seq_len: int
    data_config: RLDataConfig
    shuffle: bool = False
    seed: int = 0
    data_rank: int = 0
    data_world_size: int = 1

    def __post_init__(self) -> None:
        self.step = 0
        self.epoch = 0
        self.num_samples: dict[str, int] = defaultdict(int)
        self.num_tokens: dict[str, int] = defaultdict(int)
        self.skipped = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "num_samples": dict(self.num_samples),
            "num_tokens": dict(self.num_tokens),
            "skipped": self.skipped,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = int(state_dict["step"])
        self.epoch = int(state_dict["epoch"])
        self.num_samples = defaultdict(int, state_dict.get("num_samples", {}))
        self.num_tokens = defaultdict(int, state_dict.get("num_tokens", {}))
        self.skipped = int(state_dict.get("skipped", 0))

    def stats(self) -> dict[str, Any]:
        return {
            "samples": dict(self.num_samples),
            "tokens": dict(self.num_tokens),
            "skipped": self.skipped,
        }

    def __iter__(self) -> Iterator[RLSample]:
        num_examples = len(self.records)
        if num_examples == 0:
            return
        data_rank, data_world_size = self._effective_data_partition()
        while True:
            next_step = self.step + 1
            epoch = (next_step - 1) // num_examples
            if epoch != self.epoch:
                self.epoch = epoch
            sample_index = (next_step - 1) % num_examples
            self.step = next_step
            if (next_step - 1) % data_world_size != data_rank:
                continue

            record_index = self._order_for_epoch(epoch)[sample_index]
            record = self.records[record_index]
            base_sample = build_sample(
                Example(
                    prompt=record.prompt,
                    completion=record.completion,
                    tools=record.tools,
                    chat_template_kwargs=record.chat_template_kwargs,
                    source=record.source,
                ),
                self.tokenizer,
                seq_len=self.seq_len,
                loss_mask_config=self.data_config.loss_mask,
            )
            if base_sample is None:
                self.skipped += 1
                continue

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

            source = record.source or "dataset"
            self.num_samples[source] += 1
            self.num_tokens[source] += len(base_sample["input_ids"])

            sample: RLSample = {
                "input_ids": base_sample["input_ids"],
                "position_ids": base_sample["position_ids"],
                "target_ids": base_sample["target_ids"],
                "loss_mask": base_sample["loss_mask"],
                "advantages": advantages,
                "temperatures": temperatures,
                "reward": record.reward,
            }
            if inference_logprobs is not None:
                sample["inference_logprobs"] = inference_logprobs
            if teacher_logprobs is not None:
                sample["teacher_logprobs"] = teacher_logprobs
            yield sample

    def _order_for_epoch(self, epoch: int) -> list[int]:
        order = list(range(len(self.records)))
        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(order)
        return order

    def _effective_data_partition(self) -> tuple[int, int]:
        worker_info = get_worker_info()
        if worker_info is None:
            return self.data_rank, self.data_world_size
        return (
            self.data_rank * worker_info.num_workers + worker_info.id,
            self.data_world_size * worker_info.num_workers,
        )


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
            if values is None:
                expanded.append(default)
            else:
                expanded.append(float(values[trainable_index]))
            trainable_index += 1
        else:
            expanded.append(default)
    return expanded


def collate_rl_batch(
    batch: list[RLSample],
    *,
    pad_token_id: int,
) -> RLBatch:
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_out = []
    attention_mask_out = []
    position_ids_out = []
    target_ids_out = []
    labels_out = []
    loss_mask_out = []
    advantages_out = []
    rewards_out = []
    has_inference_out = []
    inference_logprobs_out = []
    has_teacher_out = []
    teacher_logprobs_out = []
    temperatures_out = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        mask = list(item["loss_mask"])
        num_trainable = sum(mask)
        if len(item["advantages"]) != num_trainable:
            raise ValueError(
                "advantages must align with the number of trainable tokens in the sample"
            )
        if len(item["temperatures"]) != num_trainable:
            raise ValueError(
                "temperatures must align with the number of trainable tokens in the sample"
            )

        inference_logprobs = item.get("inference_logprobs")
        if inference_logprobs is not None and len(inference_logprobs) != num_trainable:
            raise ValueError(
                "inference_logprobs must align with the number of trainable tokens in the sample"
            )

        teacher_logprobs = item.get("teacher_logprobs")
        if teacher_logprobs is not None and len(teacher_logprobs) != num_trainable:
            raise ValueError(
                "teacher_logprobs must align with the number of trainable tokens in the sample"
            )

        expanded_advantages = _expand_trainable_values(
            item["advantages"], mask=mask, default=0.0
        )
        expanded_inference = _expand_trainable_values(
            inference_logprobs, mask=mask, default=0.0
        )
        expanded_teacher = _expand_trainable_values(
            teacher_logprobs, mask=mask, default=0.0
        )
        expanded_temperatures = _expand_trainable_values(
            item["temperatures"], mask=mask, default=1.0
        )

        input_ids_out.append(
            torch.tensor(item["input_ids"] + [pad_token_id] * pad_len, dtype=torch.long)
        )
        attention_mask_out.append(
            torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long)
        )
        position_ids_out.append(
            torch.tensor(
                item["position_ids"] + list(range(seq_len, max_len)), dtype=torch.long
            )
        )
        target_ids_out.append(
            torch.tensor(item["target_ids"] + [0] * pad_len, dtype=torch.long)
        )
        labels = [
            target_id if trainable else IGNORE_INDEX
            for target_id, trainable in zip(item["target_ids"], mask, strict=True)
        ] + [IGNORE_INDEX] * pad_len
        labels_out.append(torch.tensor(labels, dtype=torch.long))
        loss_mask_out.append(torch.tensor(mask + [False] * pad_len, dtype=torch.bool))
        advantages_out.append(
            torch.tensor(expanded_advantages + [0.0] * pad_len, dtype=torch.float32)
        )
        rewards_out.append(
            torch.tensor(
                float("nan") if item["reward"] is None else float(item["reward"]),
                dtype=torch.float32,
            )
        )
        has_inference_out.append(
            torch.tensor(inference_logprobs is not None, dtype=torch.bool)
        )
        inference_logprobs_out.append(
            torch.tensor(expanded_inference + [0.0] * pad_len, dtype=torch.float32)
        )
        has_teacher_out.append(
            torch.tensor(teacher_logprobs is not None, dtype=torch.bool)
        )
        teacher_logprobs_out.append(
            torch.tensor(expanded_teacher + [0.0] * pad_len, dtype=torch.float32)
        )
        temperatures_out.append(
            torch.tensor(expanded_temperatures + [1.0] * pad_len, dtype=torch.float32)
        )

    return {
        "input_ids": torch.stack(input_ids_out),
        "attention_mask": torch.stack(attention_mask_out),
        "position_ids": torch.stack(position_ids_out),
        "target_ids": torch.stack(target_ids_out),
        "labels": torch.stack(labels_out),
        "loss_mask": torch.stack(loss_mask_out),
        "advantages": torch.stack(advantages_out),
        "rewards": torch.stack(rewards_out),
        "has_inference_logprobs": torch.stack(has_inference_out),
        "inference_logprobs": torch.stack(inference_logprobs_out),
        "has_teacher_logprobs": torch.stack(has_teacher_out),
        "teacher_logprobs": torch.stack(teacher_logprobs_out),
        "temperatures": torch.stack(temperatures_out),
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
    return RLDataset(
        records=load_rl_records(config),
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

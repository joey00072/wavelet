from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import torch
from datasets import Dataset, interleave_datasets, load_dataset
from torch import Tensor
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from wavelet.configs.sft import DataConfig, LossMaskConfig
from wavelet.data._stateful import StatefulDatasetMixin


@dataclass
class Example:
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    source: str = "dataset"


class Stats(TypedDict):
    samples: dict[str, int]
    tokens: dict[str, int]
    skipped: int


def _coerce_messages(value: Any, role: str | None) -> list[dict[str, str]]:
    if isinstance(value, str):
        if role is None:
            raise ValueError("String messages require an explicit role.")
        return [{"role": role, "content": value}]
    if isinstance(value, list):
        messages: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("Message items must be objects.")
            if "role" not in item and role is None:
                raise ValueError("Message items must include a role.")
            content = item.get("content")
            messages.append(
                {
                    **item,
                    "role": str(item.get("role", role)),
                    # Tool-call assistant messages commonly carry ``content: null``.
                    "content": "" if content is None else str(content),
                }
            )
        return messages
    raise ValueError("Prompt/completion must be either strings or message lists.")


def _deserialize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            normalized.append(message)
            continue
        normalized_tool_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                normalized_tool_calls.append(tool_call)
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(
                function.get("arguments"), str
            ):
                try:
                    arguments: object = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    arguments = function["arguments"]
                normalized_tool_calls.append(
                    {
                        **tool_call,
                        "function": {
                            **function,
                            "arguments": arguments,
                        },
                    }
                )
            else:
                normalized_tool_calls.append(tool_call)
        normalized.append({**message, "tool_calls": normalized_tool_calls})
    return normalized


def _strip_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            stripped.append({**message, "content": content.strip()})
        else:
            stripped.append(message)
    return stripped


def _merge_message_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for message in messages:
        thinking = message.get("thinking")
        content = message.get("content")
        if isinstance(thinking, str) and thinking.strip() and isinstance(content, str):
            merged.append(
                {
                    **message,
                    "content": f"<think>\n{thinking.strip()}\n</think>\n{content}",
                }
            )
        else:
            merged.append(message)
    return merged


def _split_messages_for_sft(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("messages rows must include at least one assistant message.")
    split_at = assistant_indices[-1]
    prompt = messages[:split_at]
    completion = messages[split_at:]
    if not completion:
        raise ValueError("messages rows must produce a non-empty completion.")
    return prompt, completion


def _prepend_system_prompt(
    messages: list[dict[str, str]],
    system_prompt: str | None,
) -> list[dict[str, str]]:
    if not system_prompt:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def _paths(value: Path | list[Path]) -> list[Path]:
    if isinstance(value, list):
        return [Path(path) for path in value]
    return [Path(value)]


def _load_payloads(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected {path} to contain a JSON list.")
        return [{**dict(row), "__source": path.name} for row in payload]

    payloads: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payloads.append({**dict(json.loads(stripped)), "__source": path.name})
    return payloads


def _load_local_payload_groups(config: DataConfig) -> list[list[dict[str, Any]]]:
    return [_load_payloads(path) for path in _paths(config.path)]


def _load_hf_payload_groups(config: DataConfig) -> list[list[dict[str, Any]]]:
    subsets_and_splits = _hf_subsets_and_splits(config)
    payload_groups: list[list[dict[str, Any]]] = []
    for subset, split in subsets_and_splits:
        dataset = load_dataset(config.hf_name, subset, split=split)
        payload_groups.append([dict(row) for row in dataset])
    return payload_groups


def _load_fake_payload_groups(config: DataConfig) -> list[list[dict[str, Any]]]:
    prompt = " ".join(f"tok{i}" for i in range(max(1, config.seq_len // 2)))
    completion = " ".join(
        f"tok{i}"
        for i in range(
            max(1, config.seq_len // 2),
            max(2, config.seq_len),
        )
    )
    payloads = [
        {
            config.prompt_column: prompt,
            config.completion_column: completion,
            "__source": "fake",
        }
        for _ in range(config.max_examples or config.batch_size)
    ]
    return [payloads]


def _hf_subsets_and_splits(
    config: DataConfig,
) -> list[tuple[str | None, str]]:
    if config.hf_subsets is None and config.hf_splits is None:
        return [(None, "train")]
    if config.hf_subsets is not None and config.hf_splits is None:
        return [(subset, "train") for subset in config.hf_subsets]
    if config.hf_subsets is None and config.hf_splits is not None:
        return [(None, split) for split in config.hf_splits]
    assert config.hf_subsets is not None and config.hf_splits is not None
    return list(zip(config.hf_subsets, config.hf_splits, strict=True))


def _mix_payload_groups(
    payload_groups: list[list[dict[str, Any]]],
    *,
    probabilities: list[float] | None,
    stopping_strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    if not payload_groups:
        return []
    if len(payload_groups) == 1:
        return payload_groups[0]
    if probabilities is not None and len(probabilities) != len(payload_groups):
        raise ValueError("probabilities must match the number of dataset groups")

    datasets_list = []
    for index, payloads in enumerate(payload_groups):
        if not payloads:
            continue
        dataset = Dataset.from_list(payloads)
        datasets_list.append(dataset)
    if not datasets_list:
        return []
    mixed = interleave_datasets(
        datasets_list,
        probabilities=probabilities,
        stopping_strategy=stopping_strategy,
        seed=seed,
    )
    return [dict(row) for row in mixed]


def normalize_record(payload: dict[str, Any], config: DataConfig) -> Example:
    """Normalize one raw data row into Wavelet's shared message format."""
    tools = payload.get(config.tools_column)
    if tools is not None and not isinstance(tools, list):
        raise ValueError("tools must be a list when provided.")

    chat_template_kwargs = payload.get(config.chat_template_kwargs_column)
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object when provided.")

    has_messages = (
        config.messages_column in payload
        and payload.get(config.messages_column) is not None
    )
    has_prompt_completion = (
        config.prompt_column in payload
        and config.completion_column in payload
        and payload.get(config.prompt_column) is not None
        and payload.get(config.completion_column) is not None
    )

    if has_messages and not has_prompt_completion:
        messages = _coerce_messages(payload[config.messages_column], None)
        messages = _strip_message_content(_deserialize_tool_calls(messages))
        if config.merge_messages_thinking:
            messages = _merge_message_thinking(messages)
        prompt, completion = _split_messages_for_sft(messages)
    else:
        if not has_prompt_completion:
            raise ValueError(
                "Each row must contain either prompt/completion columns or a "
                "messages column."
            )
        prompt = _strip_message_content(
            _deserialize_tool_calls(
                _coerce_messages(payload[config.prompt_column], "user")
            )
        )
        prompt = _prepend_system_prompt(prompt, config.system_prompt)
        completion = _strip_message_content(
            _deserialize_tool_calls(
                _coerce_messages(payload[config.completion_column], "assistant")
            )
        )

    return Example(
        prompt=prompt,
        completion=completion,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
        source=str(
            payload.get("__subset")
            or payload.get("__split")
            or payload.get("__source")
            or config.source
        ),
    )


def load_data_payloads(config: DataConfig) -> list[dict[str, Any]]:
    """Load and mix raw rows from the configured data sources."""
    if config.source == "hf":
        payload_groups = _load_hf_payload_groups(config)
    elif config.source == "fake":
        payload_groups = _load_fake_payload_groups(config)
    else:
        payload_groups = _load_local_payload_groups(config)
    return _mix_payload_groups(
        payload_groups,
        probabilities=config.probabilities,
        stopping_strategy=config.stopping_strategy,
        seed=config.seed,
    )


def load_records(config: DataConfig) -> list[Example]:
    payloads = load_data_payloads(config)
    rows = [normalize_record(payload, config) for payload in payloads]
    if config.max_examples is not None:
        rows = rows[: config.max_examples]
    if not rows:
        raise ValueError("No training rows found for the configured data source.")
    return rows


logger = logging.getLogger(__name__)


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]


def _token_ids(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if hasattr(value, "input_ids"):
        input_ids = value.input_ids
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return [int(item) for item in input_ids[0]]
            return [int(item) for item in input_ids]
    if hasattr(value, "ids"):
        return [int(item) for item in value.ids]
    raise TypeError(f"Unsupported tokenized value: {type(value)!r}")


def _should_mask(role: str, loss_mask_config: LossMaskConfig) -> bool:
    match role:
        case "system":
            return loss_mask_config.system
        case "user":
            return loss_mask_config.user
        case "assistant":
            return loss_mask_config.assistant
        case "tool":
            return loss_mask_config.tool
        case _:
            raise ValueError(f"Unsupported message role: {role}")


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if chat_template_kwargs is not None:
        kwargs.update(chat_template_kwargs)
    return _token_ids(tokenizer.apply_chat_template(messages, **kwargs))


def _build_loss_mask_fast(
    tokenizer: PreTrainedTokenizerBase,
    record: Example,
    loss_mask_config: LossMaskConfig,
) -> tuple[list[int], list[bool]] | None:
    """Fast path using return_assistant_tokens_mask (one tokenizer call).

    Only applicable for the default config (assistant=True, all others False).
    Returns None if the chat template doesn't support it, falling back to the
    incremental approach.
    """
    if not (
        loss_mask_config.assistant
        and not loss_mask_config.system
        and not loss_mask_config.user
        and not loss_mask_config.tool
    ):
        return None

    messages = record.prompt + record.completion
    # Per-turn step_loss_mask overrides are only honored by the incremental path.
    if any(not message.get("step_loss_mask", 1) for message in messages):
        return None
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
    }
    if record.tools is not None:
        kwargs["tools"] = record.tools
    if record.chat_template_kwargs is not None:
        kwargs.update(record.chat_template_kwargs)

    try:
        result = tokenizer.apply_chat_template(messages, **kwargs)
    except ValueError as exc:
        if "char_to_token() is not available" not in str(exc):
            raise
        return None
    raw_mask: list[int] = result.get("assistant_masks", [])  # type: ignore[union-attr]

    # If all zeros the template doesn't implement {% generation %} — fall back
    if not any(raw_mask):
        return None

    full_ids = _token_ids(result)
    loss_mask = [bool(m) for m in raw_mask]
    return full_ids, loss_mask


def _build_loss_mask(
    tokenizer: PreTrainedTokenizerBase,
    record: Example,
    loss_mask_config: LossMaskConfig,
) -> tuple[list[int], list[bool]]:
    """Incremental tokenization: apply the chat template to messages[0..i] at
    each step and measure the token delta since the previous step.

    This naturally handles Qwen3 and other models where a message's token
    representation depends on its position in the conversation (e.g. whether a
    system prompt is prepended, or whether thinking tokens are inserted). We
    avoid tokenizing turns in isolation so the rendered prefix remains exact."""
    messages = record.prompt + record.completion
    loss_mask: list[bool] = []
    prev_ids: list[int] = []
    prev_len = 0
    for index, message in enumerate(messages):
        # Parallel tool responses (e.g., qwen3 etc)
        if (
            message["role"] == "tool"
            and index + 1 < len(messages)
            and messages[index + 1]["role"] == "tool"
        ):
            continue
        # Rendering the generation prompt keeps the next assistant header out of
        # the trainable span regardless of which role precedes it.
        add_generation_prompt = (
            index + 1 < len(messages) and messages[index + 1]["role"] == "assistant"
        )
        current_ids = _apply_chat_template(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=add_generation_prompt,
            tools=record.tools,
            chat_template_kwargs=record.chat_template_kwargs,
        )
        if prev_ids and prev_ids != current_ids[:prev_len]:
            prefilled = _maybe_append_assistant_prefill_delta(
                tokenizer,
                messages,
                index=index,
                prev_ids=prev_ids,
                current_ids=current_ids,
                loss_mask=loss_mask,
                loss_mask_config=loss_mask_config,
                tools=record.tools,
                chat_template_kwargs=record.chat_template_kwargs,
            )
            if prefilled is None:
                raise ValueError(
                    "Chat template incremental tokenization mismatch at "
                    f"message index {index}."
                )
            prev_ids, prev_len = prefilled
            continue
        # Per-turn override: if the message carries step_loss_mask=0, mask the
        # whole turn regardless of role. This is useful for curriculum learning
        # or filtering bad turns in multi-turn data.
        role_mask = _should_mask(message["role"], loss_mask_config)
        turn_mask = bool(message.get("step_loss_mask", 1)) and role_mask
        header_start = len(current_ids)
        if turn_mask and add_generation_prompt:
            # The trailing generation prompt is the next turn's header, so it
            # must not be trained as part of this assistant message.
            content_ids = _apply_chat_template(
                tokenizer,
                messages[: index + 1],
                add_generation_prompt=False,
                tools=record.tools,
                chat_template_kwargs=record.chat_template_kwargs,
            )
            if (
                prev_len <= len(content_ids)
                and content_ids == current_ids[: len(content_ids)]
            ):
                header_start = len(content_ids)
        loss_mask.extend([turn_mask] * (header_start - prev_len))
        loss_mask.extend([False] * (len(current_ids) - header_start))
        prev_ids = current_ids
        prev_len = len(current_ids)
    return prev_ids, loss_mask


def _maybe_append_assistant_prefill_delta(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    index: int,
    prev_ids: list[int],
    current_ids: list[int],
    loss_mask: list[bool],
    loss_mask_config: LossMaskConfig,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> tuple[list[int], int] | None:
    """Handle templates whose generation prompt pre-fills assistant text.

    Unsloth's GRPO template appends ``<start_working_out>`` when
    ``add_generation_prompt=True``. vLLM then returns only tokens after that
    prefill, so the full assistant message no longer has ``prev_ids`` as a
    prefix. In that case we keep the prefilled prompt tokens and append only the
    assistant-message delta from the non-prefilled prompt.
    """
    if messages[index]["role"] != "assistant":
        return None

    prompt_ids = _apply_chat_template(
        tokenizer,
        messages[:index],
        add_generation_prompt=False,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_len = len(prompt_ids)
    prompt_text = tokenizer.decode(prompt_ids)
    if not tokenizer.decode(prev_ids).startswith(prompt_text):
        return None

    if current_ids[:prompt_len] == prompt_ids:
        delta = current_ids[prompt_len:]
    else:
        current_text = tokenizer.decode(current_ids)
        if not current_text.startswith(prompt_text):
            return None
        delta = _token_ids(
            tokenizer(
                current_text[len(prompt_text) :],
                add_special_tokens=False,
            )
        )
    if not delta:
        return None
    turn_mask = bool(messages[index].get("step_loss_mask", 1)) and _should_mask(
        messages[index]["role"],
        loss_mask_config,
    )
    loss_mask.extend([turn_mask] * len(delta))
    full_ids = prev_ids + delta
    return full_ids, len(full_ids)


def build_sample(
    record: Example,
    tokenizer: PreTrainedTokenizerBase,
    *,
    seq_len: int,
    loss_mask_config: LossMaskConfig,
) -> Sample | None:
    result = _build_loss_mask_fast(tokenizer, record, loss_mask_config)
    if result is None:
        result = _build_loss_mask(tokenizer, record, loss_mask_config)
    full_ids, loss_mask = result

    if tokenizer.eos_token_id not in full_ids:
        logger.warning(
            f"EOS token id {tokenizer.eos_token_id} not found in sample. "
            "Is something wrong with the chat template? Appending EOS token."
        )
        full_ids.append(tokenizer.eos_token_id)
        loss_mask.append(True)

    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]
    loss_mask = loss_mask[1:]

    input_ids = input_ids[:seq_len]
    target_ids = target_ids[:seq_len]
    loss_mask = loss_mask[:seq_len]

    if sum(loss_mask) == 0:
        logger.warning(
            "Skipping sample: no trainable tokens found within the context "
            f"window ({seq_len}). This prevents NaN loss."
        )
        return None

    assert len(input_ids) == len(loss_mask) == len(target_ids), (
        f"Length mismatch: {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
    )
    return {
        "input_ids": input_ids,
        "position_ids": list(range(len(input_ids))),
        "target_ids": target_ids,
        "loss_mask": loss_mask,
    }


def trainable_target_ids(sample: Sample) -> list[int]:
    return [
        int(target_id)
        for target_id, trainable in zip(
            sample["target_ids"],
            sample["loss_mask"],
            strict=True,
        )
        if trainable
    ]


def validate_token_logprob_alignment(
    sample: Sample,
    logprobs: list[float],
    *,
    field_name: str = "inference_logprobs",
) -> None:
    trainable_tokens = sum(bool(value) for value in sample["loss_mask"])
    if len(logprobs) != trainable_tokens:
        raise ValueError(
            f"{field_name} must align with trainable tokens "
            f"({len(logprobs)} != {trainable_tokens})."
        )


IGNORE_INDEX = -100


def collate_batch(
    batch: list[Sample],
    *,
    pad_token_id: int,
    include_attention_mask: bool = True,
) -> dict[str, Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_out = []
    attention_mask_out = []
    position_ids_out = []
    labels_out = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids_out.append(
            torch.tensor(
                item["input_ids"] + [pad_token_id] * pad_len,
                dtype=torch.long,
            )
        )
        if include_attention_mask:
            attention_mask_out.append(
                torch.tensor(
                    [1] * seq_len + [0] * pad_len,
                    dtype=torch.long,
                )
            )
        position_ids_out.append(
            torch.tensor(
                item["position_ids"] + list(range(seq_len, max_len)),
                dtype=torch.long,
            )
        )
        # Merge target_ids and loss_mask into labels with IGNORE_INDEX (-100).
        # Positions that don't contribute to the loss (role-masked or padding)
        # are set to -100 so CrossEntropyLoss(ignore_index=-100) skips them
        # automatically
        labels = [
            tid if mask else IGNORE_INDEX
            for tid, mask in zip(item["target_ids"], item["loss_mask"])
        ] + [IGNORE_INDEX] * pad_len
        labels_out.append(torch.tensor(labels, dtype=torch.long))

    batch_out = {
        "input_ids": torch.stack(input_ids_out),
        "position_ids": torch.stack(position_ids_out),
        "labels": torch.stack(labels_out),
    }
    if include_attention_mask:
        batch_out["attention_mask"] = torch.stack(attention_mask_out)
    return batch_out


class Batch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor


@dataclass
class SFTDataset(StatefulDatasetMixin[Example], IterableDataset[Sample]):
    def __init__(
        self,
        records: list[Example],
        tokenizer: PreTrainedTokenizerBase,
        *,
        seq_len: int,
        loss_mask_config: LossMaskConfig,
        shuffle: bool = False,
        seed: int = 0,
        data_rank: int = 0,
        data_world_size: int = 1,
        max_epochs_per_iteration: int | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.shuffle = shuffle
        self.seed = seed
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.max_epochs_per_iteration = max_epochs_per_iteration
        self._initialize_iteration_state()

    def __iter__(self) -> Iterator[Sample]:
        stop_step = None
        if self.max_epochs_per_iteration is not None:
            stop_step = self.step + len(self.records) * self.max_epochs_per_iteration
        for record_index in self._local_record_indexes(stop_step=stop_step):
            record = self.records[record_index]
            sample = build_sample(
                record,
                self.tokenizer,
                seq_len=self.seq_len,
                loss_mask_config=self.loss_mask_config,
            )

            if sample is None:
                self.skipped += 1
                continue

            self._record_sample(record.source, len(sample["input_ids"]))
            yield sample


class CatDataset(IterableDataset[Sample]):
    """Concatenative packing: fills seq_len exactly by concatenating samples.

    Zero padding waste. Each yielded chunk uses sequential position IDs
    [0, 1, ..., seq_len-1] so RoPE embeddings are monotonically increasing
    within the context window. This matches TRL's packing behavior.

    For true document-aware attention (no cross-doc leakage), use
    micro_batch_size=1 with Flash Attention 2 — FA2 detects reset
    position IDs and switches to varlen mode automatically.
    """

    def __init__(self, base: SFTDataset, seq_len: int) -> None:
        self.base = base
        self.seq_len = seq_len
        self._pending_input_ids: list[int] = []
        self._pending_target_ids: list[int] = []
        self._pending_loss_mask: list[bool] = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.base.state_dict(),
            "pending": {
                "input_ids": list(self._pending_input_ids),
                "target_ids": list(self._pending_target_ids),
                "loss_mask": list(self._pending_loss_mask),
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "dataset" not in state_dict:
            self.base.load_state_dict(state_dict)
            self._clear_pending()
            return

        self.base.load_state_dict(state_dict["dataset"])
        pending = state_dict.get("pending", {})
        self._pending_input_ids = [int(value) for value in pending.get("input_ids", [])]
        self._pending_target_ids = [
            int(value) for value in pending.get("target_ids", [])
        ]
        self._pending_loss_mask = [
            bool(value) for value in pending.get("loss_mask", [])
        ]
        if not (
            len(self._pending_input_ids)
            == len(self._pending_target_ids)
            == len(self._pending_loss_mask)
        ):
            raise ValueError("Packed SFT checkpoint has misaligned pending streams.")

    def stats(self) -> dict[str, Any]:
        return self.base.stats()

    def __iter__(self) -> Iterator[Sample]:
        for sample in self.base:
            self._pending_input_ids.extend(sample["input_ids"])
            self._pending_target_ids.extend(sample["target_ids"])
            self._pending_loss_mask.extend(sample["loss_mask"])

            while len(self._pending_input_ids) >= self.seq_len:
                packed = {
                    "input_ids": self._pending_input_ids[: self.seq_len],
                    "target_ids": self._pending_target_ids[: self.seq_len],
                    "loss_mask": self._pending_loss_mask[: self.seq_len],
                    "position_ids": list(range(self.seq_len)),
                }
                del self._pending_input_ids[: self.seq_len]
                del self._pending_target_ids[: self.seq_len]
                del self._pending_loss_mask[: self.seq_len]
                yield packed

    def _clear_pending(self) -> None:
        self._pending_input_ids = []
        self._pending_target_ids = []
        self._pending_loss_mask = []


def setup_dataset(
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
    *,
    data_rank: int,
    data_world_size: int,
    records: list[Example] | None = None,
    max_epochs_per_iteration: int | None = None,
) -> SFTDataset | CatDataset:
    base = SFTDataset(
        load_records(config) if records is None else records,
        tokenizer,
        seq_len=config.seq_len,
        loss_mask_config=config.loss_mask,
        shuffle=config.shuffle,
        seed=config.seed,
        data_rank=data_rank,
        data_world_size=data_world_size,
        max_epochs_per_iteration=max_epochs_per_iteration,
    )
    if config.pack_function == "cat":
        return CatDataset(base, config.seq_len)
    return base


def setup_dataloader(
    dataset: IterableDataset,
    config: DataConfig,
    pad_token_id: int,
) -> StatefulDataLoader:
    if config.pack_function in ("pad", "cat"):
        include_attention_mask = config.pack_function != "cat"
        return StatefulDataLoader(
            dataset,
            batch_size=config.micro_batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.num_workers > 0,
            snapshot_every_n_steps=1,
            collate_fn=partial(
                collate_batch,
                pad_token_id=pad_token_id,
                include_attention_mask=include_attention_mask,
            ),
        )
    raise NotImplementedError(
        f"Pack function '{config.pack_function}' not implemented yet"
    )

from __future__ import annotations

from typing import cast

import torch

from wavelet.data.collation import IGNORE_INDEX
from wavelet.data.rl_types import RLBatch, RLSample


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

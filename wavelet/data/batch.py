from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Self

import torch
from torch import Tensor


class TensorBatch:
    """A dataloader batch whose fields are tensors on the same device."""

    def to(self, device: torch.device) -> Self:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                setattr(self, field.name, value.to(device, non_blocking=True))
        return self

    def pin_memory(self) -> Self:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                setattr(self, field.name, value.pin_memory())
        return self


@dataclass(slots=True)
class SFTBatch(TensorBatch):
    input_ids: Tensor
    position_ids: Tensor
    labels: Tensor
    attention_mask: Tensor | None = None


@dataclass(slots=True)
class RLBatch(TensorBatch):
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    target_ids: Tensor
    loss_mask: Tensor
    advantages: Tensor
    rewards: Tensor
    has_inference_logprobs: Tensor
    inference_logprobs: Tensor
    has_teacher_logprobs: Tensor
    teacher_logprobs: Tensor
    temperatures: Tensor
    sample_counts: Tensor

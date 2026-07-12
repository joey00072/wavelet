from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(slots=True)
class LossOutput:
    """Differentiable loss and tensor metrics produced by a loss function."""

    loss: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class TrainOutput:
    """Result of one trainer micro-batch."""

    loss: LossOutput
    stepped: bool
    step: int
    micro_step: int
    metrics: dict[str, float] = field(default_factory=dict)
    skipped: bool = False

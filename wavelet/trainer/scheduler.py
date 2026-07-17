from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from wavelet.configs.sft import SchedulerConfig


def _resolve_warmup_steps(
    scheduler_config: SchedulerConfig,
    total_steps: int,
) -> int:
    if scheduler_config.warmup_steps is not None:
        return scheduler_config.warmup_steps
    return int(total_steps * scheduler_config.warmup_ratio)


def _resolve_decay_steps(
    scheduler_config: SchedulerConfig,
    total_steps: int,
    warmup_steps: int,
) -> int:
    if scheduler_config.decay_steps is not None and scheduler_config.decay_steps > 0:
        return scheduler_config.decay_steps
    return max(int(total_steps * scheduler_config.decay_ratio) - warmup_steps, 1)


def _resolve_min_lr_factor(
    min_lr: float,
    min_lr_factor: float | None,
    lr: float,
) -> float:
    if min_lr_factor is not None:
        return min_lr_factor
    if min_lr <= 0.0:
        return 1e-8
    return min_lr / lr


def setup_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig,
    *,
    total_steps: int,
    lr: float,
) -> LRScheduler:
    warmup_steps = _resolve_warmup_steps(scheduler_config, total_steps)
    decay_steps = _resolve_decay_steps(
        scheduler_config,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    if scheduler_config.type == "constant":
        return setup_constant_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=lr,
            min_lr=scheduler_config.min_lr,
            min_lr_factor=scheduler_config.min_lr_factor,
        )
    if scheduler_config.type == "linear":
        return setup_linear_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            lr=lr,
            min_lr=scheduler_config.min_lr,
            min_lr_factor=scheduler_config.min_lr_factor,
        )
    if scheduler_config.type == "cosine":
        return setup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=lr,
            min_lr=scheduler_config.min_lr,
            min_lr_factor=scheduler_config.min_lr_factor,
            decay_steps=decay_steps,
        )
    if scheduler_config.type == "sqrt":
        return setup_sqrt_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            lr=lr,
            min_lr=scheduler_config.min_lr,
            min_lr_factor=scheduler_config.min_lr_factor,
        )
    raise ValueError(f"Invalid scheduler type: {scheduler_config.type}")


def setup_constant_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    min_lr: float,
    min_lr_factor: float | None = None,
) -> LRScheduler:
    min_lr_factor = _resolve_min_lr_factor(
        min_lr,
        min_lr_factor,
        lr,
    )

    if total_steps <= 0:
        raise ValueError("Constant scheduler requires total_steps > 0")
    if warmup_steps <= 0:
        return ConstantLR(optimizer, factor=1.0)

    warmup_steps = min(warmup_steps, total_steps)
    schedulers: list[LRScheduler] = [
        LinearLR(
            optimizer,
            start_factor=min_lr_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
    ]
    milestones: list[int] = [warmup_steps]

    constant_steps = max(total_steps - warmup_steps, 1)
    schedulers.append(
        LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=constant_steps,
        )
    )
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def setup_linear_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    lr: float,
    min_lr: float,
    min_lr_factor: float | None = None,
) -> LRScheduler:
    min_lr_factor = _resolve_min_lr_factor(
        min_lr,
        min_lr_factor,
        lr,
    )

    if total_steps <= 0:
        raise ValueError("Linear scheduler requires total_steps > 0")
    if warmup_steps <= 0 and decay_steps <= 0:
        raise ValueError(
            "Linear scheduler requires warmup_steps or decay_steps to be set."
        )

    # Clamp warmup so it never exceeds total_steps (e.g. short benchmark runs)
    warmup_steps = min(warmup_steps, total_steps)

    effective_total = max(total_steps - 1, 1)

    decay_steps = max(min(decay_steps, effective_total), 1)

    schedulers: list[LRScheduler] = []
    milestones: list[int] = []

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=min_lr_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        milestones.append(warmup_steps)

    decay_start = min(total_steps - decay_steps, effective_total)
    if decay_steps > 0 and decay_start >= warmup_steps:
        constant_steps = decay_start - warmup_steps
        if constant_steps > 0:
            schedulers.append(
                LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=1.0,
                    total_iters=constant_steps,
                )
            )
            milestones.append(decay_start)
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=min_lr_factor,
                total_iters=max(decay_steps - 1, 1),
            )
        )

    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def setup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    min_lr: float,
    min_lr_factor: float | None = None,
    decay_steps: int | None = None,
) -> LRScheduler:
    if total_steps <= 0:
        raise ValueError("Cosine scheduler requires total_steps > 0")

    min_lr_factor = _resolve_min_lr_factor(min_lr, min_lr_factor, lr)

    schedulers: list[LRScheduler] = []
    milestones: list[int] = []

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=min_lr_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        milestones.append(warmup_steps)

    effective_total = max(total_steps - 1, 1)
    decay_steps = min(
        max(effective_total - warmup_steps, 1), decay_steps or effective_total
    )
    schedulers.append(
        CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=min_lr,
        )
    )

    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def setup_sqrt_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    lr: float,
    min_lr: float,
    min_lr_factor: float | None = None,
) -> LRScheduler:
    if total_steps <= 0:
        raise ValueError("SQRT scheduler requires total_steps > 0")

    min_lr_factor = _resolve_min_lr_factor(min_lr, min_lr_factor, lr)
    decay_steps = max(decay_steps, 1)

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")

    schedulers: list[LRScheduler] = []
    milestones: list[int] = []

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=min_lr_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        milestones.append(warmup_steps)

    decay_span = max(total_steps - warmup_steps - 1, 1)

    def _sqrt_factor(step: int) -> float:
        if step <= 0:
            return 1.0
        t = min(step / decay_span, 1.0)
        return min_lr_factor + (1.0 - min_lr_factor) * math.sqrt(max(1.0 - t, 0.0))

    schedulers.append(
        LambdaLR(
            optimizer,
            _sqrt_factor,
        )
    )

    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

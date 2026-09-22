"""Shared `nn.Module` helpers with no trainer dependency."""

from __future__ import annotations

from torch import nn

_TRAINING_WRAPPER_SEGMENTS = frozenset(
    {"_fsdp_wrapped_module", "_checkpoint_wrapped_module"}
)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Peel off DDP/FSDP `.module` wrapper attributes to reach the base model."""
    current = model
    while hasattr(current, "module"):
        current = current.module  # type: ignore[assignment]
    return current


def strip_training_wrapper_segments(key: str) -> str:
    """Drop FSDP/activation-checkpoint wrapper segments from a state-dict key."""
    return ".".join(
        segment
        for segment in key.split(".")
        if segment not in _TRAINING_WRAPPER_SEGMENTS
    )

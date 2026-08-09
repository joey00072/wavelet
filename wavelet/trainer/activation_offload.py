"""Compatibility exports for activation offloading in ``trainer.optim``."""

from wavelet.trainer.optim import (
    OffloadActivations,
    maybe_activation_offloading,
    patch_model_gradient_checkpointing,
)

__all__ = [
    "OffloadActivations",
    "maybe_activation_offloading",
    "patch_model_gradient_checkpointing",
]

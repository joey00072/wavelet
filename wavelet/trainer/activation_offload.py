"""Compatibility exports for activation offloading in ``trainer.optim``."""

from wavelet.trainer.optim import (
    OffloadActivations,
    maybe_activation_offloading,
)

__all__ = [
    "OffloadActivations",
    "maybe_activation_offloading",
]

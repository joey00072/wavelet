"""Compatibility exports for schedulers now owned by ``trainer.optim``."""

from wavelet.trainer.optim import (
    setup_constant_scheduler,
    setup_cosine_scheduler,
    setup_linear_scheduler,
    setup_scheduler,
    setup_sqrt_scheduler,
)

__all__ = [
    "setup_constant_scheduler",
    "setup_cosine_scheduler",
    "setup_linear_scheduler",
    "setup_scheduler",
    "setup_sqrt_scheduler",
]

"""Compatibility alias for LoRA helpers now owned by trainer.model."""

import sys

from wavelet.trainer import model as _canonical

sys.modules[__name__] = _canonical

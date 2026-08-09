"""Compatibility alias for :mod:`wavelet.trainer.distributed`."""

import sys
from wavelet.trainer import distributed as _canonical

sys.modules[__name__] = _canonical

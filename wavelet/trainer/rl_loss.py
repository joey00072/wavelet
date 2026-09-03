"""Compatibility alias for consolidated trainer losses."""

import sys

from wavelet.trainer import losses as _losses

sys.modules[__name__] = _losses

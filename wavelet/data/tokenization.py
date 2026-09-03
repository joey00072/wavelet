"""Compatibility alias for :mod:`wavelet.data.sft`."""

import sys

from wavelet.data import sft as _sft

sys.modules[__name__] = _sft

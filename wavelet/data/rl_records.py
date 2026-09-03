"""Compatibility alias for :mod:`wavelet.data.rl`."""

import sys

from wavelet.data import rl as _rl

sys.modules[__name__] = _rl

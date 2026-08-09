"""Compatibility alias for diagnostics consolidated in :mod:`wavelet.debug`."""

import sys

from wavelet import debug as _debug

sys.modules[__name__] = _debug

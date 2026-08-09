"""Compatibility alias for observability consolidated in :mod:`wavelet.monitor`."""

import sys

from wavelet import monitor as _monitor

sys.modules[__name__] = _monitor

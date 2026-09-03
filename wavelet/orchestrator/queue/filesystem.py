"""Compatibility alias for :mod:`wavelet.transport.queue`."""

import sys

from wavelet.transport import queue as _queue

sys.modules[__name__] = _queue

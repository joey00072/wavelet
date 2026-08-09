"""Compatibility alias for policy transport mechanics."""

import sys

from wavelet.transport import policy as _policy

sys.modules[__name__] = _policy

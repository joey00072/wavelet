"""Compatibility alias for verifier APIs consolidated into the scheduler."""

import sys

from wavelet.orchestrator import scheduler as _scheduler

sys.modules[__name__] = _scheduler

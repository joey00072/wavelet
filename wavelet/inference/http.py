"""Compatibility exports for the consolidated inference engine."""

import sys

from wavelet.inference.engine import (
    HTTPPolicyInferenceEngine,
    _shift_completion_sample,
)

__all__ = ["HTTPPolicyInferenceEngine", "_shift_completion_sample"]

sys.modules[__name__] = sys.modules["wavelet.inference.engine"]

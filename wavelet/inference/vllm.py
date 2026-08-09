"""Compatibility exports for the consolidated inference engine."""

import sys

from wavelet.inference.engine import (
    VLLMPolicyInferenceEngine,
    _OpenAIBatchRequest,
    _sampling_params_type,
    _vllm_dtype,
)

__all__ = [
    "VLLMPolicyInferenceEngine",
    "_OpenAIBatchRequest",
    "_sampling_params_type",
    "_vllm_dtype",
]

sys.modules[__name__] = sys.modules["wavelet.inference.engine"]

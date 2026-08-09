"""Compatibility wrapper; implementation lives in wavelet.data.rl."""

from wavelet.data.rl import (
    RLSample as RLSample,
    RLBatch as RLBatch,
    RLExample as RLExample,
    rl_example_to_payload as rl_example_to_payload,
    rl_example_from_payload as rl_example_from_payload,
    rl_examples_to_payload as rl_examples_to_payload,
    rl_examples_from_payload as rl_examples_from_payload,
)

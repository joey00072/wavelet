"""Compatibility wrapper; implementation lives in wavelet.data.rl."""

from wavelet.data.rl import (
    _validate_trainable_values as _validate_trainable_values,
    _expand_trainable_values as _expand_trainable_values,
    collate_rl_batch as collate_rl_batch,
    _append_sample as _append_sample,
)

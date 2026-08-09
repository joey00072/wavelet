"""Compatibility wrapper; implementation lives in wavelet.data.rl."""

from wavelet.data.rl import (
    trainable_token_count as trainable_token_count,
    trainable_sequence_count as trainable_sequence_count,
    pack_samples as pack_samples,
    pad_bins_for_distribution as pad_bins_for_distribution,
    _merge_samples as _merge_samples,
    _pad_token_streams as _pad_token_streams,
    _zero_loss_copy as _zero_loss_copy,
)

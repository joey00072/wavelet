"""Compatibility wrapper; implementation lives in wavelet.data.rl."""

from wavelet.data.rl import (
    _merge_samples as _merge_samples,
)
from wavelet.data.rl import (
    _pad_token_streams as _pad_token_streams,
)
from wavelet.data.rl import (
    _zero_loss_copy as _zero_loss_copy,
)
from wavelet.data.rl import (
    pack_samples as pack_samples,
)
from wavelet.data.rl import (
    pad_bins_for_distribution as pad_bins_for_distribution,
)
from wavelet.data.rl import (
    trainable_sequence_count as trainable_sequence_count,
)
from wavelet.data.rl import (
    trainable_token_count as trainable_token_count,
)

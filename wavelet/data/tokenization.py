"""Compatibility wrapper; implementation lives in wavelet.data.sft."""

from wavelet.data.sft import (
    Sample as Sample,
)
from wavelet.data.sft import (
    _apply_chat_template as _apply_chat_template,
)
from wavelet.data.sft import (
    _build_loss_mask as _build_loss_mask,
)
from wavelet.data.sft import (
    _build_loss_mask_fast as _build_loss_mask_fast,
)
from wavelet.data.sft import (
    _maybe_append_assistant_prefill_delta as _maybe_append_assistant_prefill_delta,
)
from wavelet.data.sft import (
    _should_mask as _should_mask,
)
from wavelet.data.sft import (
    _token_ids as _token_ids,
)
from wavelet.data.sft import (
    build_sample as build_sample,
)
from wavelet.data.sft import (
    logger as logger,
)
from wavelet.data.sft import (
    trainable_target_ids as trainable_target_ids,
)
from wavelet.data.sft import (
    validate_token_logprob_alignment as validate_token_logprob_alignment,
)

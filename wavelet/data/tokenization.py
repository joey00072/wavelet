"""Compatibility wrapper; implementation lives in wavelet.data.sft."""

from wavelet.data.sft import (
    logger as logger,
    Sample as Sample,
    _token_ids as _token_ids,
    _should_mask as _should_mask,
    _apply_chat_template as _apply_chat_template,
    _build_loss_mask_fast as _build_loss_mask_fast,
    _build_loss_mask as _build_loss_mask,
    _maybe_append_assistant_prefill_delta as _maybe_append_assistant_prefill_delta,
    build_sample as build_sample,
    trainable_target_ids as trainable_target_ids,
    validate_token_logprob_alignment as validate_token_logprob_alignment,
)

from __future__ import annotations

import pytest

from wavelet.configs.sft import LossMaskConfig
from wavelet.data.loading import Example
from wavelet.data.tokenization import (
    build_sample,
    trainable_target_ids,
    validate_token_logprob_alignment,
)
from wavelet.trainer.debug import build_debug_tokenizer


def _tokenizer():
    return build_debug_tokenizer(model_max_length=256)


def _trainable_text(sample) -> str:
    return _tokenizer().decode(trainable_target_ids(sample))


def test_alignment_masks_only_assistant_completion_tokens() -> None:
    sample = build_sample(
        Example(
            prompt=[
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "hi"},
            ],
            completion=[{"role": "assistant", "content": "ok"}],
            source="test",
        ),
        _tokenizer(),
        seq_len=128,
        loss_mask_config=LossMaskConfig(),
    )

    assert sample is not None
    assert _trainable_text(sample) == "ok<eos>"
    validate_token_logprob_alignment(sample, [-0.1, -0.2, -0.3])


def test_alignment_masks_tool_context_and_trains_final_assistant_turn() -> None:
    sample = build_sample(
        Example(
            prompt=[
                {"role": "user", "content": "run"},
                {"role": "assistant", "content": "tool call", "step_loss_mask": 0},
                {"role": "tool", "content": "result"},
            ],
            completion=[{"role": "assistant", "content": "done"}],
            source="test",
        ),
        _tokenizer(),
        seq_len=256,
        loss_mask_config=LossMaskConfig(),
    )

    assert sample is not None
    assert _trainable_text(sample) == "done<eos>"


def test_alignment_can_train_tool_tokens_when_configured() -> None:
    sample = build_sample(
        Example(
            prompt=[{"role": "user", "content": "lookup"}],
            completion=[
                {"role": "assistant", "content": "call", "step_loss_mask": 0},
                {"role": "tool", "content": "42"},
                {"role": "assistant", "content": "answer"},
            ],
            source="test",
        ),
        _tokenizer(),
        seq_len=256,
        loss_mask_config=LossMaskConfig(tool=True),
    )

    assert sample is not None
    assert "42<eos>" in _trainable_text(sample)
    assert _trainable_text(sample).endswith("answer<eos>")


def test_alignment_keeps_trainable_prefix_when_eos_is_beyond_context() -> None:
    sample = build_sample(
        Example(
            prompt=[{"role": "user", "content": "x"}],
            completion=[{"role": "assistant", "content": "y" * 128}],
            source="test",
        ),
        _tokenizer(),
        seq_len=32,
        loss_mask_config=LossMaskConfig(),
    )

    assert sample is not None
    assert len(sample["input_ids"]) == 32
    assert sum(sample["loss_mask"]) > 0
    assert _tokenizer().eos_token_id not in trainable_target_ids(sample)


def test_validate_token_logprob_alignment_rejects_mismatch() -> None:
    sample = build_sample(
        Example(
            prompt=[{"role": "user", "content": "hi"}],
            completion=[{"role": "assistant", "content": "ok"}],
            source="test",
        ),
        _tokenizer(),
        seq_len=128,
        loss_mask_config=LossMaskConfig(),
    )

    assert sample is not None
    with pytest.raises(ValueError, match="inference_logprobs"):
        validate_token_logprob_alignment(sample, [-0.1])

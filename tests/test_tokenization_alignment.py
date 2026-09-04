from __future__ import annotations

import pytest
from transformers import BatchEncoding

from wavelet.configs.sft import LossMaskConfig
from wavelet.data.loading import Example
from wavelet.data.sft import _build_loss_mask_fast
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


class _AssistantMaskTokenizer:
    """Fake tokenizer whose template supports ``return_assistant_tokens_mask``.

    Non-assistant messages render as two tokens; assistant messages render as
    the generation-prompt token (8), one content token, and eos (9).
    """

    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        ids: list[int] = []
        assistant_mask: list[int] = []
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                ids.extend([8, index, 9])
                assistant_mask.extend([0, 1, 1])
            else:
                ids.extend([index, index])
                assistant_mask.extend([0, 0])
        if kwargs.get("add_generation_prompt"):
            ids.append(8)
            assistant_mask.append(0)
        if kwargs.get("return_assistant_tokens_mask"):
            return BatchEncoding({"input_ids": ids, "assistant_masks": assistant_mask})
        return ids


def test_fast_loss_mask_defers_to_incremental_path_for_step_loss_mask() -> None:
    tokenizer = _AssistantMaskTokenizer()
    record = Example(
        prompt=[{"role": "user", "content": "run"}],
        completion=[{"role": "assistant", "content": "done"}],
    )

    fast = _build_loss_mask_fast(tokenizer, record, LossMaskConfig())  # type: ignore[arg-type]
    assert fast is not None
    assert fast[1] == [False, False, False, True, True]

    record.completion[0]["step_loss_mask"] = 0  # type: ignore[index]
    assert _build_loss_mask_fast(tokenizer, record, LossMaskConfig()) is None  # type: ignore[arg-type]


def test_step_loss_mask_is_honored_when_template_supports_assistant_masks() -> None:
    tokenizer = _AssistantMaskTokenizer()
    sample = build_sample(
        Example(
            prompt=[
                {"role": "user", "content": "run"},
                {"role": "assistant", "content": "tool call", "step_loss_mask": 0},
                {"role": "user", "content": "result"},
            ],
            completion=[{"role": "assistant", "content": "done"}],
        ),
        tokenizer,  # type: ignore[arg-type]
        seq_len=64,
        loss_mask_config=LossMaskConfig(),
    )

    assert sample is not None
    assert trainable_target_ids(sample) == [3, 9]

from __future__ import annotations

import torch

from wavelet.configs.sft import SFTConfig
from wavelet.data.sft import _coerce_messages
from wavelet.trainer.trainer import SFTTrainer


def test_sft_compute_loss_with_no_valid_labels_supports_backward() -> None:
    trainer = SFTTrainer(SFTConfig())
    logits = torch.randn(1, 4, 10, requires_grad=True)
    labels = torch.full((1, 4), -100, dtype=torch.long)

    output = trainer.compute_loss(logits, labels)
    output.loss.backward()

    assert output.loss.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_coerce_messages_maps_null_content_to_empty_string() -> None:
    tool_calls = [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]

    messages = _coerce_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
        ],
        None,
    )

    assert messages[0]["content"] == "hi"
    assert messages[1]["content"] == ""
    assert messages[1]["tool_calls"] == tool_calls

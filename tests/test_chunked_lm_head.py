from __future__ import annotations

import torch

from wavelet.trainer.lm_head import ChunkedLogprobLmHead
from wavelet.trainer.rl_loss import selective_log_softmax


def test_chunked_lm_head_matches_full_logits_logprobs() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, dtype=torch.float32, requires_grad=True)
    full_head = torch.nn.Linear(7, 11, bias=False)
    chunked_head = ChunkedLogprobLmHead(7, 11, chunk_size=3)
    chunked_head.weight = full_head.weight
    labels = torch.randint(0, 11, (2, 5))
    temperatures = torch.rand(2, 5) + 0.5

    full_logits = full_head(hidden) / temperatures.unsqueeze(-1)
    expected = selective_log_softmax(full_logits, labels)
    expected_entropy = torch.distributions.Categorical(logits=full_logits).entropy()
    output = chunked_head(hidden, labels=labels, temperature=temperatures)
    actual = output["logprobs"]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(output["entropy"], expected_entropy)

    expected_loss = expected.sum()
    actual_loss = actual.sum()
    expected_loss.backward(retain_graph=True)
    expected_hidden_grad = hidden.grad.detach().clone()
    expected_weight_grad = full_head.weight.grad.detach().clone()
    hidden.grad = None
    full_head.weight.grad = None

    actual_loss.backward()

    torch.testing.assert_close(hidden.grad, expected_hidden_grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        full_head.weight.grad,
        expected_weight_grad,
        atol=1e-5,
        rtol=1e-5,
    )


def test_chunked_lm_head_skips_frozen_weight_gradient() -> None:
    torch.manual_seed(1)
    hidden = torch.randn(1, 4, 5, dtype=torch.float32, requires_grad=True)
    chunked_head = ChunkedLogprobLmHead(5, 13, chunk_size=2)
    chunked_head.weight.requires_grad_(False)
    labels = torch.randint(0, 13, (1, 4))
    temperatures = torch.ones(1, 4)

    output = chunked_head(hidden, labels=labels, temperature=temperatures)
    output["logprobs"].sum().backward()

    assert hidden.grad is not None
    assert chunked_head.weight.grad is None

from __future__ import annotations

import pytest
import torch

from wavelet.trainer.losses import (
    ChunkedLogprobLmHead,
    selective_log_softmax,
    selective_log_softmax_with_sampling_mask,
)


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


def test_sampling_mask_replay_normalizes_only_vllm_support() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, -1.0]]])
    labels = torch.tensor([[3, 0]])
    mask_ids = torch.tensor([[[1, 3], [0, 2]]])
    mask_lengths = torch.tensor([[2, 0]])

    actual, entropy = selective_log_softmax_with_sampling_mask(
        logits, labels, mask_ids, mask_lengths
    )
    expected_first = torch.log_softmax(torch.tensor([2.0, 4.0]), dim=0)[1]
    expected_second = torch.log_softmax(logits[0, 1], dim=0)[0]
    assert actual[0].tolist() == pytest.approx(
        [expected_first.item(), expected_second.item()], abs=1e-6
    )
    assert entropy[0, 0].item() == pytest.approx(
        torch.distributions.Categorical(logits=logits[0, 0]).entropy().item()
    )


def test_chunked_lm_head_sampling_mask_matches_forward_and_backward() -> None:
    torch.manual_seed(4)
    hidden = torch.randn(1, 4, 5, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = torch.randn(9, 5, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    labels = torch.tensor([[1, 4, 2, 8]])
    temperatures = torch.tensor([[0.7, 1.1, 0.9, 1.3]])
    mask_ids = torch.tensor([[[1, 3, 7], [0, 4, 0], [0, 0, 0], [8, 0, 0]]])
    mask_lengths = torch.tensor([[3, 2, 0, 1]])

    reference_logits = reference_hidden @ reference_weight.t()
    reference_logits = reference_logits / temperatures.unsqueeze(-1)
    support_logits = reference_logits.clone()
    for position, length in enumerate(mask_lengths[0].tolist()):
        if length:
            allowed = mask_ids[0, position, :length]
            blocked = torch.ones(9, dtype=torch.bool)
            blocked[allowed] = False
            support_logits[0, position, blocked] = float("-inf")
    reference_logprobs = torch.gather(
        support_logits.log_softmax(dim=-1), -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    reference_logprobs.sum().backward()

    head = ChunkedLogprobLmHead(5, 9, chunk_size=2)
    head.weight = torch.nn.Parameter(weight.detach().clone())
    actual = head(
        hidden,
        labels=labels,
        temperature=temperatures,
        sampling_mask_ids=mask_ids,
        sampling_mask_lengths=mask_lengths,
    )
    actual["logprobs"].sum().backward()

    torch.testing.assert_close(actual["logprobs"], reference_logprobs)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        head.weight.grad, reference_weight.grad, atol=1e-5, rtol=1e-5
    )

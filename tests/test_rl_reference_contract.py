from __future__ import annotations

from collections import defaultdict

import pytest
import torch
from torch import Tensor

from wavelet.configs.rl_config import RLConfig, RLLossConfig
from wavelet.orchestrator.advantage import group_reward_advantages
from wavelet.orchestrator.schedule import chunks_per_step, rollout_groups_for_chunk
from wavelet.trainer.losses import compute_loss
from wavelet.transport.queue import RolloutChunkAccumulator

UPSTREAM_AUDITED_COMMIT = "ef9dea17815756f21bd20028fd8a8dcf29319763"


def _upstream_grpo_reference(rewards: Tensor, group_ids: Tensor) -> Tensor:
    """Independently encode unnormalized reward-minus-group-mean GRPO."""
    grouped: dict[int, list[Tensor]] = defaultdict(list)
    for reward, group_id in zip(rewards, group_ids, strict=True):
        grouped[int(group_id.item())].append(reward)
    means = {
        group_id: torch.stack(values).mean() for group_id, values in grouped.items()
    }
    return torch.stack(
        [
            reward - means[int(group_id.item())]
            for reward, group_id in zip(rewards, group_ids, strict=True)
        ]
    )


def _upstream_ipo_reference(
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    advantages: Tensor,
    loss_mask: Tensor,
    *,
    epsilon: float,
    adv_tau: float,
    kl_tau: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Independently encode the IPO equation and symmetric trust region."""
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    probability_delta = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    masked_high = probability_delta > epsilon
    masked_low = probability_delta < -epsilon
    masked = masked_high | masked_low
    keep_mask = loss_mask & ~masked

    policy_gradient = keep_mask * (adv_tau * advantages) * importance_ratio
    kl = loss_mask * log_importance_ratio.square()
    loss = (-policy_gradient + kl_tau * kl).sum() / loss_mask.sum()
    return loss, masked, masked_high, masked_low


def _frozen_batch(
    lora_adapter: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dtype = lora_adapter.dtype
    target_probabilities = torch.tensor(
        [
            [0.35, 0.25, 0.55, 0.10],
            [0.62, 0.18, 0.42, 0.78],
            [0.28, 0.73, 0.15, 0.48],
            [0.81, 0.33, 0.58, 0.21],
        ],
        dtype=dtype,
    )
    other_probabilities = (1.0 - target_probabilities) / 2.0
    base_logits = torch.stack(
        [
            target_probabilities.log(),
            other_probabilities.log(),
            other_probabilities.log(),
        ],
        dim=-1,
    )
    lora_features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.5, 1.0]],
            [[-1.0, 0.5], [0.7, 0.2], [0.0, -1.0], [1.0, 1.0]],
            [[0.4, 0.8], [-0.6, 0.3], [1.0, -0.4], [0.2, -0.7]],
            [[-0.8, -0.2], [0.3, 1.0], [-0.2, 0.6], [0.9, -0.5]],
        ],
        dtype=dtype,
    )
    target_logit_update = torch.einsum("btr,r->bt", lora_features, lora_adapter)
    target_direction = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    trainer_logits = base_logits + target_logit_update.unsqueeze(-1) * target_direction
    trainer_logprobs = trainer_logits.log_softmax(dim=-1)[..., 0]

    inference_logprobs = torch.tensor(
        [
            [0.20, 0.28, 0.40, 0.22],
            [0.48, 0.30, 0.39, 0.64],
            [0.31, 0.58, 0.29, 0.50],
            [0.65, 0.45, 0.55, 0.08],
        ],
        dtype=dtype,
    ).log()
    rewards = torch.tensor([0.2, 0.8, 0.7, 0.1], dtype=dtype)
    group_ids = torch.tensor([11, 11, 29, 29])
    loss_mask = torch.tensor(
        [
            [False, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [True, True, True, False],
        ]
    )
    return (
        trainer_logits,
        trainer_logprobs,
        inference_logprobs,
        rewards,
        group_ids,
        loss_mask,
    )


def _wavelet_group_advantages(rewards: Tensor, group_ids: Tensor) -> Tensor:
    advantages = torch.empty_like(rewards)
    for group_id in group_ids.unique(sorted=True):
        indexes = (group_ids == group_id).nonzero(as_tuple=False).flatten()
        values = [float(rewards[index].item()) for index in indexes]
        scored = group_reward_advantages(values, normalize=False)
        advantages[indexes] = torch.tensor(scored, dtype=rewards.dtype)
    return advantages


def test_frozen_upstream_grpo_ipo_and_adamw_contract() -> None:
    """Protect the audited upstream GRPO/IPO contract without importing it."""
    dtype = torch.float64
    initial_adapter = torch.tensor([0.02, -0.015], dtype=dtype)
    reference_adapter = torch.nn.Parameter(initial_adapter.clone())
    wavelet_adapter = torch.nn.Parameter(initial_adapter.clone())

    (
        _,
        reference_logprobs,
        inference_logprobs,
        rewards,
        group_ids,
        loss_mask,
    ) = _frozen_batch(reference_adapter)
    reference_rollout_advantages = _upstream_grpo_reference(rewards, group_ids)
    reference_advantages = reference_rollout_advantages[:, None].expand_as(
        reference_logprobs
    )
    reference_loss, reference_mask, reference_high, reference_low = (
        _upstream_ipo_reference(
            reference_logprobs,
            inference_logprobs,
            reference_advantages,
            loss_mask,
            epsilon=0.1,
            adv_tau=1.0,
            kl_tau=0.001,
        )
    )

    (
        frozen_logits,
        wavelet_logprobs,
        wavelet_inference_logprobs,
        wavelet_rewards,
        wavelet_group_ids,
        wavelet_loss_mask,
    ) = _frozen_batch(wavelet_adapter)
    wavelet_rollout_advantages = _wavelet_group_advantages(
        wavelet_rewards, wavelet_group_ids
    )
    wavelet_advantages = wavelet_rollout_advantages[:, None].expand_as(wavelet_logprobs)
    wavelet_output = compute_loss(
        wavelet_logprobs,
        wavelet_inference_logprobs,
        None,
        wavelet_advantages,
        wavelet_loss_mask,
        RLLossConfig(
            type="ipo",
            ipo_epsilon=0.1,
            adv_tau=1.0,
            kl_tau=0.001,
            normalization="token",
        ),
    )

    expected_mask = torch.tensor(
        [
            [True, False, True, True],
            [True, True, False, True],
            [False, True, True, False],
            [True, True, False, True],
        ]
    )
    assert UPSTREAM_AUDITED_COMMIT.startswith("ef9dea")
    assert frozen_logits.shape == (4, 4, 3)
    assert torch.equal(reference_mask, expected_mask)
    torch.testing.assert_close(
        wavelet_rollout_advantages,
        reference_rollout_advantages,
        rtol=0.0,
        atol=1e-15,
    )
    torch.testing.assert_close(
        wavelet_output.loss,
        reference_loss,
        rtol=1e-12,
        atol=1e-12,
    )
    assert wavelet_output.metrics["is_masked"].item() == pytest.approx(
        float(reference_mask[loss_mask].float().mean().item()),
        rel=0.0,
        abs=torch.finfo(torch.float32).eps,
    )
    assert wavelet_output.metrics["is_masked_high"].item() == pytest.approx(
        float(reference_high[loss_mask].float().mean().item()),
        rel=0.0,
        abs=torch.finfo(torch.float32).eps,
    )
    assert wavelet_output.metrics["is_masked_low"].item() == pytest.approx(
        float(reference_low[loss_mask].float().mean().item()),
        rel=0.0,
        abs=torch.finfo(torch.float32).eps,
    )

    reference_logprob_grad = torch.autograd.grad(
        reference_loss, reference_logprobs, retain_graph=True
    )[0]
    wavelet_logprob_grad = torch.autograd.grad(
        wavelet_output.loss, wavelet_logprobs, retain_graph=True
    )[0]
    torch.testing.assert_close(
        wavelet_logprob_grad,
        reference_logprob_grad,
        rtol=1e-12,
        atol=1e-12,
    )

    reference_loss.backward()
    wavelet_output.loss.backward()
    torch.testing.assert_close(
        wavelet_adapter.grad,
        reference_adapter.grad,
        rtol=1e-12,
        atol=1e-12,
    )

    reference_optimizer = torch.optim.AdamW(
        [reference_adapter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    wavelet_optimizer = torch.optim.AdamW(
        [wavelet_adapter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    reference_optimizer.step()
    wavelet_optimizer.step()
    torch.testing.assert_close(
        wavelet_adapter.detach() - initial_adapter,
        reference_adapter.detach() - initial_adapter,
        rtol=1e-12,
        atol=1e-12,
    )


def test_reference_batch_is_two_128_rollout_chunks_per_optimizer_step() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 16,
            "rollouts_per_example": 16,
            "rollout_chunk_examples": 8,
            "max_async_level": 3,
        }
    )
    accumulator = RolloutChunkAccumulator()

    assert chunks_per_step(config) == 2
    assert rollout_groups_for_chunk(config, 0) == 8
    assert rollout_groups_for_chunk(config, 1) == 8

    accumulator.mark_loaded(rows=128, chunks=1, loss_scale=128.0)
    assert accumulator.accumulated_rows == 128
    assert not accumulator.should_step(chunks_per_step=chunks_per_step(config))

    accumulator.mark_loaded(rows=128, chunks=1, loss_scale=128.0)
    assert accumulator.accumulated_rows == 256
    assert accumulator.accumulated_loss_scale == 256.0
    assert accumulator.should_step(chunks_per_step=chunks_per_step(config))

from __future__ import annotations

import math

import pytest
import torch

from wavelet.configs.rl_config import RLLossConfig
from wavelet.trainer.rl_loss import compute_loss
from wavelet.trainer.types import LossOutput


def test_loss_scale_matches_token_normalization() -> None:
    loss_config = RLLossConfig(kl_tau=0.0)
    trainer_logprobs = torch.zeros(1, 4)
    inference_logprobs = torch.zeros_like(trainer_logprobs)
    advantages = torch.tensor([[2.0, 2.0, 2.0, -1.0]])
    loss_mask = torch.tensor([[True, True, True, True]])
    position_ids = torch.tensor([[0, 1, 2, 0]])

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        advantages,
        loss_mask,
        loss_config,
        loss_scale=4,
        position_ids=position_ids,
    )

    assert isinstance(output, LossOutput)
    assert output.loss.item() == pytest.approx(-1.25)


def test_sequence_normalization_remains_available() -> None:
    loss_config = RLLossConfig(kl_tau=0.0, normalization="sequence")
    trainer_logprobs = torch.zeros(1, 4)
    inference_logprobs = torch.zeros_like(trainer_logprobs)
    advantages = torch.tensor([[2.0, 2.0, 2.0, -1.0]])
    loss_mask = torch.tensor([[True, True, True, True]])
    position_ids = torch.tensor([[0, 1, 2, 0]])

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        advantages,
        loss_mask,
        loss_config,
        loss_scale=2,
        position_ids=position_ids,
    )

    assert output.loss.item() == pytest.approx(-0.5)


def test_packed_loss_metrics_are_averaged_per_sequence() -> None:
    loss_config = RLLossConfig(kl_tau=0.0)
    trainer_logprobs = torch.tensor([[0.0, 0.0, 0.0, math.log(0.5)]])
    inference_logprobs = torch.tensor([[0.0, 0.0, 0.0, math.log(0.25)]])
    loss_mask = torch.tensor([[True, True, True, True]])
    advantages = torch.zeros_like(trainer_logprobs)
    position_ids = torch.tensor([[0, 1, 2, 0]])

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        advantages,
        loss_mask,
        loss_config,
        loss_scale=4,
        position_ids=position_ids,
    )

    expected_second_sequence_kl = 2.0 - math.log(2.0) - 1.0
    assert output.metrics["mismatch_kl"].item() == pytest.approx(
        expected_second_sequence_kl / 2
    )


def test_mixed_rl_and_ref_kl_components_normalize_independently() -> None:
    loss_config = RLLossConfig(kl_tau=0.0)
    trainer_logprobs = torch.zeros(1, 4)
    inference_logprobs = torch.zeros_like(trainer_logprobs)
    advantages = torch.tensor([[2.0, 2.0, 0.0, 0.0]])
    loss_mask = torch.ones(1, 4, dtype=torch.bool)

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        advantages,
        loss_mask,
        loss_config,
        ref_logprobs=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        rl_weights=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        ce_weights=torch.zeros(1, 4),
        ref_kl_weights=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    )

    assert output.loss.item() == pytest.approx(-3.0)
    assert output.metrics["policy_loss"].item() == pytest.approx(-2.0)
    assert output.metrics["ref_kl"].item() == pytest.approx(1.0)


def test_mixed_component_scales_can_cover_full_optimizer_batch() -> None:
    loss_config = RLLossConfig(kl_tau=0.0)
    trainer_logprobs = torch.zeros(1, 2)
    inference_logprobs = torch.zeros_like(trainer_logprobs)

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        torch.tensor([[2.0, 0.0]]),
        torch.ones(1, 2, dtype=torch.bool),
        loss_config,
        ref_logprobs=torch.tensor([[0.0, 1.0]]),
        rl_weights=torch.tensor([[1.0, 0.0]]),
        ref_kl_weights=torch.tensor([[0.0, 1.0]]),
        component_loss_scales={"rl": 2, "ref_kl": 4},
    )

    assert output.loss.item() == pytest.approx(-1.25)

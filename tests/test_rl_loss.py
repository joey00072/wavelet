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

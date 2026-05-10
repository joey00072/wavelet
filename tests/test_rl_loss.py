from __future__ import annotations

import math

import pytest
import torch

from wavelet.configs.rl_config import RLLossConfig
from wavelet.trainer.rl_loss import compute_loss


def test_loss_scale_matches_batch_token_normalization() -> None:
    loss_config = RLLossConfig(kl_tau=0.0)
    total_loss = torch.tensor(0.0)

    for advantages, mask in [
        ([2.0, 2.0, 2.0], [True, True, True]),
        ([-1.0], [True]),
    ]:
        trainer_logprobs = torch.zeros(1, len(advantages))
        inference_logprobs = torch.zeros_like(trainer_logprobs)
        loss, _ = compute_loss(
            trainer_logprobs,
            inference_logprobs,
            None,
            torch.tensor([advantages]),
            torch.tensor([mask]),
            loss_config,
            loss_scale=4,
        )
        total_loss = total_loss + loss

    assert total_loss.item() == -1.25


def test_packed_loss_metrics_are_averaged_per_sequence() -> None:
    loss_config = RLLossConfig(kl_tau=0.0)
    trainer_logprobs = torch.tensor([[0.0, 0.0, 0.0, math.log(0.5)]])
    inference_logprobs = torch.tensor([[0.0, 0.0, 0.0, math.log(0.25)]])
    loss_mask = torch.tensor([[True, True, True, True]])
    advantages = torch.zeros_like(trainer_logprobs)
    position_ids = torch.tensor([[0, 1, 2, 0]])

    _, metrics = compute_loss(
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
    assert metrics["mismatch_kl"].item() == pytest.approx(
        expected_second_sequence_kl / 2
    )

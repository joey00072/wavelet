from __future__ import annotations

import torch

from wavelet.configs.rl_config import RLLossConfig
from wavelet.trainer.rl_loss import compute_loss


def test_loss_scale_matches_prime_batch_token_normalization() -> None:
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

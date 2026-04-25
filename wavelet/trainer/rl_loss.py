from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from wavelet.configs.rl_config import RLLossConfig


@dataclass
class LossInputs:
    trainer_logprobs: Tensor
    inference_logprobs: Tensor
    teacher_logprobs: Tensor | None
    advantages: Tensor
    loss_mask: Tensor


@dataclass
class LossOutputs:
    loss: Tensor
    metrics: dict[str, Tensor]


def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def default_loss_fn(inputs: LossInputs, loss_config: RLLossConfig) -> LossOutputs:
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    trainer_probs = torch.exp(trainer_logprobs)
    inference_probs = torch.exp(inference_logprobs)
    probs_diff = trainer_probs - inference_probs
    invalid_high = probs_diff > loss_config.dppo_mask_high
    invalid_low = probs_diff < -loss_config.dppo_mask_low
    invalid = torch.where(advantages > 0, invalid_high, invalid_low)

    is_masked_high = (advantages > 0) & invalid_high
    is_masked_low = (advantages < 0) & invalid_low
    keep_mask = loss_mask & ~invalid

    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1

    scaled_advantages = loss_config.adv_tau * advantages
    if teacher_logprobs is not None:
        teacher_kl = teacher_logprobs - trainer_logprobs
        scaled_advantages = (
            scaled_advantages + loss_config.teacher_tau * teacher_kl.detach()
        )
    else:
        teacher_kl = None

    pg_loss = keep_mask * scaled_advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio.square()
    loss = (-pg_loss + loss_config.kl_tau * kl_loss).sum()

    metrics = {
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & invalid),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": _safe_mean(invalid.float(), loss_mask),
        "is_masked_low": _safe_mean(is_masked_low.float(), loss_mask),
        "is_masked_high": _safe_mean(is_masked_high.float(), loss_mask),
        "policy_loss": _safe_mean((-pg_loss), loss_mask),
        "kl_loss": _safe_mean(kl_loss, loss_mask),
        "advantage_mean": _safe_mean(advantages, loss_mask),
    }
    if teacher_kl is not None:
        metrics["teacher_kl"] = _safe_mean(teacher_kl, loss_mask)
    return LossOutputs(loss=loss, metrics=metrics)


def compute_loss(
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    teacher_logprobs: Tensor | None,
    advantages: Tensor,
    loss_mask: Tensor,
    loss_config: RLLossConfig,
) -> tuple[Tensor, dict[str, Any]]:
    outputs = default_loss_fn(
        LossInputs(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=inference_logprobs,
            teacher_logprobs=teacher_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
        ),
        loss_config,
    )
    loss_scale = torch.clamp_min(loss_mask.sum(), 1)
    scaled_loss = outputs.loss / loss_scale
    return scaled_loss, outputs.metrics

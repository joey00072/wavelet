from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from wavelet.configs.rl_config import RLLossConfig
from wavelet.trainer.types import LossOutput


@dataclass
class LossInputs:
    trainer_logprobs: Tensor
    inference_logprobs: Tensor
    teacher_logprobs: Tensor | None
    advantages: Tensor
    loss_mask: Tensor


def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def _tensor_stats(values: Tensor) -> dict[str, Tensor]:
    if values.numel() == 0:
        nan = torch.tensor(float("nan"), device=values.device)
        return {
            "mean": nan,
            "median": nan,
            "std": nan,
            "min": nan,
            "max": nan,
        }
    return {
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(unbiased=False),
        "min": values.min(),
        "max": values.max(),
    }


def default_loss_fn(inputs: LossInputs, loss_config: RLLossConfig) -> LossOutput:
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
    return LossOutput(loss=loss, metrics=metrics)


def _sequence_spans(position_ids: Tensor, seq_len: int) -> list[tuple[int, int]]:
    starts = (position_ids[:seq_len] == 0).nonzero(as_tuple=False).flatten().tolist()
    if not starts:
        return [(0, seq_len)]
    if starts[0] != 0:
        starts.insert(0, 0)
    ends = [*starts[1:], seq_len]
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end > start
    ]


def compute_loss(
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    teacher_logprobs: Tensor | None,
    advantages: Tensor,
    loss_mask: Tensor,
    loss_config: RLLossConfig,
    *,
    loss_scale: int | float | Tensor | None = None,
    position_ids: Tensor | None = None,
) -> LossOutput:
    total_loss: Tensor | None = None
    metric_values: dict[str, list[Tensor]] = {}
    sequence_count = 0

    if position_ids is None:
        spans_by_row = [
            [(0, trainer_logprobs.shape[1])] for _ in range(trainer_logprobs.shape[0])
        ]
    else:
        spans_by_row = [
            _sequence_spans(row_position_ids, trainer_logprobs.shape[1])
            for row_position_ids in position_ids
        ]

    for row_index, spans in enumerate(spans_by_row):
        for start, end in spans:
            span_loss_mask = loss_mask[row_index, start:end]
            span_trainable_tokens = span_loss_mask.sum()
            if int(span_trainable_tokens.item()) == 0:
                continue
            row_teacher = (
                None
                if teacher_logprobs is None
                else teacher_logprobs[row_index, start:end]
            )
            outputs = default_loss_fn(
                LossInputs(
                    trainer_logprobs=trainer_logprobs[row_index, start:end],
                    inference_logprobs=inference_logprobs[row_index, start:end],
                    teacher_logprobs=row_teacher,
                    advantages=advantages[row_index, start:end],
                    loss_mask=span_loss_mask,
                ),
                loss_config,
            )
            if loss_config.normalization == "sequence":
                span_loss = outputs.loss / torch.clamp_min(span_trainable_tokens, 1)
                sequence_count += 1
            else:
                span_loss = outputs.loss
            total_loss = span_loss if total_loss is None else total_loss + span_loss
            for key, value in outputs.metrics.items():
                metric_values.setdefault(key, []).append(value)

    if total_loss is None:
        total_loss = trainer_logprobs.sum() * 0.0
    if loss_scale is None:
        if loss_config.normalization == "sequence":
            scale = max(float(sequence_count), 1.0)
        else:
            scale = torch.clamp_min(loss_mask.sum(), 1)
    elif isinstance(loss_scale, Tensor):
        scale = torch.clamp_min(loss_scale.to(total_loss.device), 1)
    else:
        scale = max(float(loss_scale), 1.0)
    scaled_loss = total_loss / scale
    metrics = {
        key: torch.stack(values).mean()
        for key, values in metric_values.items()
        if values
    }
    for key, values in metric_values.items():
        if not values:
            continue
        stacked = torch.stack(values)
        for stat_name, stat_value in _tensor_stats(stacked).items():
            metrics[f"{key}/{stat_name}"] = stat_value
    return LossOutput(loss=scaled_loss, metrics=metrics)

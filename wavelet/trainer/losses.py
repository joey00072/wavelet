from __future__ import annotations

import importlib
import logging
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from torch import Tensor, nn

from wavelet.configs.rl_config import RLLossConfig
from wavelet.trainer.types import LossOutput


@dataclass
class LossInputs:
    trainer_logprobs: Tensor
    inference_logprobs: Tensor
    teacher_logprobs: Tensor | None
    advantages: Tensor
    loss_mask: Tensor
    loss_weights: Tensor | None = None


LossFn = Callable[[LossInputs], LossOutput]


def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def selective_log_softmax_with_sampling_mask(
    logits: Tensor,
    index: Tensor,
    sampling_mask_ids: Tensor,
    sampling_mask_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute target logprobs and entropy over vLLM's sampled support sets."""
    selected = torch.gather(
        logits.log_softmax(dim=-1), dim=-1, index=index.unsqueeze(-1)
    ).squeeze(-1)
    if sampling_mask_ids.shape[-1] == 0 or not sampling_mask_lengths.any():
        return selected, compute_entropy(logits)

    vocab_size = logits.shape[-1]
    if (sampling_mask_ids >= vocab_size).any():
        raise ValueError("sampling mask contains a token outside the model vocabulary")
    if (sampling_mask_lengths > sampling_mask_ids.shape[-1]).any():
        raise ValueError("sampling mask length exceeds its padded capacity")
    support_logits = torch.gather(logits, dim=-1, index=sampling_mask_ids)
    valid = torch.arange(sampling_mask_ids.shape[-1], device=logits.device).view(
        1, 1, -1
    ) < sampling_mask_lengths.unsqueeze(-1)
    target_present = ((sampling_mask_ids == index.unsqueeze(-1)) & valid).any(dim=-1)
    if (sampling_mask_lengths > 0).logical_and(~target_present).any():
        raise ValueError("sampling mask does not contain the target token")
    support_logits = support_logits.masked_fill(~valid, float("-inf"))
    support_log_z = torch.logsumexp(support_logits, dim=-1)
    masked_selected = (
        torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        - support_log_z
    )
    selected = torch.where(sampling_mask_lengths > 0, masked_selected, selected)
    return selected, compute_entropy(logits)


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token categorical entropy without retaining gradients."""
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=-1)
        return torch.logsumexp(logits, dim=-1) - (probabilities * logits).sum(dim=-1)


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
    per_token_loss = -pg_loss + loss_config.kl_tau * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()

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


def ipo_loss_fn(inputs: LossInputs, loss_config: RLLossConfig) -> LossOutput:
    """Return IPO with a symmetric sampled-token probability trust region."""
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    trainer_probs = torch.exp(trainer_logprobs)
    inference_probs = torch.exp(inference_logprobs)
    probs_diff = trainer_probs - inference_probs
    invalid = probs_diff.abs() > loss_config.ipo_epsilon
    invalid_high = probs_diff > loss_config.ipo_epsilon
    invalid_low = probs_diff < -loss_config.ipo_epsilon
    keep_mask = loss_mask & ~invalid

    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1
    scaled_advantages = loss_config.adv_tau * advantages

    pg_loss = keep_mask * scaled_advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio.square()
    per_token_loss = -pg_loss + loss_config.kl_tau * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights

    return LossOutput(
        loss=per_token_loss.sum(),
        metrics={
            "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
            "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & invalid),
            "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
            "is_masked": _safe_mean(invalid.float(), loss_mask),
            "is_masked_low": _safe_mean(invalid_low.float(), loss_mask),
            "is_masked_high": _safe_mean(invalid_high.float(), loss_mask),
            "policy_loss": _safe_mean(-pg_loss, loss_mask),
            "kl_loss": _safe_mean(kl_loss, loss_mask),
            "advantage_mean": _safe_mean(advantages, loss_mask),
        },
    )


def ce_loss_fn(inputs: LossInputs) -> LossOutput:
    """Return weighted next-token cross entropy for one sequence."""
    per_token_loss = -inputs.trainer_logprobs
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    return LossOutput(
        loss=per_token_loss[inputs.loss_mask].sum(),
        metrics={"ce/nll": _safe_mean(-inputs.trainer_logprobs, inputs.loss_mask)},
    )


def ref_kl_loss_fn(inputs: LossInputs) -> LossOutput:
    """Return weighted reverse-KL policy loss with a one-sided trust region."""
    if inputs.teacher_logprobs is None:
        raise ValueError(
            "ref_kl loss requires teacher_logprobs for every weighted token."
        )

    log_ratio = inputs.trainer_logprobs - inputs.inference_logprobs
    importance_ratio = torch.exp(log_ratio)
    mismatch_kl = importance_ratio - log_ratio - 1
    probability_delta = torch.exp(inputs.trainer_logprobs) - torch.exp(
        inputs.inference_logprobs
    )
    invalid = probability_delta < -0.2
    keep_mask = inputs.loss_mask & ~invalid
    teacher_kl = inputs.teacher_logprobs - inputs.trainer_logprobs
    policy_loss = -(keep_mask * teacher_kl.detach() * importance_ratio)
    regularizer = inputs.loss_mask * log_ratio.square()
    per_token_loss = policy_loss + 1e-3 * regularizer
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    return LossOutput(
        loss=per_token_loss.sum(),
        metrics={
            "ref_kl/masked_mismatch_kl": _safe_mean(
                mismatch_kl, inputs.loss_mask & invalid
            ),
            "ref_kl/unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
            "ref_kl/is_masked": _safe_mean(invalid.float(), inputs.loss_mask),
            "ref_kl": _safe_mean(teacher_kl, inputs.loss_mask),
        },
    )


def _import_loss_fn(import_path: str) -> Callable[..., LossOutput]:
    module_name, separator, attribute = import_path.rpartition(".")
    if not separator:
        raise ValueError(
            "Custom loss import_path must include a module and attribute name."
        )
    module = importlib.import_module(module_name)
    loss_fn = getattr(module, attribute)
    if not callable(loss_fn):
        raise TypeError(f"Custom loss object {import_path!r} is not callable.")
    return loss_fn


def setup_rl_loss_fn(loss_config: RLLossConfig) -> LossFn:
    """Resolve the configured per-sequence RL component loss."""
    if loss_config.type == "custom":
        assert loss_config.import_path is not None
        custom_fn = _import_loss_fn(loss_config.import_path)
        kwargs = dict(loss_config.kwargs)

        def custom_loss(inputs: LossInputs) -> LossOutput:
            output = custom_fn(inputs, **kwargs)
            if not isinstance(output, LossOutput):
                raise TypeError("Custom RL loss must return LossOutput.")
            return output

        return custom_loss

    if loss_config.type == "ipo":

        def ipo_loss(inputs: LossInputs) -> LossOutput:
            return ipo_loss_fn(inputs, loss_config)

        return ipo_loss

    def dppo_loss(inputs: LossInputs) -> LossOutput:
        return default_loss_fn(inputs, loss_config)

    return dppo_loss


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


def normalization_unit_count(
    loss_mask: Tensor,
    *,
    normalization: str,
    position_ids: Tensor | None = None,
) -> int:
    """Count the units used to normalize one RL loss batch."""
    if normalization == "token":
        return int(loss_mask.sum().item())
    if normalization != "sequence":
        raise ValueError(f"Unsupported RL loss normalization: {normalization!r}")

    seq_len = loss_mask.shape[1]
    spans_by_row = (
        [[(0, seq_len)] for _ in range(loss_mask.shape[0])]
        if position_ids is None
        else [_sequence_spans(row, seq_len) for row in position_ids]
    )
    return sum(
        int(loss_mask[row_index, start:end].any().item())
        for row_index, spans in enumerate(spans_by_row)
        for start, end in spans
    )


def _iter_trainable_spans(
    loss_mask: Tensor,
    position_ids: Tensor | None,
) -> Iterator[tuple[int, int, int]]:
    seq_len = loss_mask.shape[1]
    spans_by_row = (
        [[(0, seq_len)] for _ in range(loss_mask.shape[0])]
        if position_ids is None
        else [_sequence_spans(row, seq_len) for row in position_ids]
    )
    for row_index, spans in enumerate(spans_by_row):
        for start, end in spans:
            span_mask = loss_mask[row_index, start:end]
            if not bool(span_mask.any()):
                continue
            yield row_index, start, end


def _resolve_loss_scale(
    loss_scale: float | Tensor | None,
    *,
    normalization: str,
    sequence_count: int,
    loss_mask: Tensor,
    device: torch.device,
) -> float | Tensor:
    if loss_scale is None:
        if normalization == "sequence":
            return max(float(sequence_count), 1.0)
        return torch.clamp_min(loss_mask.sum(), 1)
    if isinstance(loss_scale, Tensor):
        return torch.clamp_min(loss_scale.to(device), 1)
    return max(float(loss_scale), 1.0)


def component_normalization_unit_counts(
    loss_mask: Tensor,
    *,
    rl_weights: Tensor | None = None,
    ce_weights: Tensor | None = None,
    ref_kl_weights: Tensor | None = None,
    rl_normalization: str = "token",
    position_ids: Tensor | None = None,
) -> dict[str, int]:
    """Count local normalization units for each configured loss component."""
    rl_mask = loss_mask if rl_weights is None else loss_mask & (rl_weights != 0)
    ce_mask = (
        torch.zeros_like(loss_mask)
        if ce_weights is None
        else loss_mask & (ce_weights != 0)
    )
    ref_kl_mask = (
        torch.zeros_like(loss_mask)
        if ref_kl_weights is None
        else loss_mask & (ref_kl_weights != 0)
    )
    return {
        "rl": normalization_unit_count(
            rl_mask,
            normalization=rl_normalization,
            position_ids=position_ids,
        ),
        "ce": int(ce_mask.sum().item()),
        "ref_kl": int(ref_kl_mask.sum().item()),
    }


def _aggregate_loss_metrics(
    metric_values: dict[str, list[Tensor]],
) -> dict[str, Tensor]:
    metrics = {
        key: torch.stack(values).mean()
        for key, values in metric_values.items()
        if values
    }
    for key, values in metric_values.items():
        if not values:
            continue
        for stat_name, stat_value in _tensor_stats(torch.stack(values)).items():
            metrics[f"{key}/{stat_name}"] = stat_value
    return metrics


def _accumulate_loss_output(
    component: str,
    output: LossOutput,
    component_losses: dict[str, Tensor],
    metric_values: dict[str, list[Tensor]],
    *,
    divisor: Tensor | int = 1,
) -> None:
    component_losses[component] = component_losses[component] + output.loss / divisor
    for key, value in output.metrics.items():
        metric_values.setdefault(key, []).append(value)


def _accumulate_span_components(
    *,
    row_index: int,
    start: int,
    end: int,
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    teacher_logprobs: Tensor | None,
    advantages: Tensor,
    loss_mask: Tensor,
    rl_weights: Tensor | None,
    ce_weights: Tensor | None,
    ref_kl_weights: Tensor | None,
    rl_loss_fn: LossFn,
    rl_normalization: str,
    component_losses: dict[str, Tensor],
    metric_values: dict[str, list[Tensor]],
) -> None:
    span_mask = loss_mask[row_index, start:end]
    common: dict[str, Any] = {
        "trainer_logprobs": trainer_logprobs[row_index, start:end],
        "inference_logprobs": inference_logprobs[row_index, start:end],
        "teacher_logprobs": (
            None if teacher_logprobs is None else teacher_logprobs[row_index, start:end]
        ),
        "advantages": advantages[row_index, start:end],
    }

    span_rl_weights = None if rl_weights is None else rl_weights[row_index, start:end]
    rl_mask = (
        span_mask if span_rl_weights is None else span_mask & (span_rl_weights != 0)
    )
    if bool(rl_mask.any()):
        divisor = (
            torch.clamp_min(rl_mask.sum(), 1) if rl_normalization == "sequence" else 1
        )
        _accumulate_loss_output(
            "rl",
            rl_loss_fn(
                LossInputs(
                    **common,
                    loss_mask=rl_mask,
                    loss_weights=span_rl_weights,
                )
            ),
            component_losses,
            metric_values,
            divisor=divisor,
        )

    for component, weights, loss_fn in (
        ("ce", ce_weights, ce_loss_fn),
        ("ref_kl", ref_kl_weights, ref_kl_loss_fn),
    ):
        if weights is None:
            continue
        span_weights = weights[row_index, start:end]
        component_mask = span_mask & (span_weights != 0)
        if bool(component_mask.any()):
            _accumulate_loss_output(
                component,
                loss_fn(
                    LossInputs(
                        **common,
                        loss_mask=component_mask,
                        loss_weights=span_weights,
                    )
                ),
                component_losses,
                metric_values,
            )


def compute_loss(
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    teacher_logprobs: Tensor | None,
    advantages: Tensor,
    loss_mask: Tensor,
    loss_config: RLLossConfig,
    *,
    loss_scale: float | Tensor | None = None,
    position_ids: Tensor | None = None,
    rl_weights: Tensor | None = None,
    ce_weights: Tensor | None = None,
    ref_kl_weights: Tensor | None = None,
    component_loss_scales: dict[str, float | Tensor] | None = None,
    rl_loss_fn: LossFn | None = None,
) -> LossOutput:
    component_losses = {
        "rl": trainer_logprobs.sum() * 0.0,
        "ce": trainer_logprobs.sum() * 0.0,
        "ref_kl": trainer_logprobs.sum() * 0.0,
    }
    metric_values: dict[str, list[Tensor]] = {}
    component_counts = component_normalization_unit_counts(
        loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=ref_kl_weights,
        rl_normalization=loss_config.normalization,
        position_ids=position_ids,
    )
    configured_rl_loss = rl_loss_fn or setup_rl_loss_fn(loss_config)

    for row_index, start, end in _iter_trainable_spans(loss_mask, position_ids):
        _accumulate_span_components(
            row_index=row_index,
            start=start,
            end=end,
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=inference_logprobs,
            teacher_logprobs=teacher_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            rl_weights=rl_weights,
            ce_weights=ce_weights,
            ref_kl_weights=ref_kl_weights,
            rl_loss_fn=configured_rl_loss,
            rl_normalization=loss_config.normalization,
            component_losses=component_losses,
            metric_values=metric_values,
        )

    scales = dict(component_loss_scales or {})
    if "rl" not in scales and loss_scale is not None:
        scales["rl"] = loss_scale
    scaled_loss = trainer_logprobs.sum() * 0.0
    for component, component_loss in component_losses.items():
        requested_scale = scales.get(component)
        if requested_scale is None:
            scale: float | Tensor = max(float(component_counts[component]), 1.0)
        else:
            scale = _resolve_loss_scale(
                requested_scale,
                normalization="token",
                sequence_count=component_counts[component],
                loss_mask=loss_mask,
                device=component_loss.device,
            )
        scaled_loss = scaled_loss + component_loss / scale
    return LossOutput(
        loss=scaled_loss,
        metrics=_aggregate_loss_metrics(metric_values),
    )


logger = logging.getLogger(__name__)


class ChunkedLmHeadOutput(TypedDict, total=False):
    logits: Tensor | None
    logprobs: Tensor | None
    entropy: Tensor | None


class ChunkedLogprobLmHead(nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int) -> None:
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: Tensor,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask_ids: Tensor | None = None,
        sampling_mask_lengths: Tensor | None = None,
    ) -> ChunkedLmHeadOutput:
        if labels is None:
            return {"logits": super().forward(hidden_states)}
        if temperature is None:
            raise ValueError("temperature is required for chunked logprob LM head")

        batch, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(batch * seq_len, hidden_size).contiguous()
        labels_flat = labels.reshape(batch * seq_len).contiguous()
        inv_temperature = (1.0 / temperature.reshape(batch * seq_len)).contiguous()
        if sampling_mask_ids is None:
            sampling_mask_ids = torch.empty(
                (batch * seq_len, 0), dtype=torch.long, device=hidden_states.device
            )
        else:
            sampling_mask_ids = sampling_mask_ids.reshape(
                batch * seq_len, -1
            ).contiguous()
        if sampling_mask_lengths is None:
            sampling_mask_lengths = torch.zeros(
                batch * seq_len, dtype=torch.long, device=hidden_states.device
            )
        else:
            sampling_mask_lengths = sampling_mask_lengths.reshape(
                batch * seq_len
            ).contiguous()

        logprobs, entropy = _ChunkedLogprobFn.apply(
            hidden_flat,
            self.weight,
            labels_flat,
            inv_temperature,
            self.chunk_size,
            sampling_mask_ids,
            sampling_mask_lengths,
        )
        return {
            "logprobs": logprobs.reshape(batch, seq_len),
            "entropy": entropy.reshape(batch, seq_len),
        }


def _online_softmax_stats_update(
    max_values: Tensor,
    sums: Tensor,
    weighted_sums: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    chunk_max = torch.amax(logits, dim=-1)
    new_max = torch.maximum(max_values, chunk_max)
    old_scale = torch.exp(max_values - new_max)
    exp_logits = torch.exp(logits - new_max.unsqueeze(-1))
    new_sums = sums * old_scale + exp_logits.sum(dim=-1)
    new_weighted_sums = weighted_sums * old_scale + (
        exp_logits * logits.masked_fill(~torch.isfinite(logits), 0.0)
    ).sum(dim=-1)
    return new_max, new_sums, new_weighted_sums


def _validate_chunked_logprob_inputs(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    inv_temperature: Tensor,
    chunk_size: int,
    sampling_mask_ids: Tensor,
    sampling_mask_lengths: Tensor,
) -> None:
    if hidden.dim() != 2:
        raise ValueError(f"expected hidden [N, H], got {tuple(hidden.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"expected weight [V, H], got {tuple(weight.shape)}")
    if labels.dim() != 1:
        raise ValueError(f"expected labels [N], got {tuple(labels.shape)}")
    if inv_temperature.dim() != 1:
        raise ValueError(
            f"expected inv_temperature [N], got {tuple(inv_temperature.shape)}"
        )
    if hidden.shape[0] != labels.shape[0]:
        raise ValueError("hidden and labels must have matching token count")
    if hidden.shape[0] != inv_temperature.shape[0]:
        raise ValueError("hidden and temperatures must have matching token count")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight dimensions are incompatible")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if sampling_mask_ids.dim() != 2:
        raise ValueError(
            f"expected sampling_mask_ids [N, K], got {tuple(sampling_mask_ids.shape)}"
        )
    if sampling_mask_lengths.dim() != 1:
        raise ValueError(
            "expected sampling_mask_lengths [N], "
            f"got {tuple(sampling_mask_lengths.shape)}"
        )
    if sampling_mask_ids.shape[0] != hidden.shape[0]:
        raise ValueError("hidden and sampling mask ids must have matching token count")
    if sampling_mask_lengths.shape[0] != hidden.shape[0]:
        raise ValueError(
            "hidden and sampling mask lengths must have matching token count"
        )
    if (
        sampling_mask_ids.dtype != torch.long
        or sampling_mask_lengths.dtype != torch.long
    ):
        raise ValueError("sampling masks must use int64 tensors")
    if (sampling_mask_lengths < 0).any() or (
        sampling_mask_lengths > sampling_mask_ids.shape[1]
    ).any():
        raise ValueError("sampling mask lengths exceed their padded capacity")
    if sampling_mask_ids.numel() and (sampling_mask_ids < 0).any():
        raise ValueError("sampling mask ids must be nonnegative")


class _ChunkedLogprobFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: Tensor,
        weight: Tensor,
        labels: Tensor,
        inv_temperature: Tensor,
        chunk_size: int,
        sampling_mask_ids: Tensor,
        sampling_mask_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _validate_chunked_logprob_inputs(
            hidden,
            weight,
            labels,
            inv_temperature,
            chunk_size,
            sampling_mask_ids,
            sampling_mask_lengths,
        )

        device = hidden.device
        token_count = hidden.shape[0]
        vocab_size = weight.shape[0]
        vocab_chunk_size = min(vocab_size, 8192)
        masked = sampling_mask_ids.shape[1] > 0 and sampling_mask_lengths.any()
        if masked and (sampling_mask_ids >= vocab_size).any():
            raise ValueError("sampling mask contains a token outside the vocabulary")
        logprobs = torch.empty(token_count, device=device, dtype=torch.float32)
        entropy = torch.empty(token_count, device=device, dtype=torch.float32)
        log_z = torch.empty(token_count, device=device, dtype=torch.float32)

        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_temperature_chunk = inv_temperature[start:end].unsqueeze(-1)
            chunk_tokens = end - start

            max_values = torch.full(
                (chunk_tokens,),
                float("-inf"),
                device=device,
                dtype=torch.float32,
            )
            sums = torch.zeros(chunk_tokens, device=device, dtype=torch.float32)
            weighted_sums = torch.zeros(
                chunk_tokens, device=device, dtype=torch.float32
            )
            target_logits = torch.zeros(
                chunk_tokens, device=device, dtype=torch.float32
            )
            mask_logits = (
                torch.full(
                    (chunk_tokens, sampling_mask_ids.shape[1]),
                    float("-inf"),
                    device=device,
                    dtype=torch.float32,
                )
                if masked
                else None
            )

            for vocab_start in range(0, vocab_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                weight_chunk = weight[vocab_start:vocab_end]
                logits = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits.to(torch.float32) * inv_temperature_chunk
                max_values, sums, weighted_sums = _online_softmax_stats_update(
                    max_values,
                    sums,
                    weighted_sums,
                    scaled_logits,
                )

                in_chunk = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(in_chunk):
                    indexes = (labels_chunk[in_chunk] - vocab_start).to(torch.long)
                    target_logits[in_chunk] = scaled_logits[in_chunk, indexes]
                if mask_logits is not None:
                    mask_ids = sampling_mask_ids[start:end]
                    mask_valid = torch.arange(mask_ids.shape[1], device=device).view(
                        1, -1
                    ) < sampling_mask_lengths[start:end].view(-1, 1)
                    local_ids = mask_ids - vocab_start
                    in_chunk_mask = (
                        mask_valid
                        & (local_ids >= 0)
                        & (local_ids < vocab_end - vocab_start)
                    )
                    safe_ids = local_ids.clamp(0, vocab_end - vocab_start - 1)
                    gathered = torch.gather(scaled_logits, 1, safe_ids)
                    mask_logits = torch.where(in_chunk_mask, gathered, mask_logits)

            log_z_chunk = max_values + torch.log(sums)
            log_z[start:end] = log_z_chunk
            logprobs[start:end] = target_logits - log_z_chunk
            entropy[start:end] = log_z_chunk - weighted_sums / sums

            if mask_logits is not None:
                masked_rows = sampling_mask_lengths[start:end] > 0
                if masked_rows.any():
                    support_log_z = torch.logsumexp(mask_logits[masked_rows], dim=-1)
                    masked_indices = masked_rows.nonzero(as_tuple=True)[0]
                    log_z[start:end].index_copy_(0, masked_indices, support_log_z)
                    logprobs[start:end].index_copy_(
                        0,
                        masked_indices,
                        target_logits[masked_rows] - support_log_z,
                    )

        ctx.set_materialize_grads(False)
        if masked:
            target_present = (
                (sampling_mask_ids == labels.unsqueeze(-1))
                & (
                    torch.arange(sampling_mask_ids.shape[1], device=device).view(1, -1)
                    < sampling_mask_lengths.view(-1, 1)
                )
            ).any(dim=-1)
            if ((sampling_mask_lengths > 0) & ~target_present).any():
                raise ValueError("sampling mask does not contain the target token")
        ctx.save_for_backward(
            hidden,
            weight,
            labels,
            inv_temperature,
            log_z,
            sampling_mask_ids,
            sampling_mask_lengths,
        )
        ctx.chunk_size = chunk_size
        ctx.masked = masked
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: Tensor, grad_entropy: Tensor | None):
        if grad_entropy is not None:
            raise RuntimeError("Backward through entropy is not supported.")
        if grad_logprobs is None:
            raise RuntimeError("Chunked logprob backward requires logprob gradients.")
        (
            hidden,
            weight,
            labels,
            inv_temperature,
            log_z,
            sampling_mask_ids,
            sampling_mask_lengths,
        ) = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size
        needs_grad_hidden, needs_grad_weight = ctx.needs_input_grad[:2]
        token_count = hidden.shape[0]
        vocab_size = weight.shape[0]
        vocab_chunk_size = min(vocab_size, 8192)
        masked: bool = ctx.masked

        grad_hidden = torch.zeros_like(hidden) if needs_grad_hidden else None
        grad_weight = torch.zeros_like(weight) if needs_grad_weight else None

        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = grad_logprobs[start:end].to(torch.float32)
            inv_temperature_chunk = inv_temperature[start:end].unsqueeze(-1)
            log_z_chunk = log_z[start:end]

            for vocab_start in range(0, vocab_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                weight_chunk = weight[vocab_start:vocab_end]
                logits = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits.to(torch.float32) * inv_temperature_chunk
                full_rows = sampling_mask_lengths[start:end] == 0 if masked else None
                if full_rows is None or full_rows.any():
                    rows = slice(None) if full_rows is None else full_rows
                    full_logits = scaled_logits[rows]
                    full_hidden = hidden_chunk[rows]
                    full_labels = labels_chunk[rows]
                    full_grad = grad_chunk[rows]
                    full_log_z = log_z_chunk[rows]
                    full_inv_temperature = inv_temperature_chunk[rows]
                    probs = torch.exp(full_logits - full_log_z.unsqueeze(-1))
                    grad_logits = (-full_grad).unsqueeze(-1) * probs
                    in_chunk = (full_labels >= vocab_start) & (full_labels < vocab_end)
                    if torch.any(in_chunk):
                        indexes = (full_labels[in_chunk] - vocab_start).to(torch.long)
                        grad_logits[in_chunk, indexes] += full_grad[in_chunk]
                    grad_logits = grad_logits * full_inv_temperature
                    if grad_hidden is not None:
                        hidden_grad = grad_logits.to(hidden.dtype) @ weight_chunk
                        if full_rows is None:
                            grad_hidden[start:end].add_(hidden_grad)
                        else:
                            grad_hidden[start:end].index_add_(
                                0, full_rows.nonzero(as_tuple=True)[0], hidden_grad
                            )
                    if grad_weight is not None:
                        grad_weight[vocab_start:vocab_end].add_(
                            grad_logits.to(weight.dtype).t() @ full_hidden
                        )

                if masked:
                    masked_rows = sampling_mask_lengths[start:end] > 0
                    if masked_rows.any():
                        mask_ids = sampling_mask_ids[start:end][masked_rows]
                        mask_valid = torch.arange(
                            mask_ids.shape[1], device=hidden.device
                        ).view(1, -1) < sampling_mask_lengths[start:end][
                            masked_rows
                        ].view(-1, 1)
                        local_ids = mask_ids - vocab_start
                        in_chunk_mask = (
                            mask_valid
                            & (local_ids >= 0)
                            & (local_ids < vocab_end - vocab_start)
                        )
                        safe_ids = local_ids.clamp(0, vocab_end - vocab_start - 1)
                        mask_indicator = torch.zeros_like(scaled_logits[masked_rows])
                        mask_indicator.scatter_add_(
                            1, safe_ids, in_chunk_mask.to(mask_indicator.dtype)
                        )
                        masked_log_z = log_z_chunk[masked_rows]
                        masked_probs = (
                            torch.exp(
                                scaled_logits[masked_rows] - masked_log_z.unsqueeze(-1)
                            )
                            * mask_indicator
                        )
                        masked_grad = grad_chunk[masked_rows]
                        grad_logits = -masked_grad.unsqueeze(-1) * masked_probs
                        target_mask = (labels_chunk[masked_rows] >= vocab_start) & (
                            labels_chunk[masked_rows] < vocab_end
                        )
                        if target_mask.any():
                            target_local = (
                                labels_chunk[masked_rows][target_mask] - vocab_start
                            ).to(torch.long)
                            grad_logits[target_mask, target_local] += masked_grad[
                                target_mask
                            ]
                        grad_logits = grad_logits * inv_temperature_chunk[masked_rows]
                        if grad_hidden is not None:
                            grad_hidden[start:end].index_add_(
                                0,
                                masked_rows.nonzero(as_tuple=True)[0],
                                grad_logits.to(hidden.dtype) @ weight_chunk,
                            )
                        if grad_weight is not None:
                            grad_weight[vocab_start:vocab_end].add_(
                                grad_logits.to(weight.dtype).t()
                                @ hidden_chunk[masked_rows]
                            )

        return (
            grad_hidden,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
        )


def maybe_inject_chunked_lm_head(model: nn.Module, chunk_size: int | str) -> None:
    if chunk_size == "disabled":
        return
    if chunk_size == "auto":
        chunk_size = 8192
    if not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an int, 'auto', or 'disabled'")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not hasattr(model, "model") or not isinstance(model.model, nn.Module):
        raise ValueError("chunked LM head requires a model.model backbone")
    if not hasattr(model, "lm_head") or not isinstance(model.lm_head, nn.Linear):
        raise ValueError("chunked LM head requires a linear model.lm_head")
    old_lm_head = model.lm_head
    if old_lm_head.bias is not None:
        raise ValueError("chunked LM head does not support lm_head bias")

    logger.info("Injecting chunked LM head with chunk size %s", chunk_size)
    new_lm_head = ChunkedLogprobLmHead(
        old_lm_head.in_features,
        old_lm_head.out_features,
        chunk_size=chunk_size,
    )
    new_lm_head.weight = old_lm_head.weight
    model.lm_head = new_lm_head
    _patch_causal_lm_forward(model)


def _patch_causal_lm_forward(model: nn.Module) -> None:
    def forward(
        self: nn.Module,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask_ids: Tensor | None = None,
        sampling_mask_lengths: Tensor | None = None,
        logits_to_keep: int = 0,
        **kwargs: object,
    ) -> ChunkedLmHeadOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            sequence_slice = slice(-logits_to_keep, None)
        else:
            sequence_slice = slice(None)
        return self.lm_head(
            hidden_states[:, sequence_slice, :],
            labels[:, sequence_slice] if labels is not None else None,
            temperature=(
                temperature[:, sequence_slice] if temperature is not None else None
            ),
            sampling_mask_ids=(
                sampling_mask_ids[:, sequence_slice]
                if sampling_mask_ids is not None
                else None
            ),
            sampling_mask_lengths=(
                sampling_mask_lengths[:, sequence_slice]
                if sampling_mask_lengths is not None
                else None
            ),
        )

    model.forward = types.MethodType(forward, model)

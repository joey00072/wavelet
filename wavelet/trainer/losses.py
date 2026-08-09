from __future__ import annotations

import logging
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TypedDict

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
    ref_logprobs: Tensor | None = None
    loss_weights: Tensor | None = None


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


def ref_kl_loss_fn(
    inputs: LossInputs,
    loss_config: RLLossConfig,
) -> LossOutput:
    """Compute OPD reverse-KL credit with off-policy correction."""
    ref_logprobs = inputs.ref_logprobs
    if ref_logprobs is None:
        raise ValueError("ref_kl loss requires ref_logprobs.")

    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    loss_mask = inputs.loss_mask
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1
    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    invalid = probs_diff < -loss_config.dppo_mask_low
    keep_mask = loss_mask & ~invalid
    ref_kl = ref_logprobs - trainer_logprobs

    policy_loss = -(keep_mask * ref_kl.detach() * importance_ratio)
    drift_loss = loss_mask * log_importance_ratio.square()
    per_token_loss = policy_loss + loss_config.kl_tau * drift_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()
    return LossOutput(
        loss=loss,
        metrics={
            "ref_kl": _safe_mean(ref_kl, loss_mask),
            "ref_kl/is_masked": _safe_mean(invalid.float(), loss_mask),
            "ref_kl/masked_mismatch_kl": _safe_mean(
                mismatch_kl,
                loss_mask & invalid,
            ),
            "ref_kl/unmasked_mismatch_kl": _safe_mean(
                mismatch_kl,
                keep_mask,
            ),
        },
    )


def ce_loss_fn(inputs: LossInputs) -> LossOutput:
    """Compute masked cross entropy for distillation and observation tokens."""
    per_token_loss = -inputs.trainer_logprobs
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    return LossOutput(
        loss=per_token_loss[inputs.loss_mask].sum(),
        metrics={"ce/nll": _safe_mean(-inputs.trainer_logprobs, inputs.loss_mask)},
    )


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


def _iter_trainable_spans(
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    teacher_logprobs: Tensor | None,
    advantages: Tensor,
    loss_mask: Tensor,
    position_ids: Tensor | None,
    ref_logprobs: Tensor | None = None,
    rl_weights: Tensor | None = None,
    ce_weights: Tensor | None = None,
    ref_kl_weights: Tensor | None = None,
) -> Iterator[
    tuple[
        LossInputs,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]
]:
    seq_len = trainer_logprobs.shape[1]
    spans_by_row = (
        [[(0, seq_len)] for _ in range(trainer_logprobs.shape[0])]
        if position_ids is None
        else [_sequence_spans(row, seq_len) for row in position_ids]
    )
    for row_index, spans in enumerate(spans_by_row):
        for start, end in spans:
            span_mask = loss_mask[row_index, start:end]
            trainable_tokens = span_mask.sum()
            if int(trainable_tokens.item()) == 0:
                continue
            yield (
                LossInputs(
                    trainer_logprobs=trainer_logprobs[row_index, start:end],
                    inference_logprobs=inference_logprobs[row_index, start:end],
                    teacher_logprobs=(
                        None
                        if teacher_logprobs is None
                        else teacher_logprobs[row_index, start:end]
                    ),
                    advantages=advantages[row_index, start:end],
                    loss_mask=span_mask,
                    ref_logprobs=(
                        None
                        if ref_logprobs is None
                        else ref_logprobs[row_index, start:end]
                    ),
                ),
                trainable_tokens,
                (None if rl_weights is None else rl_weights[row_index, start:end]),
                (None if ce_weights is None else ce_weights[row_index, start:end]),
                (
                    None
                    if ref_kl_weights is None
                    else ref_kl_weights[row_index, start:end]
                ),
            )


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
    ref_logprobs: Tensor | None = None,
    rl_weights: Tensor | None = None,
    ce_weights: Tensor | None = None,
    ref_kl_weights: Tensor | None = None,
    component_loss_scales: Mapping[str, float | Tensor] | None = None,
) -> LossOutput:
    component_losses = {
        "rl": trainer_logprobs.sum() * 0.0,
        "ce": trainer_logprobs.sum() * 0.0,
        "ref_kl": trainer_logprobs.sum() * 0.0,
    }
    metric_values: dict[str, list[Tensor]] = {}
    component_token_counts = {"rl": 0, "ce": 0, "ref_kl": 0}
    component_sequence_counts = {"rl": 0, "ce": 0, "ref_kl": 0}

    def record_output(
        component: str,
        outputs: LossOutput,
        component_mask: Tensor,
    ) -> None:
        component_loss = outputs.loss
        token_count = int(component_mask.sum().item())
        component_token_counts[component] += token_count
        if loss_config.normalization == "sequence":
            component_loss = component_loss / max(token_count, 1)
            component_sequence_counts[component] += 1
        component_losses[component] = component_losses[component] + component_loss
        for key, value in outputs.metrics.items():
            metric_values.setdefault(key, []).append(value)

    spans = _iter_trainable_spans(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask,
        position_ids,
        ref_logprobs,
        rl_weights,
        ce_weights,
        ref_kl_weights,
    )
    for span_inputs, _trainable_tokens, rl_w, ce_w, ref_kl_w in spans:
        rl_mask = (
            span_inputs.loss_mask
            if rl_w is None
            else (span_inputs.loss_mask & (rl_w != 0))
        )
        if bool(rl_mask.any()):
            record_output(
                "rl",
                default_loss_fn(
                    replace(span_inputs, loss_mask=rl_mask, loss_weights=rl_w),
                    loss_config,
                ),
                rl_mask,
            )

        if ce_w is not None:
            ce_mask = span_inputs.loss_mask & (ce_w != 0)
            if bool(ce_mask.any()):
                record_output(
                    "ce",
                    ce_loss_fn(
                        replace(
                            span_inputs,
                            loss_mask=ce_mask,
                            loss_weights=ce_w,
                        )
                    ),
                    ce_mask,
                )

        if ref_kl_w is not None:
            ref_kl_mask = span_inputs.loss_mask & (ref_kl_w != 0)
            if bool(ref_kl_mask.any()):
                record_output(
                    "ref_kl",
                    ref_kl_loss_fn(
                        replace(
                            span_inputs,
                            loss_mask=ref_kl_mask,
                            loss_weights=ref_kl_w,
                        ),
                        loss_config,
                    ),
                    ref_kl_mask,
                )

    scales = dict(component_loss_scales or {})
    if loss_scale is not None and "rl" not in scales:
        scales["rl"] = loss_scale
    scaled_loss = trainer_logprobs.sum() * 0.0
    for component, component_loss in component_losses.items():
        default_count = (
            component_sequence_counts[component]
            if loss_config.normalization == "sequence"
            else component_token_counts[component]
        )
        scale = scales.get(component, max(default_count, 1))
        if isinstance(scale, Tensor):
            resolved_scale: float | Tensor = torch.clamp_min(
                scale.to(component_loss.device),
                1,
            )
        else:
            resolved_scale = max(float(scale), 1.0)
        scaled_loss = scaled_loss + component_loss / resolved_scale
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
    ) -> ChunkedLmHeadOutput:
        if labels is None:
            return {"logits": super().forward(hidden_states)}
        if temperature is None:
            raise ValueError("temperature is required for chunked logprob LM head")

        batch, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(batch * seq_len, hidden_size).contiguous()
        labels_flat = labels.reshape(batch * seq_len).contiguous()
        inv_temperature = (1.0 / temperature.reshape(batch * seq_len)).contiguous()

        logprobs = _ChunkedLogprobFn.apply(
            hidden_flat,
            self.weight,
            labels_flat,
            inv_temperature,
            self.chunk_size,
        )
        return {
            "logprobs": logprobs.reshape(batch, seq_len),
            "entropy": None,
        }


def _online_logsumexp_update(
    max_values: Tensor,
    sums: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor]:
    chunk_max = torch.amax(logits, dim=-1)
    new_max = torch.maximum(max_values, chunk_max)
    old_scale = torch.exp(max_values - new_max)
    exp_logits = torch.exp(logits - new_max.unsqueeze(-1))
    new_sums = sums * old_scale + exp_logits.sum(dim=-1)
    return new_max, new_sums


def _validate_chunked_logprob_inputs(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    inv_temperature: Tensor,
    chunk_size: int,
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


class _ChunkedLogprobFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: Tensor,
        weight: Tensor,
        labels: Tensor,
        inv_temperature: Tensor,
        chunk_size: int,
    ) -> Tensor:
        _validate_chunked_logprob_inputs(
            hidden,
            weight,
            labels,
            inv_temperature,
            chunk_size,
        )

        device = hidden.device
        token_count = hidden.shape[0]
        vocab_size = weight.shape[0]
        vocab_chunk_size = min(vocab_size, 8192)
        logprobs = torch.empty(token_count, device=device, dtype=torch.float32)
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
            target_logits = torch.zeros(
                chunk_tokens, device=device, dtype=torch.float32
            )

            for vocab_start in range(0, vocab_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                weight_chunk = weight[vocab_start:vocab_end]
                logits = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits.to(torch.float32) * inv_temperature_chunk
                max_values, sums = _online_logsumexp_update(
                    max_values,
                    sums,
                    scaled_logits,
                )

                in_chunk = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(in_chunk):
                    indexes = (labels_chunk[in_chunk] - vocab_start).to(torch.long)
                    target_logits[in_chunk] = scaled_logits[in_chunk, indexes]

            log_z_chunk = max_values + torch.log(sums)
            log_z[start:end] = log_z_chunk
            logprobs[start:end] = target_logits - log_z_chunk

        ctx.save_for_backward(hidden, weight, labels, inv_temperature, log_z)
        ctx.chunk_size = chunk_size
        return logprobs

    @staticmethod
    def backward(ctx, grad_logprobs: Tensor):
        hidden, weight, labels, inv_temperature, log_z = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size
        needs_grad_hidden, needs_grad_weight = ctx.needs_input_grad[:2]
        token_count = hidden.shape[0]
        vocab_size = weight.shape[0]
        vocab_chunk_size = min(vocab_size, 8192)

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
                probs = torch.exp(scaled_logits - log_z_chunk.unsqueeze(-1))

                grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                in_chunk = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(in_chunk):
                    indexes = (labels_chunk[in_chunk] - vocab_start).to(torch.long)
                    grad_logits[in_chunk, indexes] += grad_chunk[in_chunk]
                grad_logits = grad_logits * inv_temperature_chunk

                if grad_hidden is not None:
                    grad_hidden[start:end].add_(
                        grad_logits.to(hidden.dtype) @ weight_chunk
                    )
                if grad_weight is not None:
                    grad_weight[vocab_start:vocab_end].add_(
                        grad_logits.to(weight.dtype).t() @ hidden_chunk
                    )

        return grad_hidden, grad_weight, None, None, None


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
        )

    model.forward = types.MethodType(forward, model)

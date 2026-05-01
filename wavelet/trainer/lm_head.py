from __future__ import annotations

import logging
import types
from typing import TypedDict

import torch
from torch import Tensor, nn


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
            target_logits = torch.zeros(chunk_tokens, device=device, dtype=torch.float32)

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

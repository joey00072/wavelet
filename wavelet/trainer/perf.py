"""Model FLOP estimates and model FLOPs utilization metrics."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from wavelet.trainer.types import LORA_STATE_ATTRS


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    nested = getattr(config, "text_config", None)
    return nested if isinstance(nested, PretrainedConfig) else config


def _positive_int(config: PretrainedConfig, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _positive_ints(config: PretrainedConfig, *names: str) -> tuple[int, ...] | None:
    """Return every named positive int, or None if any is missing."""
    values = [_positive_int(config, name) for name in names]
    if any(value is None for value in values):
        return None
    return tuple(cast(int, value) for value in values)


def estimate_active_matmul_parameters(config: PretrainedConfig) -> int | None:
    """Estimate parameters participating in matmuls for one token."""
    config = _text_config(config)
    vocab_size = _positive_int(config, "vocab_size")
    hidden_size = _positive_int(config, "hidden_size", "n_embd")
    num_layers = _positive_int(config, "num_hidden_layers", "n_layer")
    num_heads = _positive_int(config, "num_attention_heads", "n_head")
    if (
        vocab_size is None
        or hidden_size is None
        or num_layers is None
        or num_heads is None
    ):
        return None

    attention_params = _attention_projection_parameters(
        config,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    mlp_params = _active_mlp_parameters(
        config,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    lm_head_params = vocab_size * hidden_size
    return attention_params + mlp_params + lm_head_params


def _attention_projection_parameters(
    config: PretrainedConfig,
    *,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
) -> int:
    mla_dims = _positive_ints(
        config,
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "qk_rope_head_dim",
        "qk_nope_head_dim",
        "v_head_dim",
    )
    if mla_dims is not None:
        (
            q_lora_rank,
            kv_lora_rank,
            qk_head_dim,
            qk_rope_head_dim,
            qk_nope_head_dim,
            value_head_dim,
        ) = mla_dims
        query = num_layers * (
            hidden_size * q_lora_rank + q_lora_rank * num_heads * qk_head_dim
        )
        key_value = num_layers * (
            hidden_size * (kv_lora_rank + qk_rope_head_dim)
            + kv_lora_rank * num_heads * (qk_nope_head_dim + value_head_dim)
        )
        output = num_layers * num_heads * value_head_dim * hidden_size
        return query + key_value + output

    head_dim = _positive_int(config, "head_dim") or hidden_size // num_heads
    num_kv_heads = _positive_int(config, "num_key_value_heads") or num_heads
    query = num_layers * hidden_size * num_heads * head_dim
    key_value = 2 * num_layers * hidden_size * num_kv_heads * head_dim
    output = num_layers * hidden_size * num_heads * head_dim
    return query + key_value + output


def _active_mlp_parameters(
    config: PretrainedConfig,
    *,
    hidden_size: int,
    num_layers: int,
) -> int:
    intermediate_size = (
        _positive_int(config, "intermediate_size", "n_inner") or 4 * hidden_size
    )
    experts_per_token = _positive_int(
        config,
        "num_experts_per_tok",
        "num_experts_per_token",
    )
    if experts_per_token is None:
        return num_layers * 3 * intermediate_size * hidden_size

    dense_layers = getattr(config, "first_k_dense_replace", 0)
    if not isinstance(dense_layers, int) or isinstance(dense_layers, bool):
        dense_layers = 0
    dense_layers = min(max(dense_layers, 0), num_layers)
    sparse_layers = num_layers - dense_layers
    moe_intermediate = (
        _positive_int(config, "moe_intermediate_size") or intermediate_size
    )
    dense = dense_layers * 3 * intermediate_size * hidden_size
    routed = sparse_layers * experts_per_token * 3 * moe_intermediate * hidden_size
    shared_intermediate = _positive_int(config, "shared_expert_intermediate_size")
    if shared_intermediate is None:
        shared_experts = _positive_int(
            config,
            "num_shared_experts",
            "n_shared_experts",
        )
        shared_intermediate = (shared_experts or 0) * moe_intermediate
    shared = sparse_layers * 3 * shared_intermediate * hidden_size
    total_experts = _positive_int(config, "num_experts", "n_routed_experts") or 0
    router = sparse_layers * total_experts * hidden_size
    return dense + routed + shared + router


def estimate_attention_flops_per_token(
    config: PretrainedConfig,
    *,
    seq_len: int,
) -> int | None:
    """Estimate causal-attention forward and backward FLOPs for one token."""
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")
    config = _text_config(config)
    hidden_size = _positive_int(config, "hidden_size", "n_embd")
    num_layers = _positive_int(config, "num_hidden_layers", "n_layer")
    if hidden_size is None or num_layers is None:
        return None
    return 12 * num_layers * hidden_size * seq_len


def estimate_training_flops_per_token(model: nn.Module, *, seq_len: int) -> int | None:
    """Estimate model FLOPs per token for full fine-tuning or LoRA training."""
    config = _model_config(model)
    if config is None:
        return None
    active_parameters = estimate_active_matmul_parameters(config)
    attention_flops = estimate_attention_flops_per_token(config, seq_len=seq_len)
    if active_parameters is None or attention_flops is None:
        return None

    lora_parameters = 0
    other_trainable_parameters = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_lora_parameter(name):
            lora_parameters += parameter.numel()
        else:
            other_trainable_parameters += parameter.numel()
    if lora_parameters:
        return (
            4 * active_parameters
            + 2 * other_trainable_parameters
            + 6 * lora_parameters
            + attention_flops
        )
    return 6 * active_parameters + attention_flops


def _model_config(model: nn.Module) -> PretrainedConfig | None:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        config = getattr(current, "config", None)
        if isinstance(config, PretrainedConfig):
            return config
        nested = getattr(current, "module", None)
        if not isinstance(nested, nn.Module):
            break
        current = nested
    return None


def _is_lora_parameter(name: str) -> bool:
    return any(marker in name for marker in LORA_STATE_ATTRS)


def model_compute_dtype(model: nn.Module) -> torch.dtype | None:
    """Return the dtype holding the largest number of floating-point parameters."""
    counts: dict[torch.dtype, int] = {}
    for parameter in model.parameters():
        if parameter.is_floating_point():
            counts[parameter.dtype] = counts.get(parameter.dtype, 0) + parameter.numel()
    return max(counts.items(), key=lambda item: item[1])[0] if counts else None


def peak_flops_per_second(
    device_name: str,
    *,
    dtype: torch.dtype,
) -> float | None:
    """Return dense, non-sparse peak tensor FLOPs for supported accelerators."""
    if dtype not in {torch.bfloat16, torch.float16}:
        return None
    name = device_name.upper()
    if "GB200" in name or "GB300" in name:
        return 2.5e15
    if "B200" in name or "B300" in name:
        return 2.25e15
    if "H100" in name or "H200" in name:
        if "NVL" in name:
            return 835e12
        if "PCIE" in name:
            return 756e12
        return 989e12
    if "A100" in name:
        return 312e12
    if "MI300X" in name or "MI325X" in name:
        return 1307.4e12
    return None


def training_flop_metrics(
    *,
    flops_per_token: int | None,
    model_tokens: float,
    elapsed_seconds: float,
    world_size: int,
    dtype: torch.dtype | None,
    device_name: str | None = None,
) -> dict[str, float]:
    """Build throughput-independent FLOP and MFU metrics for one optimizer step."""
    if flops_per_token is None:
        return {}
    metrics = {"perf/model_flops_per_token": float(flops_per_token)}
    if dtype is None or model_tokens <= 0 or elapsed_seconds <= 0 or world_size < 1:
        return metrics
    if device_name is None:
        if not torch.cuda.is_available():
            return metrics
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    peak = peak_flops_per_second(device_name, dtype=dtype)
    if peak is None:
        return metrics
    metrics["perf/peak_flops_per_second_per_gpu"] = peak
    metrics["perf/mfu"] = (
        100.0 * flops_per_token * model_tokens / elapsed_seconds / peak / world_size
    )
    return metrics

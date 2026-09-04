"""Hugging Face MoE router controls and load-balance metrics."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import Tensor, nn

from wavelet.configs.sft import ModelConfig

HF_MOE_ROUTER_CLASS_NAMES = frozenset(
    {
        "GptOssTopKRouter",
        "Qwen3MoeTopKRouter",
    }
)


def hf_moe_routers(model: nn.Module) -> list[nn.Module]:
    """Return supported Hugging Face token-choice routers in model order."""
    return [
        module
        for module in model.modules()
        if type(module).__name__ in HF_MOE_ROUTER_CLASS_NAMES
    ]


def configure_hf_moe_routers(model: nn.Module, config: ModelConfig) -> int:
    """Apply router controls and enable router outputs for metrics."""
    routers = hf_moe_routers(model)
    if config.freeze_moe_router and not routers:
        raise ValueError(
            "model.freeze_moe_router=true requires a supported Hugging Face MoE "
            "model (Qwen3-MoE or GPT-OSS)."
        )

    if routers:
        model.config.output_router_logits = True
        # Wavelet owns its SFT/RL objective. Recording router logits must not
        # silently add the Transformers auxiliary router loss to fused SFT.
        if hasattr(model.config, "router_aux_loss_coef"):
            model.config.router_aux_loss_coef = 0.0

    for router in routers:
        if config.freeze_moe_router:
            for parameter in router.parameters():
                parameter.requires_grad_(False)
        if config.moe_router_dtype == "float32":
            _configure_fp32_router(router)
    return len(routers)


def _configure_fp32_router(router: nn.Module) -> None:
    router.to(dtype=torch.float32)
    if getattr(router, "_wavelet_fp32_router", False):
        return
    object.__setattr__(router, "_wavelet_original_forward", router.forward)
    object.__setattr__(
        router,
        "forward",
        MethodType(_fp32_router_forward, router),
    )
    object.__setattr__(router, "_wavelet_fp32_router", True)


def _fp32_router_forward(
    router: nn.Module,
    hidden_states: Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    original_forward = router._wavelet_original_forward  # type: ignore[attr-defined]
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        return original_forward(hidden_states.float(), *args, **kwargs)


def moe_load_balance_metrics(
    model: nn.Module,
    outputs: object,
    *,
    token_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Measure expert-load violation and selected routing confidence."""
    if isinstance(outputs, dict):
        router_logits = outputs.get("router_logits")
    else:
        router_logits = getattr(outputs, "router_logits", None)
    if not isinstance(router_logits, (list, tuple)):
        return {}

    top_k = int(getattr(model.config, "num_experts_per_tok", 1))
    flat_mask = token_mask.bool().reshape(-1) if token_mask is not None else None
    violations: list[Tensor] = []
    confidences: list[Tensor] = []
    with torch.no_grad():
        for layer_logits in router_logits:
            if not isinstance(layer_logits, Tensor) or layer_logits.numel() == 0:
                continue
            logits = layer_logits.detach().float().reshape(-1, layer_logits.shape[-1])
            if flat_mask is not None and flat_mask.numel() == logits.shape[0]:
                logits = logits[flat_mask]
            if logits.numel() == 0:
                continue
            selected = logits.topk(min(top_k, logits.shape[-1]), dim=-1).indices
            counts = torch.bincount(
                selected.reshape(-1),
                minlength=logits.shape[-1],
            ).float()
            balanced_load = counts.mean()
            if balanced_load <= 0:
                continue
            violations.append((counts.max() - balanced_load) / balanced_load)
            probabilities = logits.softmax(dim=-1)
            confidences.append(
                probabilities.gather(dim=-1, index=selected).sum(dim=-1).mean()
            )

    if not violations:
        return {}
    violation_values = torch.stack(violations)
    confidence_values = torch.stack(confidences)
    return {
        "moe/max_vio": violation_values.mean(),
        "moe/max_vio/max": violation_values.max(),
        "moe/routing_confidence": confidence_values.mean(),
    }

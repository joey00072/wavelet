"""Hugging Face MoE router controls and load-balance metrics."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from wavelet.configs.sft import ModelConfig

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from wavelet.trainer.distributed import ParallelDims

HF_MOE_ROUTER_CLASS_NAMES = frozenset(
    {
        "GptOssTopKRouter",
        "Qwen3MoeTopKRouter",
    }
)

HF_MOE_EXPERT_CLASS_NAMES = frozenset(
    {
        "GptOssExperts",
        "Qwen3MoeExperts",
    }
)


class _AllToAll(torch.autograd.Function):
    """Autograd-aware variable-split all-to-all for routed token tensors."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        output_splits: tuple[int, ...],
        input_splits: tuple[int, ...],
        group: ProcessGroup,
    ) -> Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        output = tensor.new_empty((sum(output_splits), *tensor.shape[1:]))
        dist.all_to_all_single(
            output,
            tensor.contiguous(),
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        grad_input = grad_output.new_empty(
            (sum(ctx.input_splits), *grad_output.shape[1:])
        )
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=list(ctx.input_splits),
            input_split_sizes=list(ctx.output_splits),
            group=ctx.group,
        )
        return grad_input, None, None, None


def _all_to_all(
    tensor: Tensor,
    *,
    output_splits: tuple[int, ...],
    input_splits: tuple[int, ...],
    group: ProcessGroup,
) -> Tensor:
    return _AllToAll.apply(tensor, output_splits, input_splits, group)


def _all_to_all_indices(
    tensor: Tensor,
    *,
    output_splits: tuple[int, ...],
    input_splits: tuple[int, ...],
    group: ProcessGroup,
) -> Tensor:
    output = tensor.new_empty((sum(output_splits), *tensor.shape[1:]))
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    return output


def hf_moe_routers(model: nn.Module) -> list[nn.Module]:
    """Return supported Hugging Face token-choice routers in model order."""
    return [
        module
        for module in model.modules()
        if type(module).__name__ in HF_MOE_ROUTER_CLASS_NAMES
    ]


def hf_moe_experts(model: nn.Module) -> list[nn.Module]:
    """Return supported Hugging Face expert collections in model order."""
    return [
        module
        for module in model.modules()
        if type(module).__name__ in HF_MOE_EXPERT_CLASS_NAMES
    ]


def configure_hf_moe_expert_parallel(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> int:
    """Shard supported HF experts and install token dispatch across EP ranks."""
    if not parallel_dims.ep_enabled:
        return 0
    experts = hf_moe_experts(model)
    if not experts:
        raise ValueError(
            "fsdp.ep>1 requires a supported Hugging Face MoE model "
            "(Qwen3-MoE or GPT-OSS)."
        )

    ep_mesh = parallel_dims.get_mesh("ep")
    for expert_module in experts:
        _configure_expert_module(expert_module, ep_mesh)
    return len(experts)


def _configure_expert_module(module: nn.Module, ep_mesh: DeviceMesh) -> None:
    if getattr(module, "_wavelet_expert_parallel", False):
        return
    global_experts = int(module.num_experts)  # type: ignore[attr-defined]
    ep_size = ep_mesh.size()
    if global_experts % ep_size:
        raise ValueError(
            f"MoE expert count {global_experts} must be divisible by fsdp.ep={ep_size}."
        )

    for name, parameter in tuple(module.named_parameters(recurse=False)):
        sharded = distribute_tensor(
            parameter,
            ep_mesh,
            [Shard(0)],
            src_data_rank=None,
        )
        module.register_parameter(name, nn.Parameter(sharded))

    object.__setattr__(module, "_wavelet_ep_group", ep_mesh.get_group())
    object.__setattr__(module, "_wavelet_ep_size", ep_size)
    object.__setattr__(module, "_wavelet_ep_local_experts", global_experts // ep_size)
    object.__setattr__(module, "_wavelet_ep_global_experts", global_experts)
    object.__setattr__(module, "forward", MethodType(_expert_parallel_forward, module))
    object.__setattr__(module, "_wavelet_expert_parallel", True)


def _expert_parallel_forward(
    module: nn.Module,
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
) -> Tensor:
    if hidden_states.ndim != 2:
        raise ValueError(
            "Expert-parallel hidden states must have shape [tokens, hidden]."
        )
    group = module._wavelet_ep_group  # type: ignore[attr-defined]
    ep_size = int(module._wavelet_ep_size)  # type: ignore[attr-defined]
    local_experts = int(module._wavelet_ep_local_experts)  # type: ignore[attr-defined]

    flat_experts = top_k_index.reshape(-1)
    flat_weights = top_k_weights.reshape(-1)
    token_indices = torch.arange(
        hidden_states.shape[0], device=hidden_states.device
    ).repeat_interleave(top_k_index.shape[-1])
    destination_ranks = torch.div(flat_experts, local_experts, rounding_mode="floor")
    order = torch.argsort(destination_ranks, stable=True)
    send_counts_tensor = torch.bincount(
        destination_ranks,
        minlength=ep_size,
    ).to(device=hidden_states.device, dtype=torch.int64)
    receive_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(receive_counts_tensor, send_counts_tensor, group=group)
    send_counts = tuple(int(value) for value in send_counts_tensor.tolist())
    receive_counts = tuple(int(value) for value in receive_counts_tensor.tolist())

    routed_states = _all_to_all(
        hidden_states[token_indices[order]],
        output_splits=receive_counts,
        input_splits=send_counts,
        group=group,
    )
    routed_weights = _all_to_all(
        flat_weights[order],
        output_splits=receive_counts,
        input_splits=send_counts,
        group=group,
    )
    routed_experts = _all_to_all_indices(
        flat_experts[order].remainder(local_experts),
        output_splits=receive_counts,
        input_splits=send_counts,
        group=group,
    )

    routed_output = _run_local_experts(module, routed_states, routed_experts)
    routed_output = routed_output * routed_weights.unsqueeze(-1)
    returned_output = _all_to_all(
        routed_output,
        output_splits=send_counts,
        input_splits=receive_counts,
        group=group,
    )
    output = torch.zeros_like(hidden_states)
    output.index_add_(0, token_indices[order], returned_output.to(output.dtype))
    return output


def _local_parameter(module: nn.Module, name: str) -> Tensor:
    parameter = getattr(module, name)
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def _run_local_experts(
    module: nn.Module,
    hidden_states: Tensor,
    expert_indices: Tensor,
) -> Tensor:
    output = torch.zeros_like(hidden_states)
    local_experts = int(module._wavelet_ep_local_experts)  # type: ignore[attr-defined]
    gate_up_proj = _local_parameter(module, "gate_up_proj")
    down_proj = _local_parameter(module, "down_proj")
    is_gpt_oss = type(module).__name__ == "GptOssExperts"
    for expert_index in range(local_experts):
        positions = torch.where(expert_indices == expert_index)[0]
        if positions.numel() == 0:
            continue
        current = hidden_states[positions]
        if is_gpt_oss:
            gate_up = current @ gate_up_proj[expert_index]
            gate_up = (
                gate_up + _local_parameter(module, "gate_up_proj_bias")[expert_index]
            )
            activated = module._apply_gate(gate_up)  # type: ignore[attr-defined]
            current_output = activated @ down_proj[expert_index]
            current_output = (
                current_output
                + _local_parameter(module, "down_proj_bias")[expert_index]
            )
        else:
            gate, up = nn.functional.linear(
                current,
                gate_up_proj[expert_index],
            ).chunk(2, dim=-1)
            activated = module.act_fn(gate) * up  # type: ignore[attr-defined]
            current_output = nn.functional.linear(
                activated,
                down_proj[expert_index],
            )
        output.index_copy_(0, positions, current_output.to(output.dtype))
    return output


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

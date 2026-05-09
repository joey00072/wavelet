from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.distributed
from peft import LoraConfig as PeftLoraConfig
from peft import (
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from safetensors.torch import save_file as save_safetensors
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel

from wavelet.configs.sft import LoRAConfig
from wavelet.distributed.parallel_dims import ParallelDims
from wavelet.trainer.debug import DEBUG_LORA_TARGET_MODULES


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "experts",
]
LORA_STATE_ATTRS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
TP_REPLICATED_LORA_ATTRS = {
    "colwise": ("lora_A", "lora_embedding_A"),
    "rowwise": ("lora_B", "lora_embedding_B"),
}


def apply_lora(
    model: PreTrainedModel,
    config: LoRAConfig | None,
    *,
    match_base_dtype: bool = False,
    lora_dtype: torch.dtype | None = None,
) -> PreTrainedModel:
    if config is None:
        return model
    if isinstance(model, PeftModel):
        if match_base_dtype:
            _align_lora_dtypes(model)
        return model
    _normalize_hf_tp_linear_feature_metadata(model)
    target_modules = _resolve_lora_target_modules(model, config)
    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=int(config.alpha),
        lora_dropout=config.dropout,
        target_modules=target_modules,
        modules_to_save=config.modules_to_save or None,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    if lora_dtype is not None:
        _cast_lora_dtypes(model, lora_dtype)
    elif match_base_dtype:
        _align_lora_dtypes(model)
    return model


def prepare_hf_tp_lora_for_training(
    model: nn.Module,
    parallel_dims: ParallelDims | None,
) -> None:
    if not _tp_distributed_enabled(parallel_dims):
        return

    try:
        from transformers.integrations.tensor_parallel import all_reduce_forward
    except ImportError:
        return

    tp_mesh = parallel_dims.get_mesh("tp")
    for module in _hf_tp_lora_modules(model, "rowwise"):
        for child in _lora_children(module, "lora_B", "lora_embedding_B"):
            if not _needs_lora_allreduce_hook(child):
                continue

            def _hook(
                _module: nn.Module,
                _inputs: tuple[object, ...],
                output: torch.Tensor,
                *,
                mesh=tp_mesh,
            ) -> torch.Tensor:
                return all_reduce_forward(output, mesh)

            child.register_forward_hook(_hook)
            child._wavelet_tp_lora_allreduce_hook = True


def sync_hf_tp_lora_replicated_grads(
    model: nn.Module,
    parallel_dims: ParallelDims | None,
) -> None:
    if not _tp_distributed_enabled(parallel_dims):
        return

    group = _mesh_process_group(parallel_dims, "tp")
    if torch.distributed.get_world_size(group=group) <= 1:
        return

    for module, tp_plan in _hf_tp_lora_modules_with_plan(model):
        replicated_attrs = TP_REPLICATED_LORA_ATTRS[tp_plan]
        for parameter in _lora_parameters(module, *replicated_attrs):
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(
                parameter.grad,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )


def save_lora_adapter_snapshot(
    model: PreTrainedModel,
    output_dir: Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    is_main_process: bool = True,
    parallel_dims: ParallelDims | None = None,
) -> Path:
    """Save only the mutable LoRA adapter files needed for hot policy reload."""
    if not isinstance(model, PeftModel):
        raise TypeError("Lightweight policy snapshots require a PeftModel.")

    target = output_dir / "adapter"
    if _model_uses_hf_tensor_parallel_lora(model):
        state_dict = _gather_hf_tp_lora_state_dict(
            model,
            state_dict=state_dict,
            parallel_dims=parallel_dims,
        )
        if state_dict is None:
            return target
    elif not is_main_process:
        return target

    target.mkdir(parents=True, exist_ok=True)
    adapter_name = _active_adapter_name(model)
    peft_config = model.peft_config[adapter_name]
    peft_config.save_pretrained(target)
    lora_state = get_peft_model_state_dict(
        model,
        state_dict=state_dict,
        adapter_name=adapter_name,
    )
    cpu_state = {
        _strip_fsdp_wrapped_module_segments(key): value.detach().cpu().contiguous()
        for key, value in lora_state.items()
    }
    save_safetensors(cpu_state, target / "adapter_model.safetensors")
    return target


def save_lora_adapter_snapshot_from_fsdp(
    model: FSDP,
    output_dir: Path,
    *,
    is_main_process: bool = True,
    parallel_dims: ParallelDims | None = None,
) -> Path:
    """Save a PEFT LoRA adapter from an FSDP-wrapped model without a full state dict."""
    unwrapped = _unwrap_model(model)
    if not isinstance(unwrapped, PeftModel):
        raise TypeError("FSDP lightweight policy snapshots require a wrapped PeftModel.")

    state_dict = _gather_fsdp_lora_state_dict(
        model,
        unwrapped,
        parallel_dims=parallel_dims,
    )
    return save_lora_adapter_snapshot(
        unwrapped,
        output_dir,
        state_dict=state_dict,
        is_main_process=is_main_process,
        parallel_dims=parallel_dims,
    )


def _save_lora_adapter_snapshot_from_fsdp_full_params(
    model: FSDP,
    output_dir: Path,
    *,
    is_main_process: bool = True,
) -> Path:
    unwrapped = _unwrap_model(model)
    if not isinstance(unwrapped, PeftModel):
        raise TypeError("FSDP lightweight policy snapshots require a wrapped PeftModel.")
    with FSDP.summon_full_params(
        model,
        recurse=True,
        writeback=False,
        rank0_only=True,
        offload_to_cpu=True,
    ):
        return save_lora_adapter_snapshot(
            unwrapped,
            output_dir,
            state_dict=None,
            is_main_process=is_main_process,
        )


def _tp_distributed_enabled(parallel_dims: ParallelDims | None) -> bool:
    return bool(
        parallel_dims is not None
        and parallel_dims.tp_enabled
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _needs_lora_allreduce_hook(child: object) -> bool:
    return isinstance(child, nn.Module) and not getattr(
        child,
        "_wavelet_tp_lora_allreduce_hook",
        False,
    )


def _hf_tp_lora_modules(model: nn.Module, *plans: str) -> list[nn.Module]:
    return [
        module
        for module, tp_plan in _hf_tp_lora_modules_with_plan(model)
        if tp_plan in plans
    ]


def _hf_tp_lora_modules_with_plan(model: nn.Module) -> list[tuple[nn.Module, str]]:
    modules: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        tp_plan = _hf_tp_plan(module)
        if tp_plan in TP_REPLICATED_LORA_ATTRS:
            modules.append((module, tp_plan))
    return modules


def _hf_tp_plan(module: nn.Module) -> str | None:
    if not _is_lora_wrapped(module):
        return None
    base_layer = getattr(module, "base_layer", None)
    tp_plan = getattr(base_layer, "_hf_tp_plan", None)
    return tp_plan if isinstance(tp_plan, str) else None


def _lora_children(module: nn.Module, *attrs: str) -> list[object]:
    children: list[object] = []
    for attr in attrs:
        container = getattr(module, attr, None)
        if container is None:
            continue
        if isinstance(container, dict):
            children.extend(container.values())
        elif hasattr(container, "values"):
            children.extend(list(container.values()))
        else:
            children.append(container)
    return children


def _lora_parameters(module: nn.Module, *attrs: str) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for child in _lora_children(module, *attrs):
        if isinstance(child, nn.Parameter):
            parameters.append(child)
            continue
        if isinstance(child, nn.Module):
            parameters.extend(child.parameters())
    return parameters


def _normalize_hf_tp_linear_feature_metadata(model: nn.Module) -> None:
    for module in model.modules():
        if getattr(module, "_hf_tp_plan", None) not in {"colwise", "rowwise"}:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        module.out_features = int(weight.shape[0])
        module.in_features = int(weight.shape[1])


def _cast_lora_dtypes(model: nn.Module, lora_dtype: torch.dtype) -> None:
    for _, wrapped in model.named_modules():
        if not _is_lora_wrapped(wrapped):
            continue
        for attr in ("lora_A", "lora_B"):
            container = getattr(wrapped, attr, None)
            if container:
                for child in container.values():
                    child.to(dtype=lora_dtype)


def _model_uses_hf_tensor_parallel_lora(model: PeftModel) -> bool:
    return bool(_hf_tp_lora_modules_with_plan(model))


def _gather_hf_tp_lora_state_dict(
    model: PeftModel,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    parallel_dims: ParallelDims | None = None,
) -> dict[str, torch.Tensor] | None:
    local_state = state_dict or _local_lora_state_dict(model)
    if not torch.distributed.is_initialized():
        return local_state

    group = _mesh_process_group(parallel_dims, "tp")
    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, torch.Tensor] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_state, group=group)
    if torch.distributed.get_rank() != 0:
        return None

    state: dict[str, torch.Tensor] = {}
    for key, value in local_state.items():
        gather_dim = _hf_tp_lora_gather_dim(model, key)
        if gather_dim is None:
            state[key] = value
            continue
        parts = _state_parts(gathered, key)
        state[key] = torch.cat(parts, dim=gather_dim).contiguous()
    return state


def _local_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.named_parameters()
        if "lora_" in name and value.numel() > 0
    }


def _hf_tp_lora_gather_dim(model: PeftModel, key: str) -> int | None:
    module_name, attr = _split_lora_state_key(key)
    if module_name is None or attr is None:
        return None
    module = dict(model.named_modules()).get(module_name)
    if module is None:
        return None
    base_layer = getattr(module, "base_layer", None)
    tp_plan = getattr(base_layer, "_hf_tp_plan", None)
    if tp_plan == "colwise" and attr in {"lora_B", "lora_embedding_B"}:
        return 0
    if tp_plan == "rowwise" and attr in {"lora_A", "lora_embedding_A"}:
        return 1
    return None


def _split_lora_state_key(key: str) -> tuple[str | None, str | None]:
    for attr in LORA_STATE_ATTRS:
        marker = f".{attr}."
        if marker in key:
            return key.split(marker, 1)[0], attr
    return None, None


def _strip_fsdp_wrapped_module_segments(key: str) -> str:
    return key.replace("._fsdp_wrapped_module.", ".").removeprefix(
        "_fsdp_wrapped_module."
    )


def _lora_parameter_shapes(model: PeftModel) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for module_name, module in model.named_modules():
        for attr in LORA_STATE_ATTRS:
            container = getattr(module, attr, None)
            if container is None:
                continue
            for adapter_name, child in container.items():
                weight = getattr(child, "weight", child)
                if not isinstance(weight, nn.Parameter):
                    continue
                if isinstance(child, nn.Linear):
                    shape = (child.out_features, child.in_features)
                else:
                    shape = tuple(weight.shape)
                shapes[f"{module_name}.{attr}.{adapter_name}.weight"] = shape
    return shapes


def _gather_fsdp_lora_state_dict(
    model: FSDP,
    unwrapped: PeftModel,
    *,
    parallel_dims: ParallelDims | None = None,
) -> dict[str, torch.Tensor] | None:
    if not torch.distributed.is_initialized():
        return {
            name.removeprefix("_fsdp_wrapped_module."): value.detach()
            .cpu()
            .contiguous()
            for name, value in model.named_parameters()
            if "lora_" in name and value.numel() > 0
        }

    expected_shapes = _lora_parameter_shapes(unwrapped)
    local_state = {
        name.removeprefix("_fsdp_wrapped_module."): value.detach().cpu().reshape(-1)
        for name, value in model.named_parameters()
        if name.removeprefix("_fsdp_wrapped_module.") in expected_shapes
        and value.numel() > 0
    }

    group = _mesh_process_group(parallel_dims, "hsdp")
    group_rank = _distributed_group_rank(group)
    group_world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, torch.Tensor] | None] | None
    if group_rank == 0:
        gathered = [None for _ in range(group_world_size)]
    else:
        gathered = None
    torch.distributed.gather_object(
        local_state,
        gathered,
        group=group,
        group_dst=0 if group is not None else None,
        dst=0 if group is None else None,
    )
    if group_rank != 0:
        return None
    if gathered is None:
        raise RuntimeError("FSDP LoRA gather returned no state on rank 0.")

    state: dict[str, torch.Tensor] = {}
    for name, shape in expected_shapes.items():
        parts = _state_parts(gathered, name)
        flat = torch.cat(parts, dim=0)
        expected_numel = _numel_from_shape(shape)
        if flat.numel() != expected_numel:
            raise RuntimeError(
                "FSDP LoRA gather produced the wrong size for "
                f"{name}: {flat.numel()} != {expected_numel}."
            )
        state[name] = flat.reshape(shape).contiguous()
    return state


def _state_parts(
    gathered: list[dict[str, torch.Tensor] | None],
    key: str,
) -> list[torch.Tensor]:
    parts = [
        shard[key]
        for shard in gathered
        if shard is not None and key in shard and shard[key].numel() > 0
    ]
    if not parts:
        raise RuntimeError(f"LoRA state gather found no shards for {key}.")
    return parts


def _numel_from_shape(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _mesh_process_group(
    parallel_dims: ParallelDims | None,
    name: str,
) -> object | None:
    if parallel_dims is None:
        return None
    if name == "tp" and not parallel_dims.tp_enabled:
        return None
    if name == "hsdp" and not parallel_dims.fsdp_enabled:
        return None
    return parallel_dims.get_mesh(name).get_group()


def _distributed_group_rank(group: object | None) -> int:
    if group is None:
        return torch.distributed.get_rank()
    return torch.distributed.get_rank(group=group)


def _active_adapter_name(model: PeftModel) -> str:
    active_adapters = getattr(model, "active_adapters", None)
    adapters = active_adapters() if callable(active_adapters) else active_adapters
    if not adapters:
        return "default"
    return str(adapters[0])


def _is_lora_wrapped(module: nn.Module) -> bool:
    return hasattr(module, "base_layer") and hasattr(module, "lora_B")


def _resolve_lora_target_modules(
    model: PreTrainedModel,
    config: LoRAConfig,
) -> list[str]:
    configured = list(config.target_modules)
    if configured != DEFAULT_LORA_TARGET_MODULES:
        return configured
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "gpt2":
        return configured
    return DEBUG_LORA_TARGET_MODULES


def _align_lora_dtypes(model: nn.Module) -> None:
    for module in model.modules():
        if not _is_lora_wrapped(module):
            continue
        base_weight = getattr(module.base_layer, "weight", None)
        if base_weight is None:
            continue
        target_dtype = base_weight.dtype
        target_device = base_weight.device
        for attr in LORA_STATE_ATTRS:
            container = getattr(module, attr, None)
            if container is None:
                continue
            for child in container.values():
                if isinstance(child, nn.Module):
                    child.to(device=target_device, dtype=target_dtype)
                elif isinstance(child, nn.Parameter):
                    child.data = child.data.to(
                        device=target_device,
                        dtype=target_dtype,
                    )


def _unwrap_model(model: nn.Module) -> PreTrainedModel:
    current = model
    while hasattr(current, "module"):
        current = cast(nn.Module, getattr(current, "module"))
    return cast(PreTrainedModel, current)

"""Gather model tensors for policy weight export and NCCL broadcast."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from wavelet.transport.weights.nccl import (
    _LAYER_KEY_RE,
    NCCLWeightBroadcaster,
    _partition_state_dict,
)
from wavelet.transport.weights.wire import _convert_layer_to_hf
from wavelet.utils.modules import unwrap_model as _unwrap_model


def _model_named_tensors(model: nn.Module) -> dict[str, Tensor]:
    """Return parameter and buffer references without materializing full state."""
    unwrapped = _unwrap_model(model)
    tensors = dict(unwrapped.named_parameters())
    for prefix, module in unwrapped.named_modules():
        for name, buffer in module.named_buffers(recurse=False):
            if name not in module._non_persistent_buffers_set:
                tensors[f"{prefix}.{name}" if prefix else name] = buffer
    return tensors


def _materialize_wire_tensors(
    model: nn.Module,
    state_dict: dict[str, Tensor],
    layer_index: int,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """Gather only this layer's sharded tensors and retain their wire dtype."""
    conversion_model = _unwrap_model(model)
    parameter_names = dict(conversion_model.named_parameters())
    keep_fp32 = getattr(conversion_model, "keep_in_fp32_for_weight_transfer", None)

    def wire_tensor(name: str, tensor: Tensor) -> Tensor:
        if (
            dtype is None
            or name not in parameter_names
            or not tensor.is_floating_point()
        ):
            return tensor
        target = torch.float32 if callable(keep_fp32) and keep_fp32(name) else dtype
        return tensor.to(target)

    owner: nn.Module | None = None
    if layer_index < 0:
        owner = model
    else:
        match = next(
            (match for name in state_dict if (match := _LAYER_KEY_RE.match(name))),
            None,
        )
        if match is not None:
            owner = conversion_model.get_submodule(
                f"{match.group('prefix')}.{layer_index}"
            )

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        FSDP = None  # type: ignore[assignment,misc]

    if FSDP is not None and isinstance(owner, FSDP):
        with FSDP.summon_full_params(
            owner,
            recurse=layer_index >= 0,
            rank0_only=False,
            offload_to_cpu=True,
        ):
            return {
                name: wire_tensor(name, tensor).detach().clone().contiguous()
                for name, tensor in state_dict.items()
            }

    materialized: dict[str, Tensor] = {}
    for name, tensor in state_dict.items():
        tensor = wire_tensor(name, tensor)
        full_tensor = getattr(tensor, "full_tensor", None)
        if callable(full_tensor):
            tensor = full_tensor()
        materialized[name] = tensor.detach().contiguous()
    return materialized


def _iter_layer_state_dicts(
    model: nn.Module, dtype: torch.dtype | None = None
) -> Iterator[dict[str, Tensor]]:
    conversion_model = _unwrap_model(model)
    tensors = _model_named_tensors(model)
    for layer_index, layer in enumerate(_partition_state_dict(tensors)):
        wire_layer = _materialize_wire_tensors(model, layer, layer_index - 1, dtype)
        yield _convert_layer_to_hf(conversion_model, wire_layer, layer_index - 1)


@torch.no_grad()
def broadcast_model(broadcaster: NCCLWeightBroadcaster, model: nn.Module) -> None:
    """Convert a model's layers to wire format and broadcast them via NCCL."""
    broadcaster.broadcast_layers(
        _iter_layer_state_dicts(model, broadcaster.dtype),
        layer_count=len(_partition_state_dict(_model_named_tensors(model))),
    )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import Tensor, nn


NamedTensor = tuple[str, Tensor]


def _require_vllm_nccl() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )
    except ImportError as exc:
        raise ImportError(
            "NCCL weight broadcast requires vLLM NCCL internals. Install vLLM and "
            "run this on CUDA workers."
        ) from exc
    return (
        PyNcclCommunicator,
        StatelessProcessGroup,
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )


def update_info_for_named_tensors(
    named_tensors: Iterable[NamedTensor],
    *,
    packed: bool = False,
) -> dict[str, Any]:
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    for name, tensor in named_tensors:
        names.append(name)
        dtype_names.append(str(tensor.dtype).removeprefix("torch."))
        shapes.append(list(tensor.shape))
    return {
        "names": names,
        "dtype_names": dtype_names,
        "shapes": shapes,
        "packed": packed,
    }


def update_info_for_model(model: nn.Module, *, packed: bool = False) -> dict[str, Any]:
    return update_info_for_named_tensors(model.state_dict().items(), packed=packed)


@dataclass(slots=True)
class NCCLWeightBroadcaster:
    host: str
    port: int
    rank: int
    world_size: int
    device: torch.device | str | int = "cuda"
    timeout_seconds: int = 600
    source_rank: int = 0
    packed: bool = False
    _communicator: Any = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _process_group: Any = field(init=False, repr=False)
    _send_args_type: type[Any] = field(init=False, repr=False)
    _transfer_engine_type: type[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL weight broadcast requires CUDA.")
        (
            communicator_type,
            process_group_type,
            send_args_type,
            transfer_engine_type,
        ) = _require_vllm_nccl()
        self._device = torch.device(self.device)
        self._process_group = process_group_type.create(
            host=self.host,
            port=self.port,
            rank=self.rank,
            world_size=self.world_size,
            store_timeout=self.timeout_seconds,
        )
        self._communicator = communicator_type(self._process_group, device=self._device)
        self._send_args_type = send_args_type
        self._transfer_engine_type = transfer_engine_type

    @torch.no_grad()
    def broadcast_named_tensors(self, named_tensors: Iterable[NamedTensor]) -> None:
        if self.rank != self.source_rank:
            raise RuntimeError("Only the source rank can broadcast model weights.")
        args = self._send_args_type(
            group=self._communicator,
            src=self.source_rank,
            packed=self.packed,
            post_iter_func=lambda item: item[1].detach().to(self._device).contiguous(),
        )
        self._transfer_engine_type.trainer_send_weights(iter(named_tensors), args)

    @torch.no_grad()
    def broadcast_model(self, model: nn.Module) -> None:
        self.broadcast_named_tensors(model.state_dict().items())

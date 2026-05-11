from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from wavelet.utils.policy_transfer import NCCL_UPDATE_INFO_FILENAME

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object


def _require_vllm_nccl() -> tuple[type[object], type[object]]:
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
    except ImportError as exc:
        raise ImportError(
            "NCCL weight updates require vLLM NCCL internals. Install vLLM and "
            "run the inference server on CUDA workers."
        ) from exc
    return PyNcclCommunicator, StatelessProcessGroup


def _worker_model(worker: object) -> Module:
    model_runner = worker.model_runner  # type: ignore[attr-defined]
    if hasattr(model_runner.model, "runnable"):
        model = model_runner.model.runnable
    else:
        model = model_runner.model
    assert isinstance(model, Module)
    return model


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for in-place full-weight updates from disk."""

    def liveness_probe(self) -> None:
        return None

    def update_weights_from_path(self, weight_path: str) -> None:
        model = _worker_model(self)

        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        local_source = DefaultModelLoader.Source(
            weight_path,
            revision=None,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        model.load_weights(weights_iterator)  # type: ignore[arg-type]

        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for in-place full-weight updates over NCCL."""

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
    ) -> None:
        if getattr(self, "_wavelet_nccl_communicator", None) is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL weight updates require CUDA.")
        communicator_type, process_group_type = _require_vllm_nccl()

        device = getattr(self, "device", None)
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        local_rank = getattr(device, "index", None)
        if local_rank is None:
            local_rank = torch.cuda.current_device()
        rank = rank_offset + int(local_rank)
        world_size = rank_offset + inference_world_size

        process_group = process_group_type.create(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
            store_timeout=timeout,
        )
        self._wavelet_nccl_communicator = communicator_type(
            process_group,
            device=device,
        )

    def liveness_probe(self) -> None:
        return None

    @torch.no_grad()
    def update_weights_from_path(self, weight_path: str) -> None:
        communicator = getattr(self, "_wavelet_nccl_communicator", None)
        if communicator is None:
            raise RuntimeError(
                "NCCL weight update receiver was not initialized. Call "
                "/init_broadcaster before /load_policy."
            )

        update_info_path = Path(weight_path) / NCCL_UPDATE_INFO_FILENAME
        update_info = json.loads(update_info_path.read_text())
        model = _worker_model(self)
        for name, dtype_name, shape in zip(
            update_info["names"],
            update_info["dtype_names"],
            update_info["shapes"],
            strict=True,
        ):
            dtype = getattr(torch, dtype_name)
            weight = torch.empty(shape, dtype=dtype, device=communicator.device)
            communicator.broadcast(weight, src=0, stream=torch.cuda.current_stream())
            model.load_weights([(name, weight)])  # type: ignore[arg-type]
            del weight

        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)

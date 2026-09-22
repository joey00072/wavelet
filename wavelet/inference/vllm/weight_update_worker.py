"""vLLM worker extensions for Wavelet weight updates."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import Module

from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.v1.worker.gpu_worker import Worker
from wavelet.transport.weights.handshake import NCCL_UPDATE_INFO_FILENAME
from wavelet.transport.weights.nccl import (
    NCCLWeightBroadcaster,
    _broadcast_bytes,
    _broadcast_integer,
    _indexed_cuda_device,
    _partition_state_dict,
    _require_vllm_nccl,
    nccl_world_size,
)
from wavelet.transport.weights.wire import _convert_layer_to_hf

NamedTensor = tuple[str, torch.Tensor]

__all__ = [
    "FileSystemWeightUpdateWorker",
    "NCCLWeightBroadcaster",
    "NCCLWeightUpdateWorker",
    "_convert_layer_to_hf",
    "_indexed_cuda_device",
    "_partition_state_dict",
]


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
        model_loader_factory = get_model_loader
        default_model_loader = DefaultModelLoader
        current_config = set_current_vllm_config
        initialize_reload = initialize_layerwise_reload
        finalize_reload = finalize_layerwise_reload
        if any(
            value is None
            for value in (
                model_loader_factory,
                default_model_loader,
                current_config,
                initialize_reload,
                finalize_reload,
            )
        ):
            from vllm.config import set_current_vllm_config as current_config
            from vllm.model_executor.model_loader import (
                DefaultModelLoader as default_model_loader,
            )
            from vllm.model_executor.model_loader import (
                get_model_loader as model_loader_factory,
            )
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload as finalize_reload,
            )
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload as initialize_reload,
            )

        model = _worker_model(self)

        model_loader = model_loader_factory(self.load_config)
        assert isinstance(model_loader, default_model_loader)
        local_source = default_model_loader.Source(
            weight_path,
            revision=None,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        device = next(model.parameters()).device
        with torch.device(device), current_config(self.vllm_config):
            initialize_reload(model)
            model.load_weights(weights_iterator)  # type: ignore[arg-type]
            finalize_reload(model, self.model_runner.model_config)


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
        communicator_type, process_group_type = _require_vllm_nccl(
            "NCCL weight updates require vLLM NCCL internals. Install vLLM and "
            "run the inference server on CUDA workers."
        )

        device = getattr(self, "device", None)
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        local_rank = getattr(device, "index", None)
        if local_rank is None:
            local_rank = torch.cuda.current_device()
        rank = rank_offset + int(local_rank)
        world_size = nccl_world_size(inference_world_size)

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
        current_config = set_current_vllm_config
        initialize_reload = initialize_layerwise_reload
        finalize_reload = finalize_layerwise_reload
        if any(
            value is None
            for value in (current_config, initialize_reload, finalize_reload)
        ):
            from vllm.config import set_current_vllm_config as current_config
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload as finalize_reload,
            )
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload as initialize_reload,
            )

        communicator = getattr(self, "_wavelet_nccl_communicator", None)
        if communicator is None:
            raise RuntimeError(
                "NCCL weight update receiver was not initialized. Call "
                "/init_broadcaster before /load_policy."
            )

        update_info_path = Path(weight_path) / NCCL_UPDATE_INFO_FILENAME
        update_info = json.loads(update_info_path.read_text())
        model = _worker_model(self)
        if update_info.get("protocol") != "layerwise_v1":
            raise ValueError("Unsupported NCCL weight update protocol.")

        device = next(model.parameters()).device
        layer_count = _broadcast_integer(
            0,
            communicator,
            device=device,
            source=False,
        )
        with torch.device(device), current_config(self.vllm_config):
            initialize_reload(model)
            for _ in range(layer_count):
                metadata = json.loads(
                    _broadcast_bytes(
                        None,
                        communicator,
                        device=device,
                        source=False,
                    ).decode("utf-8")
                )
                loaded: list[NamedTensor] = []
                for dtype_name, entries in metadata.items():
                    dtype = getattr(torch, dtype_name)
                    total_numel = sum(int(entry["numel"]) for entry in entries)
                    concatenated = torch.empty(
                        total_numel,
                        dtype=dtype,
                        device=device,
                    )
                    communicator.broadcast(concatenated, src=0)
                    offset = 0
                    for entry in entries:
                        numel = int(entry["numel"])
                        tensor = concatenated[offset : offset + numel].view(
                            entry["shape"]
                        )
                        loaded.append((entry["name"], tensor))
                        offset += numel
                    if offset != total_numel:
                        raise ValueError(
                            f"NCCL metadata size mismatch for dtype {dtype_name!r}."
                        )
                model.load_weights(loaded)  # type: ignore[arg-type]
            finalize_reload(model, self.model_runner.model_config)

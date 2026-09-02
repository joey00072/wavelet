"""Filesystem and NCCL policy transport mechanics."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from torch import Tensor, nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from wavelet.orchestrator.policy_metadata import (
    adapter_artifact_metadata,
    policy_metadata,
)
from wavelet.trainer.distributed import barrier
from wavelet.transport.queue import (
    POLICY_META_FILENAME,
    STABLE_BATCH_MARKER,
    STEP_DIR_PREFIX,
    QueueEvent,
    append_event_best_effort,
    get_policy_step_dir,
    resolve_policy_dir,
    utc_now,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object

NCCL_READY_MARKER = "NCCL_READY"
NCCL_UPDATE_INFO_FILENAME = "update_info.json"
NamedTensor = tuple[str, Tensor]


def prune_policy_snapshots(policy_dir: Path, *, keep_last: int | None) -> list[Path]:
    """Remove old stable filesystem policy snapshots and return removed paths."""
    if keep_last is None or not policy_dir.exists():
        return []

    snapshots: list[tuple[int, Path]] = []
    for candidate in policy_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(STEP_DIR_PREFIX):
            continue
        try:
            step = int(candidate.name.removeprefix(STEP_DIR_PREFIX))
        except ValueError:
            continue
        if (candidate / STABLE_BATCH_MARKER).exists():
            snapshots.append((step, candidate))

    removed: list[Path] = []
    for _, path in sorted(snapshots)[:-keep_last]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


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


def _require_vllm_receiver_nccl() -> tuple[type[object], type[object]]:
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
        communicator_type, process_group_type = _require_vllm_receiver_nccl()

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
        stream = (
            torch.cuda.current_stream(communicator.device)
            if communicator.device.type == "cuda"
            else None
        )
        for name, dtype_name, shape in zip(
            update_info["names"],
            update_info["dtype_names"],
            update_info["shapes"],
            strict=True,
        ):
            dtype = getattr(torch, dtype_name)
            weight = torch.empty(shape, dtype=dtype, device=communicator.device)
            communicator.broadcast(weight, src=0, stream=stream)
            model.load_weights([(name, weight)])  # type: ignore[arg-type]
            del weight

        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)


class PolicyExportMixin:
    """Trainer-side filesystem and NCCL policy publication mechanics."""

    def _init_policy_transport(self) -> None:
        self._nccl_broadcaster_executor: ThreadPoolExecutor | None = None
        self._nccl_broadcaster_future: Future[NCCLWeightBroadcaster] | None = None

    def _close_policy_transport(self) -> None:
        if self._nccl_broadcaster_executor is not None:
            self._nccl_broadcaster_executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
            self._nccl_broadcaster_executor = None

    def should_export_policy(self, step: int) -> bool:
        if step == 0:
            return self.config.policy_transfer.export_initial
        return step % self.config.policy_transfer.export_every_steps == 0

    def export_policy(self, *, step: int | None = None) -> Path | None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        if self.world is None:
            raise RuntimeError("World not set up")

        export_step = self.step if step is None else step
        if not self.should_export_policy(export_step):
            return None
        if self.config.policy_transfer.type == "nccl":
            return self._export_nccl_policy(export_step)

        policy_dir = resolve_policy_dir(self.output_dir, self.config.policy_transfer)
        step_dir = get_policy_step_dir(policy_dir, export_step)
        tmp_dir = step_dir.with_name(f".{step_dir.name}.tmp")
        self._prepare_export_directory(tmp_dir, step_dir)
        saved_path = self._save_filesystem_policy(tmp_dir)
        if self.world.is_main:
            self._write_policy_metadata(
                tmp_dir,
                export_step=export_step,
                kind=saved_path.name,
            )
        self.offload_after_refit()
        self._publish_export_directory(
            tmp_dir,
            step_dir,
            export_step=export_step,
        )
        return step_dir

    def _save_filesystem_policy(self, tmp_dir: Path) -> Path:
        from wavelet.trainer.model import (
            export_model_for_save,
            save_lora_adapter_snapshot,
            save_lora_adapter_snapshot_from_fsdp,
            save_model,
        )

        if (
            self.config.lora is not None
            and self.config.policy_transfer.lightweight_lora
            and isinstance(self.model, FSDP)
        ):
            return save_lora_adapter_snapshot_from_fsdp(
                self.model,
                tmp_dir,
                is_main_process=self.world.is_main,
                parallel_dims=self.parallel_dims,
            )
        export_dtype = torch.bfloat16 if self.config.lora is None else None
        export_model, state_dict = export_model_for_save(
            self.model,
            state_dict_dtype=export_dtype,
        )
        if self.config.policy_transfer.lightweight_lora and isinstance(
            export_model, PeftModel
        ):
            return save_lora_adapter_snapshot(
                export_model,
                tmp_dir,
                state_dict=state_dict,
                is_main_process=self.world.is_main,
                parallel_dims=self.parallel_dims,
            )
        return save_model(
            export_model,
            self.tokenizer,
            tmp_dir,
            state_dict=state_dict,
            is_main_process=self.world.is_main,
        )

    def _prepare_export_directory(self, tmp_dir: Path, step_dir: Path) -> None:
        if not self.world.is_main:
            return
        for path in (tmp_dir, step_dir):
            if path.exists():
                shutil.rmtree(path)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def _write_policy_metadata(
        self,
        tmp_dir: Path,
        *,
        export_step: int,
        kind: str,
    ) -> None:
        artifact = adapter_artifact_metadata(tmp_dir / "adapter")
        metadata = policy_metadata(
            config=self.config,
            format_version=1,
            step=export_step,
            kind=kind,
            created_at=datetime.now(timezone.utc).isoformat(),
            extra={"artifact": artifact} if artifact is not None else None,
        )
        (tmp_dir / POLICY_META_FILENAME).write_text(json.dumps(metadata))

    def _publish_export_directory(
        self,
        tmp_dir: Path,
        step_dir: Path,
        *,
        export_step: int,
    ) -> None:
        if self.world.world_size > 1:
            barrier(self.world)
        if self.world.is_main:
            (tmp_dir / STABLE_BATCH_MARKER).touch()
            tmp_dir.replace(step_dir)
            self._record_policy_export(export_step)
        if self.world.world_size > 1:
            barrier(self.world)
        if self.world.is_main:
            prune_policy_snapshots(
                step_dir.parent,
                keep_last=self.config.policy_transfer.keep_last,
            )

    def _export_nccl_policy(self, export_step: int) -> Path:
        if self.model is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        if self.world is None:
            raise RuntimeError("World not set up")
        if self.config.lora is not None:
            raise NotImplementedError(
                "NCCL policy transfer is only implemented for full-model updates. "
                "Use filesystem policy transfer for LoRA adapters."
            )

        policy_dir = resolve_policy_dir(self.output_dir, self.config.policy_transfer)
        step_dir = get_policy_step_dir(policy_dir, export_step)
        tmp_dir = step_dir.with_name(f".{step_dir.name}.tmp")
        self._prepare_export_directory(tmp_dir, step_dir)
        state_dict = self._nccl_export_state_dict(export_step)
        if self.world.is_main:
            self._write_nccl_export(
                tmp_dir,
                step_dir,
                export_step=export_step,
                state_dict=state_dict,
            )

        self.offload_after_refit()
        self._broadcast_nccl_export(
            step_dir,
            export_step=export_step,
            state_dict=state_dict,
        )
        if self.world.world_size > 1:
            barrier(self.world)
        if self.world.is_main:
            self._record_policy_export(export_step)
        if self.world.world_size > 1:
            barrier(self.world)
        return step_dir

    def _nccl_export_state_dict(
        self,
        export_step: int,
    ) -> dict[str, Tensor] | None:
        if export_step == 0:
            return None
        from wavelet.trainer.model import export_model_for_save

        _, state_dict = export_model_for_save(
            self.model,
            state_dict_dtype=torch.bfloat16,
        )
        return state_dict

    def _write_nccl_export(
        self,
        tmp_dir: Path,
        step_dir: Path,
        *,
        export_step: int,
        state_dict: dict[str, Tensor] | None,
    ) -> None:
        named_tensors = [] if state_dict is None else list(state_dict.items())
        update_info = update_info_for_named_tensors(named_tensors)
        (tmp_dir / NCCL_UPDATE_INFO_FILENAME).write_text(json.dumps(update_info))
        self._write_policy_metadata(
            tmp_dir,
            export_step=export_step,
            kind="nccl",
        )
        (tmp_dir / STABLE_BATCH_MARKER).touch()
        tmp_dir.replace(step_dir)
        if export_step > 0:
            self._start_nccl_broadcaster()

    def _broadcast_nccl_export(
        self,
        step_dir: Path,
        *,
        export_step: int,
        state_dict: dict[str, Tensor] | None,
    ) -> None:
        if export_step == 0 or not self.world.is_main:
            return
        self._wait_for_nccl_ready(step_dir)
        if state_dict is None:
            raise RuntimeError("Missing state dict for NCCL policy broadcast.")
        self._nccl_broadcaster().broadcast_named_tensors(state_dict.items())

    def _record_policy_export(self, export_step: int) -> None:
        append_event_best_effort(
            self.config.output_dir / "events",
            QueueEvent(
                time=utc_now(),
                kind="policy_export_completed",
                policy_step=export_step,
            ),
        )

    def _wait_for_nccl_ready(self, step_dir: Path) -> None:
        ready_path = step_dir / NCCL_READY_MARKER
        deadline = monotonic() + self.config.policy_transfer.nccl_timeout_seconds
        while monotonic() < deadline:
            if ready_path.exists():
                return
            sleep(0.1)
        raise TimeoutError(
            "Timed out waiting for inference workers to enter NCCL policy update "
            f"for {step_dir}."
        )

    def _start_nccl_broadcaster(self) -> None:
        if self._nccl_broadcaster_future is not None:
            return
        if self._nccl_broadcaster_executor is None:
            self._nccl_broadcaster_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="wavelet-nccl-broadcaster",
            )
        self._nccl_broadcaster_future = self._nccl_broadcaster_executor.submit(
            self._create_nccl_broadcaster
        )

    def _nccl_broadcaster(self) -> NCCLWeightBroadcaster:
        if self._nccl_broadcaster_future is None:
            self._start_nccl_broadcaster()
        assert self._nccl_broadcaster_future is not None
        return self._nccl_broadcaster_future.result(
            timeout=self.config.policy_transfer.nccl_timeout_seconds
        )

    def _create_nccl_broadcaster(self) -> NCCLWeightBroadcaster:
        device = (
            self.world.device
            if self.world is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        return NCCLWeightBroadcaster(
            host=self.config.policy_transfer.nccl_host,
            port=self.config.policy_transfer.nccl_port,
            rank=0,
            world_size=(
                self.config.policy_transfer.nccl_inference_world_size
                + self.config.policy_transfer.nccl_rank_offset
            ),
            device=device,
            timeout_seconds=self.config.policy_transfer.nccl_timeout_seconds,
        )

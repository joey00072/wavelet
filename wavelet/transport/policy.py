"""Filesystem and NCCL policy transport mechanics."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from torch import Tensor, nn
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

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


def prune_policy_snapshots_beyond(policy_dir: Path, *, step: int) -> list[Path]:
    """Remove policy directories from an abandoned run beyond a resume step."""
    if not policy_dir.exists():
        return []
    removed: list[Path] = []
    for candidate in policy_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(STEP_DIR_PREFIX):
            continue
        try:
            candidate_step = int(candidate.name.removeprefix(STEP_DIR_PREFIX))
        except ValueError:
            continue
        if candidate_step > step:
            shutil.rmtree(candidate)
            removed.append(candidate)
    return sorted(removed)


def _is_reusable_policy_snapshot(
    step_dir: Path,
    *,
    step: int,
    expected_kind: str,
) -> bool:
    metadata_path = step_dir / POLICY_META_FILENAME
    if not (step_dir / STABLE_BATCH_MARKER).is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        return False
    metadata_step = metadata.get("step")
    if (
        not isinstance(metadata_step, int)
        or isinstance(metadata_step, bool)
        or metadata_step != step
    ):
        return False
    kind = metadata.get("kind")
    if kind != expected_kind:
        return False
    if kind == "adapter":
        return (step_dir / "adapter" / "adapter_model.safetensors").is_file()
    if kind == "model":
        return (step_dir / "model").is_dir()
    return False


def _require_vllm_nccl() -> tuple[type[Any], type[Any]]:
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
    except ImportError as exc:
        raise ImportError(
            "NCCL weight broadcast requires vLLM NCCL internals. Install vLLM and "
            "run this on CUDA workers."
        ) from exc
    return (
        PyNcclCommunicator,
        StatelessProcessGroup,
    )


_LAYER_KEY_RE = re.compile(
    r"^(?P<prefix>(?:.+\.)?(?:layers|blocks|h))\."
    r"(?P<index>\d+)(?:\.|$)"
)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _layer_groups(
    named_tensors: Iterable[NamedTensor],
) -> dict[str, list[tuple[str, Tensor]]]:
    groups: dict[str, list[tuple[str, Tensor]]] = {}
    for name, tensor in named_tensors:
        groups.setdefault(_dtype_name(tensor.dtype), []).append((name, tensor))
    return groups


def _layer_metadata(
    named_tensors: Iterable[NamedTensor],
) -> dict[str, list[dict[str, Any]]]:
    return {
        dtype_name: [
            {"name": name, "shape": list(tensor.shape), "numel": tensor.numel()}
            for name, tensor in tensors
        ]
        for dtype_name, tensors in _layer_groups(named_tensors).items()
    }


def _partition_state_dict(
    state_dict: dict[str, Tensor],
) -> list[dict[str, Tensor]]:
    """Partition checkpoint tensors into non-layer and transformer-layer groups."""
    layer_matches = [(name, _LAYER_KEY_RE.match(name)) for name in state_dict]
    if not any(match is not None for _, match in layer_matches):
        return [state_dict]
    layer_indices = {
        int(match.group("index")) for _, match in layer_matches if match is not None
    }
    prefix = next(match.group("prefix") for _, match in layer_matches if match)
    groups: list[dict[str, Tensor]] = [
        {
            name: tensor
            for name, tensor in state_dict.items()
            if not name.startswith(f"{prefix}.")
        }
    ]
    for index in range(max(layer_indices) + 1):
        groups.append(
            {
                name: tensor
                for name, tensor in state_dict.items()
                if name.startswith(f"{prefix}.{index}.")
            }
        )
    return groups


def _model_named_tensors(model: nn.Module) -> dict[str, Tensor]:
    """Return parameter and buffer references without materializing full state."""
    from wavelet.trainer.model import unwrap_model

    unwrapped = unwrap_model(model)
    tensors = dict(unwrapped.named_parameters())
    tensors.update(dict(unwrapped.named_buffers()))
    return tensors


def _materialize_wire_tensors(
    model: nn.Module,
    state_dict: dict[str, Tensor],
    layer_index: int,
) -> dict[str, Tensor]:
    """Gather only this layer's sharded tensors and retain their wire dtype."""
    from wavelet.trainer.model import unwrap_model

    conversion_model = unwrap_model(model)
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
                name: tensor.detach().clone().contiguous()
                for name, tensor in state_dict.items()
            }

    materialized: dict[str, Tensor] = {}
    for name, tensor in state_dict.items():
        full_tensor = getattr(tensor, "full_tensor", None)
        if callable(full_tensor):
            tensor = full_tensor()
        materialized[name] = tensor.detach().contiguous()
    return materialized


def _convert_layer_to_hf(
    model: nn.Module,
    state_dict: dict[str, Tensor],
    layer_index: int,
) -> dict[str, Tensor]:
    """Convert one trainer layer to the checkpoint names vLLM consumes."""
    convert_layer = getattr(model, "convert_layer_to_hf", None)
    if callable(convert_layer):
        converted = convert_layer(state_dict, layer_index)
        return state_dict if converted is None else converted
    try:
        from transformers.core_model_loading import revert_weight_conversion
    except ImportError:
        return state_dict
    return revert_weight_conversion(model, state_dict)


def _iter_layer_state_dicts(model: nn.Module) -> Iterator[dict[str, Tensor]]:
    from wavelet.trainer.model import unwrap_model

    conversion_model = unwrap_model(model)
    tensors = _model_named_tensors(model)
    for layer_index, layer in enumerate(_partition_state_dict(tensors)):
        wire_layer = _materialize_wire_tensors(model, layer, layer_index - 1)
        yield _convert_layer_to_hf(conversion_model, wire_layer, layer_index - 1)


def _broadcast_integer(
    value: int,
    communicator: Any,
    *,
    device: torch.device,
    source: bool,
) -> int:
    integer = torch.tensor(
        [value if source else 0],
        dtype=torch.int64,
        device=device,
    )
    communicator.broadcast(integer, src=0)
    return int(integer.item())


def _broadcast_bytes(
    payload: bytes | None,
    communicator: Any,
    *,
    device: torch.device,
    source: bool,
) -> bytes:
    size = _broadcast_integer(
        0 if payload is None else len(payload),
        communicator,
        device=device,
        source=source,
    )
    buffer = (
        torch.tensor(list(payload), dtype=torch.uint8, device=device)
        if source
        else torch.empty(size, dtype=torch.uint8, device=device)
    )
    communicator.broadcast(buffer, src=0)
    return bytes(buffer.cpu().tolist())


def nccl_world_size(inference_world_size: int) -> int:
    """Return the NCCL group size: every inference rank plus the one trainer rank.

    Replicas receive different rank offsets, so the group size must be derived
    from the total inference rank count rather than from a replica's offset.
    """
    return inference_world_size + 1


@dataclass(slots=True)
class NCCLWeightBroadcaster:
    host: str
    port: int
    rank: int
    world_size: int
    device: torch.device | str | int = "cuda"
    timeout_seconds: int = 600
    source_rank: int = 0
    _communicator: Any = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _process_group: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL weight broadcast requires CUDA.")
        communicator_type, process_group_type = _require_vllm_nccl()
        self._device = torch.device(self.device)
        self._process_group = process_group_type.create(
            host=self.host,
            port=self.port,
            rank=self.rank,
            world_size=self.world_size,
            store_timeout=self.timeout_seconds,
        )
        self._communicator = communicator_type(self._process_group, device=self._device)

    @torch.no_grad()
    def broadcast_layers(
        self,
        layers: Iterable[dict[str, Tensor]],
        *,
        layer_count: int,
    ) -> None:
        if self.rank != self.source_rank:
            raise RuntimeError("Only the source rank can broadcast model weights.")
        _broadcast_integer(
            layer_count,
            self._communicator,
            device=self._device,
            source=True,
        )
        for layer in layers:
            payload = json.dumps(_layer_metadata(layer.items())).encode("utf-8")
            _broadcast_bytes(
                payload,
                self._communicator,
                device=self._device,
                source=True,
            )
            for dtype_name, tensors in _layer_groups(layer.items()).items():
                del dtype_name
                flattened = [
                    tensor.detach().to(self._device).contiguous().view(-1)
                    for _, tensor in tensors
                ]
                concatenated = torch.cat(flattened)
                self._communicator.broadcast(concatenated, src=self.source_rank)
                del concatenated

    @torch.no_grad()
    def broadcast_model(self, model: nn.Module) -> None:
        self.broadcast_layers(
            _iter_layer_state_dicts(model),
            layer_count=len(_partition_state_dict(_model_named_tensors(model))),
        )


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
        device = next(model.parameters()).device
        with torch.device(device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            model.load_weights(weights_iterator)  # type: ignore[arg-type]
            finalize_layerwise_reload(model, self.model_runner.model_config)


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
        with torch.device(device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
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
            finalize_layerwise_reload(model, self.model_runner.model_config)


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

    def export_policy(
        self,
        *,
        step: int | None = None,
        force: bool = False,
    ) -> Path | None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        if self.world is None:
            raise RuntimeError("World not set up")

        export_step = self.step if step is None else step
        if not force and not self.should_export_policy(export_step):
            return None
        policy_dir = resolve_policy_dir(self.output_dir, self.config.policy_transfer)
        if force:
            # A resumed run must not leave newer snapshots from the crashed run
            # visible; inference would otherwise load a policy the trainer never
            # produced (and, for NCCL, wait on a handshake nobody completes).
            if self.world.is_main:
                prune_policy_snapshots_beyond(policy_dir, step=export_step)
            if self.world.world_size > 1:
                barrier(self.world)
        if self.config.policy_transfer.type == "nccl":
            return self._export_nccl_policy(export_step)

        step_dir = get_policy_step_dir(policy_dir, export_step)
        if force:
            expected_kind = (
                "adapter"
                if self.config.lora is not None
                and self.config.policy_transfer.lightweight_lora
                else "model"
            )
            if _is_reusable_policy_snapshot(
                step_dir,
                step=export_step,
                expected_kind=expected_kind,
            ):
                self.offload_after_refit()
                return step_dir
        elif (step_dir / STABLE_BATCH_MARKER).is_file():
            raise FileExistsError(
                f"Stable policy step {export_step} already exists at '{step_dir}'."
            )
        tmp_dir = step_dir.with_name(f".{step_dir.name}.tmp")
        self._prepare_export_directory(tmp_dir, step_dir)
        if self.world.world_size > 1:
            barrier(self.world)
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
            is_fsdp_model,
            save_lora_adapter_snapshot,
            save_lora_adapter_snapshot_from_fsdp,
            save_model,
        )

        if (
            self.config.lora is not None
            and self.config.policy_transfer.lightweight_lora
            and is_fsdp_model(self.model)
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
            created_at=datetime.now(UTC).isoformat(),
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
        if self.world.is_main:
            self._write_nccl_export(
                tmp_dir,
                step_dir,
                export_step=export_step,
            )

        self._broadcast_nccl_export(
            step_dir,
            export_step=export_step,
        )
        self.offload_after_refit()
        if self.world.world_size > 1:
            barrier(self.world)
        if self.world.is_main:
            self._record_policy_export(export_step)
        if self.world.world_size > 1:
            barrier(self.world)
        return step_dir

    def _nccl_export_layers(self) -> Iterator[dict[str, Tensor]]:
        if self.model is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        yield from _iter_layer_state_dicts(self.model)

    def _nccl_export_layer_count(self) -> int:
        if self.model is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        return len(_partition_state_dict(_model_named_tensors(self.model)))

    def _write_nccl_export(
        self,
        tmp_dir: Path,
        step_dir: Path,
        *,
        export_step: int,
    ) -> None:
        update_info = {"protocol": "layerwise_v1"}
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
    ) -> None:
        if export_step == 0:
            return
        layer_count = self._nccl_export_layer_count()
        layers = self._nccl_export_layers()
        if self.world.is_main:
            self._wait_for_nccl_ready(step_dir)
            self._nccl_broadcaster().broadcast_layers(
                layers,
                layer_count=layer_count,
            )
        else:
            for _ in layers:
                pass

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
            world_size=nccl_world_size(
                self.config.policy_transfer.nccl_inference_world_size
            ),
            device=device,
            timeout_seconds=self.config.policy_transfer.nccl_timeout_seconds,
        )

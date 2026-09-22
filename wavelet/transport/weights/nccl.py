"""Filesystem policy snapshot bookkeeping and the NCCL weight broadcaster.

This module owns the NCCL broadcast mechanics and the plain
`dict[str, Tensor]` wire format; it holds no model-side (`nn.Module`)
conversion logic. See `wavelet.trainer.export_tensors` for gathering tensors
from a model, and `wavelet.transport.weights.wire` for wire-format shaping.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from wavelet.transport.rollouts.filesystem import (
    POLICY_META_FILENAME,
    STABLE_BATCH_MARKER,
    parse_step,
)

NamedTensor = tuple[str, Tensor]


def prune_policy_snapshots(policy_dir: Path, *, keep_last: int | None) -> list[Path]:
    """Remove old stable filesystem policy snapshots and return removed paths."""
    if keep_last is None or not policy_dir.exists():
        return []

    snapshots: list[tuple[int, Path]] = []
    for candidate in policy_dir.iterdir():
        step = parse_step(candidate)
        if step is None:
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
        candidate_step = parse_step(candidate)
        if candidate_step is None:
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


def _require_vllm_nccl(message: str) -> tuple[type[Any], type[Any]]:
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
    except ImportError as exc:
        raise ImportError(message) from exc
    return PyNcclCommunicator, StatelessProcessGroup


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


def _indexed_cuda_device(device: torch.device | str | int) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


@dataclass(slots=True)
class NCCLWeightBroadcaster:
    host: str
    port: int
    rank: int
    world_size: int
    device: torch.device | str | int = "cuda"
    timeout_seconds: int = 600
    source_rank: int = 0
    dtype: torch.dtype | None = None
    _communicator: Any = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _process_group: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL weight broadcast requires CUDA.")
        communicator_type, process_group_type = _require_vllm_nccl(
            "NCCL weight broadcast requires vLLM NCCL internals. Install vLLM and "
            "run this on CUDA workers."
        )
        self._device = _indexed_cuda_device(self.device)
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
            for tensors in _layer_groups(layer.items()).values():
                flattened = [
                    tensor.detach().to(self._device).contiguous().view(-1)
                    for _, tensor in tensors
                ]
                concatenated = torch.cat(flattened)
                self._communicator.broadcast(concatenated, src=self.source_rank)
                del concatenated

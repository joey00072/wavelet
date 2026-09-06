"""Distributed world state and parallel device-mesh dimensions."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

import torch
import torch.distributed as dist
from torch._utils import _get_available_device_type
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

DEFAULT_DEVICE_TYPE = _get_available_device_type() or "cuda"


def _mesh_device_type() -> str:
    if not dist.is_available() or not dist.is_initialized():
        return DEFAULT_DEVICE_TYPE
    return "cuda" if distributed_uses_cuda() else "cpu"


@dataclass
class ParallelDims:
    dp_replicate: int = 1
    dp_shard: int = -1
    cp: int = 1
    tp: int = 1
    ep: int = 1
    pp: int = 1
    world_size: int = 1

    _world_mesh: DeviceMesh | None = field(default=None, init=False, repr=False)
    _submeshes: dict[str, DeviceMesh] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name, value in (
            ("dp_replicate", self.dp_replicate),
            ("cp", self.cp),
            ("tp", self.tp),
            ("ep", self.ep),
            ("pp", self.pp),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")

        if self.dp_shard == -1:
            denom = self.dp_replicate * self.cp * self.tp * self.pp
            if self.world_size % denom != 0:
                raise ValueError(
                    "world_size must be divisible by dp_replicate * cp * tp * pp "
                    "when dp_shard is inferred"
                )
            self.dp_shard = self.world_size // denom
        elif self.dp_shard < 1:
            raise ValueError("dp_shard must be -1 or >= 1")

        total = self.dp_replicate * self.dp_shard * self.cp * self.tp * self.pp
        if total != self.world_size:
            raise ValueError(
                "Invalid parallel dims: "
                f"dp_replicate({self.dp_replicate}) * dp_shard({self.dp_shard}) * "
                f"cp({self.cp}) * tp({self.tp}) * pp({self.pp}) != world_size({self.world_size})"
            )

        if self.ep > 1:
            if self.tp > 1:
                raise ValueError(
                    "tp>1 with ep>1 is not implemented in this repository."
                )
            if self.ep % self.cp != 0 or (self.dp_shard * self.cp) % self.ep != 0:
                raise ValueError("ep must divide dp_shard * cp and be divisible by cp.")

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self._build_mesh()
        return self._world_mesh

    def get_mesh(self, name: str) -> DeviceMesh:
        mesh = self.world_mesh
        if name in self._submeshes:
            return self._submeshes[name]
        return mesh[name]

    def _build_mesh(self) -> DeviceMesh:
        if self.ep_enabled:
            return self._build_mesh_with_ep()
        return self._build_mesh_without_ep()

    def _build_mesh_with_ep(self) -> DeviceMesh:
        device_type = _mesh_device_type()
        dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
        dp_shard_in_ep = self.ep // self.cp

        dims: list[int] = []
        names: list[str] = []
        for value, name in (
            (self.pp, "pp"),
            (self.dp_replicate, "dp_replicate"),
            (dp_shard_mod_ep, "dp_shard_mod_ep"),
            (dp_shard_in_ep, "dp_shard_in_ep"),
            (self.cp, "cp"),
        ):
            if value > 1 or name == "dp_shard_mod_ep":
                dims.append(value)
                names.append(name)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=tuple(names))

        dp_mesh_dim_names: list[str] = []
        dp_shard_cp_dim_names: list[str] = ["dp_shard_mod_ep"]
        ep_mesh_dim_names: list[str] = []
        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
        dp_mesh_dim_names.append("dp_shard_mod_ep")
        if "dp_shard_in_ep" in names:
            dp_mesh_dim_names.append("dp_shard_in_ep")
            dp_shard_cp_dim_names.append("dp_shard_in_ep")
            ep_mesh_dim_names.append("dp_shard_in_ep")
        if self.cp_enabled:
            dp_shard_cp_dim_names.append("cp")
            ep_mesh_dim_names.append("cp")

        if self.cp_enabled:
            dp_cp_mesh_dim_names = [
                name
                for name in (
                    "dp_replicate",
                    "dp_shard_mod_ep",
                    "dp_shard_in_ep",
                    "cp",
                )
                if name in names
            ]
            self._submeshes["dp_cp"] = mesh[tuple(dp_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_cp"
            )

        self._submeshes["dp"] = mesh[tuple(dp_mesh_dim_names)]._flatten(
            mesh_dim_name="dp"
        )
        self._submeshes["dp_shard_cp"] = mesh[tuple(dp_shard_cp_dim_names)]._flatten(
            mesh_dim_name="dp_shard_cp"
        )
        dp_mod_ep_dim_names = [
            name for name in ("dp_replicate", "dp_shard_mod_ep") if name in names
        ]
        self._submeshes["dp_mod_ep"] = mesh[tuple(dp_mod_ep_dim_names)]._flatten(
            mesh_dim_name="dp_mod_ep"
        )
        if ep_mesh_dim_names:
            self._submeshes["ep"] = mesh[tuple(ep_mesh_dim_names)]._flatten(
                mesh_dim_name="ep"
            )

        self._register_hsdp_submesh(mesh, device_type, dp_shard_cp_dim_names)
        return mesh

    def _register_hsdp_submesh(
        self,
        mesh: DeviceMesh,
        device_type: str,
        dp_shard_cp_dim_names: list[str],
    ) -> None:
        if not self.dp_replicate_enabled:
            self._submeshes["hsdp"] = self._submeshes["dp_shard_cp"]
            return
        parent = mesh[tuple(["dp_replicate"] + dp_shard_cp_dim_names)]
        hsdp_tensor = parent.mesh.reshape(self.dp_replicate, -1)
        self._submeshes["hsdp"] = DeviceMesh(
            device_type,
            hsdp_tensor,
            mesh_dim_names=("dp_replicate", "dp_shard_cp"),
        )

    def _build_mesh_without_ep(self) -> DeviceMesh:
        device_type = _mesh_device_type()
        dims: list[int] = []
        names: list[str] = []
        for value, name in (
            (self.pp, "pp"),
            (self.dp_replicate, "dp_replicate"),
            (self.dp_shard, "dp_shard"),
            (self.cp, "cp"),
            (self.tp, "tp"),
        ):
            if value > 1 or name == "dp_shard":
                dims.append(value)
                names.append(name)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=tuple(names))

        dp_mesh_dim_names: list[str] = []
        dp_shard_cp_dim_names: list[str] = []
        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
        dp_mesh_dim_names.append("dp_shard")
        dp_shard_cp_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_dim_names.append("cp")
        if dp_mesh_dim_names:
            self._submeshes["dp"] = mesh[tuple(dp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp"
            )
        if dp_shard_cp_dim_names:
            self._submeshes["dp_shard_cp"] = mesh[
                tuple(dp_shard_cp_dim_names)
            ]._flatten(mesh_dim_name="dp_shard_cp")
        if self.cp_enabled:
            dp_cp_mesh_dim_names = [*dp_mesh_dim_names, "cp"]
            self._submeshes["dp_cp"] = mesh[tuple(dp_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_cp"
            )
        self._register_hsdp_submesh(mesh, device_type, dp_shard_cp_dim_names)
        if self.tp_enabled:
            self._submeshes["tp"] = mesh["tp"]
        return mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1

    @property
    def fsdp_enabled(self) -> bool:
        return self.dp_shard_enabled or self.cp_enabled

    @cached_property
    def seq_len_divisor(self) -> int:
        return self.cp * 2 if self.cp > 1 else 1


@dataclass
class World:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


_world: World | None = None


def distributed_uses_cuda() -> bool:
    if not dist.is_initialized():
        return torch.cuda.is_available()
    # Hybrid backends report "cpu:gloo,cuda:nccl"; any NCCL/CUDA component
    # means the model and mesh live on CUDA.
    backend = str(dist.get_backend()).lower()
    return "nccl" in backend or "cuda" in backend


def world_is_distributed(world: World | None) -> bool:
    return world is not None and world.world_size > 1


@torch.no_grad()
def clip_grad_norm_across_meshes_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
) -> torch.Tensor:
    """Clip one model whose gradients may live on different DTensor meshes."""
    grouped: dict[int | None, list[torch.Tensor]] = {}
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        mesh_key = id(grad.device_mesh) if isinstance(grad, DTensor) else None
        grouped.setdefault(mesh_key, []).append(parameter)

    if not grouped:
        return torch.tensor(0.0)
    if len(grouped) == 1:
        return torch.nn.utils.clip_grad_norm_(
            next(iter(grouped.values())),
            max_norm,
        )

    group_norms: list[torch.Tensor] = []
    for group in grouped.values():
        norm = torch.nn.utils.get_total_norm(
            [parameter.grad for parameter in group],
            norm_type=2.0,
        )
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        group_norms.append(norm)

    target_device = group_norms[0].device
    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(target_device) for norm in group_norms]),
        ord=2.0,
    )
    for group in grouped.values():
        torch.nn.utils.clip_grads_with_norm_(group, max_norm, total_norm)
    return total_norm


def get_world() -> World:
    global _world
    if _world is None:
        rank = (
            int(torch.distributed.get_rank())
            if torch.distributed.is_initialized()
            else 0
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = (
            int(torch.distributed.get_world_size())
            if torch.distributed.is_initialized()
            else 1
        )
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 1
        )
        if torch.cuda.is_available() and distributed_uses_cuda():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        _world = World(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            device=device,
        )
    return _world


def set_world(world: World) -> None:
    global _world
    _world = world


def collective_device(world: World) -> torch.device:
    """Device for small flag tensors in collectives (CUDA only under NCCL)."""
    if distributed_uses_cuda() and world.device.type == "cuda":
        return world.device
    return torch.device("cpu")


def barrier(world: World | None = None) -> None:
    if not dist.is_initialized():
        return
    if world is None:
        world = get_world()
    if collective_device(world).type == "cuda":
        dist.barrier(device_ids=[world.local_rank])
    else:
        dist.barrier()


def _all_reduce_flag(flag: bool, op: dist.ReduceOp, device: torch.device) -> bool:
    if not dist.is_initialized():
        return flag
    tensor = torch.tensor(int(flag), dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=op)
    return bool(tensor.item())


def all_ranks_true(flag: bool, *, device: torch.device) -> bool:
    """Return True only if every rank passed True (no-op without a process group)."""
    return _all_reduce_flag(flag, dist.ReduceOp.MIN, device)


def any_rank_true(flag: bool, *, device: torch.device) -> bool:
    """Return True if any rank passed True (no-op without a process group)."""
    return _all_reduce_flag(flag, dist.ReduceOp.MAX, device)

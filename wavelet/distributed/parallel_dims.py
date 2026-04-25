from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import torch.distributed as dist
from torch._utils import _get_available_device_type
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


DEFAULT_DEVICE_TYPE = _get_available_device_type() or "cuda"


def _mesh_device_type() -> str:
    if not dist.is_available() or not dist.is_initialized():
        return DEFAULT_DEVICE_TYPE
    backend = dist.get_backend()
    if backend == "nccl":
        return "cuda"
    return "cpu"


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

        self._submeshes["dp"] = mesh[tuple(dp_mesh_dim_names)]._flatten(
            mesh_dim_name="dp"
        )
        self._submeshes["dp_shard_cp"] = mesh[tuple(dp_shard_cp_dim_names)]._flatten(
            mesh_dim_name="dp_shard_cp"
        )
        if ep_mesh_dim_names:
            self._submeshes["ep"] = mesh[tuple(ep_mesh_dim_names)]._flatten(
                mesh_dim_name="ep"
            )

        if self.dp_replicate_enabled:
            parent = mesh[tuple(["dp_replicate"] + dp_shard_cp_dim_names)]
            hsdp_tensor = parent.mesh.reshape(self.dp_replicate, -1)
            self._submeshes["hsdp"] = DeviceMesh(
                device_type,
                hsdp_tensor,
                mesh_dim_names=("dp_replicate", "dp_shard_cp"),
            )
        else:
            self._submeshes["hsdp"] = self._submeshes["dp_shard_cp"]
        return mesh

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
        if self.dp_replicate_enabled:
            parent = mesh[tuple(["dp_replicate"] + dp_shard_cp_dim_names)]
            hsdp_tensor = parent.mesh.reshape(self.dp_replicate, -1)
            self._submeshes["hsdp"] = DeviceMesh(
                device_type,
                hsdp_tensor,
                mesh_dim_names=("dp_replicate", "dp_shard_cp"),
            )
        else:
            self._submeshes["hsdp"] = self._submeshes["dp_shard_cp"]
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

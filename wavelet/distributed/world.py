from __future__ import annotations

import os
from dataclasses import dataclass

import torch


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

    @property
    def is_local_main(self) -> bool:
        return self.local_rank == 0


_world: World | None = None


def distributed_uses_cuda() -> bool:
    if not torch.distributed.is_initialized():
        return torch.cuda.is_available()
    backend = str(torch.distributed.get_backend()).lower()
    return "nccl" in backend or "cuda" in backend


def get_world() -> World:
    global _world
    if _world is None:
        rank = (
            int(torch.distributed.get_rank())
            if torch.distributed.is_initialized()
            else 0
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = (
            int(torch.distributed.get_world_size())
            if torch.distributed.is_initialized()
            else 1
        )
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 0)) or (
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


def barrier(world: World | None = None) -> None:
    if not torch.distributed.is_initialized():
        return
    if world is None:
        world = get_world()
    if distributed_uses_cuda() and world.device.type == "cuda":
        torch.distributed.barrier(device_ids=[world.local_rank])
    else:
        torch.distributed.barrier()

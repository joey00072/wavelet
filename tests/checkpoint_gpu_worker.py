"""Run with torchrun to check async checkpointing with NCCL training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from wavelet.configs.config import CheckpointConfig, FSDPConfig, ModelConfig
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.distributed import ParallelDims, World
from wavelet.trainer.model import maybe_wrap_fsdp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(48)
        device = torch.device("cuda", local_rank)
        world = World(
            rank=dist.get_rank(),
            local_rank=local_rank,
            world_size=dist.get_world_size(),
            local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
            device=device,
        )
        model = torch.nn.Linear(32, 32)
        model.config = SimpleNamespace(model_type="checkpoint-test")
        model = maybe_wrap_fsdp(
            model,
            model_config=ModelConfig(torch_dtype="float32"),
            fsdp_config=FSDPConfig(enabled=True, impl="fsdp1"),
            world=world,
            parallel_dims=ParallelDims(world_size=world.world_size),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        manager = CheckpointManager(
            model,
            optimizer,
            None,
            CheckpointConfig(mode="async", interval=1),
            args.output,
            world,
        )
        data = torch.randn(4, 32, device=device)
        model(data).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        expected = [p.detach().clone() for p in model.parameters()]
        expected_optim = {
            p: state["exp_avg"].clone()
            for p, state in optimizer.state.items()
            if "exp_avg" in state
        }
        assert manager.save(TrainerState(step=1, micro_step=1))
        # Exercise training collectives while the CPU checkpoint is outstanding.
        model(data).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        manager.wait_for_pending_save()
        restored = manager.load(args.output / "checkpoint-1")
        assert restored.step == 1
        for actual, saved in zip(model.parameters(), expected, strict=True):
            torch.testing.assert_close(actual, saved)
        for parameter, saved in expected_optim.items():
            torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], saved)
        assert manager.save(TrainerState(step=2, micro_step=2))
        manager.wait_for_pending_save()
        assert (args.output / "checkpoint-2" / "STABLE").exists()
        if world.is_main:
            print(
                "PASS: NCCL training, async Gloo save, concurrent update, and restore"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

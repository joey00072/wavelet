from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from safetensors.torch import load_file as load_safetensors
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from wavelet.configs.sft import FSDPConfig, LoRAConfig, ModelConfig
from wavelet.trainer.ckpt import AppState
from wavelet.trainer.debug import DEBUG_MODEL_NAME, build_debug_model
from wavelet.trainer.distributed import ParallelDims, World
from wavelet.trainer.model import (
    apply_lora,
    is_fsdp_model,
    maybe_wrap_fsdp,
    save_lora_adapter_snapshot_from_fsdp,
)


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        return value.to_local()
    return value


def _fsdp2_checkpoint_worker(
    rank: int,
    init_file: str,
    checkpoint_dir: str,
    export_dir: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(7)
        model = build_debug_model(max_seq_length=16)
        model = apply_lora(model, LoRAConfig(rank=2, alpha=4))
        parallel_dims = ParallelDims(world_size=2)
        world = World(
            rank=rank,
            local_rank=rank,
            world_size=2,
            local_world_size=2,
            device=torch.device("cpu"),
        )
        model = maybe_wrap_fsdp(
            model,
            model_config=ModelConfig(
                name=DEBUG_MODEL_NAME,
                torch_dtype="float32",
            ),
            fsdp_config=FSDPConfig(enabled=True, impl="fsdp2"),
            world=world,
            parallel_dims=parallel_dims,
        )

        assert is_fsdp_model(model)
        assert isinstance(model, FSDPModule)
        assert all(isinstance(block, FSDPModule) for block in model.transformer.h)

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dcp.save(
            {"app": AppState(model, optimizer, None)},
            checkpoint_id=checkpoint_dir,
        )
        expected_model = {
            name: _local_tensor(parameter).detach().clone()
            for name, parameter in model.named_parameters()
        }
        expected_optimizer = {
            name: _local_tensor(optimizer.state[parameter]["exp_avg"]).detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        with torch.no_grad():
            for parameter in model.parameters():
                _local_tensor(parameter).fill_(99)
                if parameter.requires_grad:
                    _local_tensor(optimizer.state[parameter]["exp_avg"]).fill_(99)

        dcp.load(
            {"app": AppState(model, optimizer, None)},
            checkpoint_id=checkpoint_dir,
        )

        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                _local_tensor(parameter),
                expected_model[name],
            )
            if parameter.requires_grad:
                torch.testing.assert_close(
                    _local_tensor(optimizer.state[parameter]["exp_avg"]),
                    expected_optimizer[name],
                )

        save_lora_adapter_snapshot_from_fsdp(
            model,
            Path(export_dir),
            is_main_process=rank == 0,
            parallel_dims=parallel_dims,
        )
        dist.barrier()
        if rank == 0:
            adapter_dir = Path(export_dir) / "adapter"
            assert (adapter_dir / "adapter_config.json").is_file()
            adapter_state = load_safetensors(adapter_dir / "adapter_model.safetensors")
            assert adapter_state
            assert all("lora_" in name for name in adapter_state)
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.slow
def test_fsdp2_wrap_and_dcp_checkpoint_round_trip(tmp_path: Path) -> None:
    mp.start_processes(
        _fsdp2_checkpoint_worker,
        args=(
            str(tmp_path / "gloo-init"),
            str(tmp_path / "checkpoint"),
            str(tmp_path / "export"),
        ),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

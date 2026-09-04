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
from transformers import LlamaConfig, LlamaForCausalLM

from wavelet.configs.sft import FSDPConfig, LoRAConfig, ModelConfig
from wavelet.trainer.ckpt import AppState
from wavelet.trainer.distributed import ParallelDims, World
from wavelet.trainer.model import (
    apply_lora,
    export_model_for_save,
    is_fsdp_model,
    load_fsdp2_model_from_hf,
    maybe_wrap_fsdp,
    save_lora_adapter_snapshot_from_fsdp,
    setup_model,
)


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        return value.to_local()
    return value


def _fsdp2_checkpoint_worker(
    rank: int,
    init_file: str,
    model_dir: str,
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
        model_config = ModelConfig(
            name=model_dir,
            meta_device_init=True,
            torch_dtype="float32",
            attn_implementation="sdpa",
        )
        parallel_dims = ParallelDims(world_size=2)
        model = setup_model(
            model_config,
            max_seq_length=16,
            distributed=True,
            parallel_dims=parallel_dims,
            initialize_on_meta=True,
        )
        assert all(parameter.is_meta for parameter in model.parameters())
        model = apply_lora(model, LoRAConfig(rank=2, alpha=4))
        world = World(
            rank=rank,
            local_rank=rank,
            world_size=2,
            local_world_size=2,
            device=torch.device("cpu"),
        )
        model = maybe_wrap_fsdp(
            model,
            model_config=model_config,
            fsdp_config=FSDPConfig(enabled=True, impl="fsdp2"),
            world=world,
            parallel_dims=parallel_dims,
        )

        assert is_fsdp_model(model)
        assert isinstance(model, FSDPModule)
        assert all(
            isinstance(block, FSDPModule)
            for block in model.base_model.model.model.layers
        )

        load_fsdp2_model_from_hf(model, model_config, world=world)
        _, loaded_state = export_model_for_save(model)
        if rank == 0:
            assert loaded_state is not None
            source_state = load_safetensors(Path(model_dir) / "model.safetensors")
            for name, value in loaded_state.items():
                if ".lora_A." in name:
                    assert torch.isfinite(value).all()
                    assert torch.count_nonzero(value) > 0
                    continue
                if ".lora_B." in name:
                    torch.testing.assert_close(value, torch.zeros_like(value))
                    continue
                source_name = name.removeprefix("base_model.model.").replace(
                    ".base_layer.", "."
                )
                if source_name == "lm_head.weight":
                    source_name = "model.embed_tokens.weight"
                torch.testing.assert_close(value, source_state[source_name])

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
    torch.manual_seed(11)
    source = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
    )
    model_dir = tmp_path / "model"
    source.save_pretrained(model_dir)
    mp.start_processes(
        _fsdp2_checkpoint_worker,
        args=(
            str(tmp_path / "gloo-init"),
            str(model_dir),
            str(tmp_path / "checkpoint"),
            str(tmp_path / "export"),
        ),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

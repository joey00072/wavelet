from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor
from transformers import (
    NemotronHConfig,
    NemotronHForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from wavelet.configs.config import FSDPConfig, ModelConfig
from wavelet.trainer.ckpt import AppState
from wavelet.trainer.distributed import (
    ParallelDims,
    World,
    clip_grad_norm_across_meshes_,
)
from wavelet.trainer.model import (
    load_fsdp2_model_from_hf,
    maybe_wrap_ddp,
    maybe_wrap_fsdp,
    setup_model,
)
from wavelet.trainer.models.deepseek_v4 import DeepseekV4Config
from wavelet.trainer.models.deepseek_v4.modeling import DeepseekV4ForCausalLM
from wavelet.trainer.moe import (
    configure_hf_moe_expert_parallel,
    configure_hf_moe_routers,
    hf_moe_experts,
    install_moe_load_balance_hook,
)


def _tiny_qwen_moe() -> Qwen3MoeForCausalLM:
    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
            head_dim=4,
            max_position_embeddings=16,
        )
    )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _frozen_expert_parallel_worker(rank: int, init_file: str, grouped: bool) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(23)
        model = _tiny_qwen_moe()
        experts = hf_moe_experts(model)[0]
        experts.requires_grad_(False)
        baseline = copy.deepcopy(experts)
        dims = ParallelDims(dp_shard=2, ep=2, world_size=2)
        configure_hf_moe_expert_parallel(model, dims, grouped_mm=grouped)
        assert all(not p.requires_grad for p in experts.parameters())
        hidden = torch.randn(4, 8, requires_grad=True)
        baseline_hidden = hidden.detach().clone().requires_grad_(True)
        selected = torch.tensor([[0, 2], [3, 1], [1, 2], [2, 0]])
        weights = torch.tensor([[0.7, 0.3]]).expand(4, -1).clone().requires_grad_()
        baseline_weights = weights.detach().clone().requires_grad_()
        output = experts(hidden, selected, weights)
        expected = baseline(baseline_hidden, selected, baseline_weights)
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(hidden.grad, baseline_hidden.grad)
        torch.testing.assert_close(weights.grad, baseline_weights.grad)
        assert all(p.grad is None for p in experts.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.parametrize("grouped", [False, True])
def test_frozen_experts_preserve_input_and_router_gradients(
    tmp_path: Path, grouped
) -> None:
    mp.spawn(
        _frozen_expert_parallel_worker,
        args=(str(tmp_path / "frozen-ep-init"), grouped),
        nprocs=2,
        join=True,
    )


def _tiny_new_expert_model(kind: str) -> torch.nn.Module:
    if kind == "qwen35":
        return Qwen3_5MoeForCausalLM(
            Qwen3_5MoeTextConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                moe_intermediate_size=4,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                num_experts=4,
                num_experts_per_tok=2,
                layer_types=["full_attention"],
            )
        )
    return NemotronHForCausalLM(
        NemotronHConfig(
            vocab_size=32,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            n_routed_experts=4,
            moe_intermediate_size=4,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            layers_block_type=["moe"],
            use_mamba_kernels=False,
        )
    )


def _expert_parallel_worker(
    rank: int,
    init_file: str,
    model_dir: str,
    checkpoint_dir: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(17)
        model = _tiny_qwen_moe()
        baseline = copy.deepcopy(hf_moe_experts(model)[0])
        dims = ParallelDims(dp_shard=2, ep=2, world_size=2)

        assert configure_hf_moe_expert_parallel(model, dims) == 1
        experts = hf_moe_experts(model)[0]
        assert isinstance(experts.gate_up_proj, DTensor)
        assert experts.gate_up_proj.to_local().shape[0] == 2

        torch.manual_seed(100 + rank)
        hidden = torch.randn(5, 8, requires_grad=True)
        baseline_hidden = hidden.detach().clone().requires_grad_(True)
        selected = torch.tensor(
            [[0, 2], [3, 1], [1, 2], [2, 0], [3, 0]], dtype=torch.long
        )
        weights = torch.randn(5, 2).softmax(dim=-1).requires_grad_(True)
        baseline_weights = weights.detach().clone().requires_grad_(True)

        output = experts(hidden, selected, weights)
        expected = baseline(baseline_hidden, selected, baseline_weights)
        torch.testing.assert_close(output, expected)

        output.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(hidden.grad, baseline_hidden.grad)
        torch.testing.assert_close(weights.grad, baseline_weights.grad)

        local_start = rank * 2
        for name, parameter in experts.named_parameters(recurse=False):
            baseline_parameter = getattr(baseline, name)
            assert baseline_parameter.grad is not None
            dist.all_reduce(baseline_parameter.grad)
            local_grad = _local_tensor(parameter.grad)
            torch.testing.assert_close(
                local_grad,
                baseline_parameter.grad[local_start : local_start + 2],
            )

        model_config = ModelConfig(torch_dtype="float32")
        torch.manual_seed(51)
        sharded_model = _tiny_qwen_moe()
        reference_model = copy.deepcopy(sharded_model)
        configure_hf_moe_routers(sharded_model, model_config)
        sharded_model = maybe_wrap_fsdp(
            sharded_model,
            model_config=model_config,
            fsdp_config=FSDPConfig(enabled=True, impl="fsdp2", ep=2),
            world=World(
                rank=rank,
                local_rank=rank,
                world_size=2,
                local_world_size=2,
                device=torch.device("cpu"),
            ),
            parallel_dims=dims,
        )
        optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=1e-3)
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        actual_output = sharded_model(input_ids=input_ids, labels=input_ids)
        expected_output = reference_model(input_ids=input_ids, labels=input_ids)
        torch.testing.assert_close(actual_output.logits, expected_output.logits)
        loss = actual_output.loss
        assert torch.isfinite(loss)
        loss.backward()
        expected_output.loss.backward()
        for name, parameter in sorted(
            sharded_model.named_parameters(),
            key=lambda item: ".experts." not in item[0],
        ):
            reference_gradient = reference_model.get_parameter(name).grad
            if reference_gradient is not None:
                gradient = parameter.grad
                if isinstance(gradient, DTensor):
                    gradient = gradient.full_tensor()
                torch.testing.assert_close(
                    gradient,
                    reference_gradient,
                    msg=lambda message, name=name: f"{name}: {message}",
                )
        grad_norm = clip_grad_norm_across_meshes_(sharded_model.parameters(), 1.0)
        assert torch.isfinite(grad_norm)
        assert grad_norm > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dcp.save(
            {"app": AppState(sharded_model, optimizer, None)},
            checkpoint_id=checkpoint_dir,
        )
        expected_parameters = {
            name: _local_tensor(parameter).detach().clone()
            for name, parameter in sharded_model.named_parameters()
        }
        with torch.no_grad():
            for parameter in sharded_model.parameters():
                _local_tensor(parameter).fill_(99)
        dcp.load(
            {"app": AppState(sharded_model, optimizer, None)},
            checkpoint_id=checkpoint_dir,
        )
        for name, parameter in sharded_model.named_parameters():
            torch.testing.assert_close(
                _local_tensor(parameter),
                expected_parameters[name],
            )

        meta_config = ModelConfig(
            name=model_dir,
            meta_device_init=True,
            torch_dtype="float32",
            attn_implementation="sdpa",
        )
        meta_model = setup_model(
            meta_config,
            max_seq_length=16,
            distributed=True,
            parallel_dims=dims,
            initialize_on_meta=True,
        )
        assert all(parameter.is_meta for parameter in meta_model.parameters())
        configure_hf_moe_routers(meta_model, meta_config)
        meta_model = maybe_wrap_fsdp(
            meta_model,
            model_config=meta_config,
            fsdp_config=FSDPConfig(enabled=True, impl="fsdp2", ep=2),
            world=World(
                rank=rank,
                local_rank=rank,
                world_size=2,
                local_world_size=2,
                device=torch.device("cpu"),
            ),
            parallel_dims=dims,
        )
        load_fsdp2_model_from_hf(
            meta_model,
            meta_config,
            world=World(
                rank=rank,
                local_rank=rank,
                world_size=2,
                local_world_size=2,
                device=torch.device("cpu"),
            ),
        )
        non_finite = [
            name
            for name, parameter in meta_model.named_parameters()
            if not torch.isfinite(_local_tensor(parameter)).all()
        ]
        assert not non_finite, non_finite
        unreasonable = [
            (name, float(_local_tensor(parameter).detach().abs().max()))
            for name, parameter in meta_model.named_parameters()
            if _local_tensor(parameter).numel()
            and _local_tensor(parameter).abs().max() > 10
        ]
        assert not unreasonable, unreasonable
        reference_model = Qwen3MoeForCausalLM.from_pretrained(model_dir)
        reference_model.eval()
        meta_model.eval()
        with torch.no_grad():
            loaded_logits = meta_model(input_ids=input_ids).logits
            reference_logits = reference_model(input_ids=input_ids).logits
        torch.testing.assert_close(loaded_logits, reference_logits)
        loaded_loss = meta_model(input_ids=input_ids, labels=input_ids).loss
        assert torch.isfinite(loaded_loss)
    finally:
        dist.destroy_process_group()


def _deepseek_csa_ddp_worker(rank: int, init_file: str) -> None:
    """Run real CSA backward twice, including the rank-local bias counters."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        config = DeepseekV4Config(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=4,
            q_lora_rank=4,
            partial_rotary_factor=1.0,
            layer_types=["compressed_sparse_attention"],
            compress_rates={"compressed_sparse_attention": 2},
            sliding_window=4,
            o_groups=2,
            o_lora_rank=2,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=2,
            moe_intermediate_size=8,
            n_routed_experts=2,
            num_experts_per_tok=1,
            num_hash_layers=0,
            hc_mult=2,
            max_position_embeddings=16,
        )
        model = DeepseekV4ForCausalLM(config)
        ddp = maybe_wrap_ddp(
            model,
            model_config=ModelConfig(torch_dtype="float32"),
            world=World(
                rank=rank,
                local_rank=rank,
                world_size=2,
                local_world_size=2,
                device=torch.device("cpu"),
            ),
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-3)
        handle = install_moe_load_balance_hook(optimizer, ddp)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        try:
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = ddp(ids, labels=ids).loss
                assert loss is not None and torch.isfinite(loss)
                loss.backward()
                # Deliberately make local observations unequal; the hook must
                # reduce them before applying the same centered update.
                with torch.no_grad():
                    model.model.layers[0].mlp.tokens_per_expert.add_(
                        torch.tensor([rank + 1.0, 0.0])
                    )
                optimizer.step()
            bias = model.model.layers[0].mlp.router.selection_bias
            gathered = [torch.empty_like(bias) for _ in range(2)]
            dist.all_gather(gathered, bias)
            torch.testing.assert_close(gathered[0], gathered[1])
        finally:
            handle.remove()
    finally:
        dist.destroy_process_group()


def test_deepseek_csa_ddp_two_steps_and_balancing(tmp_path):
    init_file = tmp_path / "csa-ddp-init"
    mp.spawn(
        _deepseek_csa_ddp_worker,
        args=(str(init_file),),
        nprocs=2,
        join=True,
    )


def _new_expert_parallel_worker(rank: int, init_file: str, kind: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(23)
        model = _tiny_new_expert_model(kind)
        baseline = copy.deepcopy(hf_moe_experts(model)[0])
        dims = ParallelDims(dp_shard=2, ep=2, world_size=2)
        configure_hf_moe_expert_parallel(model, dims)
        experts = hf_moe_experts(model)[0]
        hidden = torch.randn(4, 8, requires_grad=True)
        baseline_hidden = hidden.detach().clone().requires_grad_(True)
        selected = torch.full((4, 2), 0 if rank == 0 else 2, dtype=torch.long)
        weights = torch.tensor([[0.7, 0.3]]).expand(4, -1).clone()
        output = experts(hidden, selected, weights)
        expected = baseline(baseline_hidden, selected, weights)
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(hidden.grad, baseline_hidden.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.slow
def test_hf_expert_parallel_forward_and_backward(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _tiny_qwen_moe().save_pretrained(model_dir)
    mp.start_processes(
        _expert_parallel_worker,
        args=(
            str(tmp_path / "gloo-init"),
            str(model_dir),
            str(tmp_path / "checkpoint"),
        ),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("kind", ["qwen35", "nemotron"])
def test_new_hf_expert_parallel_forward_and_backward(tmp_path: Path, kind: str) -> None:
    mp.start_processes(
        _new_expert_parallel_worker,
        args=(str(tmp_path / f"{kind}-gloo-init"), kind),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

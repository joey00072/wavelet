from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    GptOssConfig,
    NemotronHConfig,
    Qwen3_5MoeConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

from wavelet.configs.config import FSDPConfig, ModelConfig, SFTConfig
from wavelet.trainer import model as model_utils
from wavelet.trainer.model import _fsdp_mixed_precision
from wavelet.trainer.moe import (
    _configure_expert_module,
    _run_local_experts,
    configure_hf_moe_routers,
    hf_moe_routers,
    moe_load_balance_metrics,
)
from wavelet.trainer.trainer import SFTTrainer


def _tiny_qwen_moe() -> Qwen3MoeForCausalLM:
    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
            head_dim=8,
            max_position_embeddings=32,
        )
    )


@pytest.mark.parametrize("trainable", [False, True])
def test_expert_parallel_preserves_parameter_trainability(monkeypatch, trainable):
    experts = _tiny_qwen_moe().model.layers[0].mlp.experts
    experts.requires_grad_(trainable)
    original = {name: p.detach().clone() for name, p in experts.named_parameters()}
    monkeypatch.setattr(
        "wavelet.trainer.moe.distribute_tensor",
        lambda parameter, *args, **kwargs: parameter.detach().clone(),
    )
    mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: object())
    _configure_expert_module(experts, mesh)
    for name, parameter in experts.named_parameters():
        assert parameter.requires_grad is trainable
        torch.testing.assert_close(parameter, original[name])


def test_hf_moe_router_controls_freeze_and_run_gate_in_fp32() -> None:
    model = _tiny_qwen_moe().to(dtype=torch.bfloat16)
    config = ModelConfig(
        freeze_moe_router=True,
        moe_router_dtype="float32",
    )

    assert configure_hf_moe_routers(model, config) == 1
    assert model.router_aux_loss_coef == 0.0

    router = hf_moe_routers(model)[0]
    assert all(parameter.dtype is torch.float32 for parameter in router.parameters())
    assert all(not parameter.requires_grad for parameter in router.parameters())
    outputs = model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert outputs.router_logits is not None
    assert outputs.router_logits[0].dtype is torch.float32

    metrics = moe_load_balance_metrics(model, outputs)

    assert set(metrics) == {
        "moe/max_vio",
        "moe/max_vio/max",
        "moe/routing_confidence",
    }
    assert metrics["moe/max_vio"] >= 0
    assert 0 < metrics["moe/routing_confidence"] <= 1


@pytest.mark.parametrize("router_dtype", ["none", "float32"])
def test_meta_moe_router_configuration_preserves_checkpoint_loadability(
    router_dtype: str,
) -> None:
    with torch.device("meta"):
        model = _tiny_qwen_moe()

    # Trainer configures routers after meta construction, before the FSDP loader
    # checks that every nonpersistent buffer can be reconstructed.
    configure_hf_moe_routers(
        model,
        ModelConfig(moe_router_dtype=router_dtype, freeze_moe_router=True),
    )
    assert all(parameter.is_meta for parameter in model.parameters())
    model_utils._validate_meta_model_buffers(model, set(model.state_dict()))


@pytest.mark.parametrize("grouped", [False, True])
def test_gpt_oss_local_expert_compute_preserves_hf_weight_layout(grouped) -> None:
    experts = GptOssExperts(
        GptOssConfig(
            hidden_size=8,
            intermediate_size=4,
            num_local_experts=4,
            num_experts_per_tok=1,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
    )
    experts._wavelet_ep_local_experts = 4
    experts._wavelet_ep_grouped_mm = grouped
    generator = torch.Generator().manual_seed(37)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    hidden = torch.randn(6, 8, requires_grad=True)
    selected = torch.tensor([0, 3, 1, 2, 0, 3])

    expected = experts(
        hidden,
        selected.unsqueeze(-1),
        torch.ones(6, 1),
    )
    actual = _run_local_experts(experts, hidden, selected)

    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    expected_hidden_grad = hidden.grad
    hidden.grad = None
    actual.sum().backward()
    torch.testing.assert_close(hidden.grad, expected_hidden_grad)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("kind", ["qwen35", "nemotron"])
def test_new_hf_expert_local_compute_matches_eager(kind: str, grouped) -> None:
    if kind == "qwen35":
        config = Qwen3_5MoeConfig(
            text_config={
                "hidden_size": 16,
                "moe_intermediate_size": 8,
                "num_experts": 4,
                "num_experts_per_tok": 2,
            }
        )
        experts = Qwen3_5MoeExperts(config.text_config)
    else:
        config = NemotronHConfig(
            hidden_size=16,
            moe_intermediate_size=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
        )
        experts = NemotronHExperts(config)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
    experts._wavelet_ep_local_experts = 4
    hidden = torch.randn(6, 16)
    selected = torch.tensor([0, 3, 1, 2, 0, 3])
    expected = experts(hidden, selected.unsqueeze(-1), torch.ones(6, 1))
    experts._wavelet_ep_grouped_mm = grouped
    actual = _run_local_experts(experts, hidden, selected)
    torch.testing.assert_close(actual, expected)

    eager_hidden = hidden.detach().clone().requires_grad_()
    local_hidden = hidden.detach().clone().requires_grad_()
    eager = experts(eager_hidden, selected.unsqueeze(-1), torch.ones(6, 1))
    local = _run_local_experts(experts, local_hidden, selected)
    eager.sum().backward()
    local.sum().backward()
    torch.testing.assert_close(local_hidden.grad, eager_hidden.grad)
    for parameter in experts.parameters():
        assert parameter.grad is not None


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped_mm parity requires CUDA"
)
def test_grouped_mm_experts_match_eager_outputs_and_parameter_gradients() -> None:
    if not hasattr(torch.nn.functional, "grouped_mm"):
        pytest.skip("grouped_mm requires a newer torch build")
    eager = _tiny_qwen_moe()
    grouped = _tiny_qwen_moe()
    grouped.load_state_dict(eager.state_dict())
    eager.set_experts_implementation("eager")
    grouped.set_experts_implementation("grouped_mm")
    eager = eager.to(device="cuda", dtype=torch.bfloat16)
    grouped = grouped.to(device="cuda", dtype=torch.bfloat16)
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    eager_out = eager(input_ids=input_ids).logits
    grouped_out = grouped(input_ids=input_ids).logits
    torch.testing.assert_close(grouped_out, eager_out, rtol=2e-3, atol=2e-3)
    eager_out.sum().backward()
    grouped_out.sum().backward()
    for left, right in zip(eager.parameters(), grouped.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("trainable", [False, True])
def test_local_grouped_experts_match_eager_gradients_on_cuda(trainable):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    torch.manual_seed(42)
    experts = Qwen3MoeExperts(
        Qwen3MoeConfig(hidden_size=256, moe_intermediate_size=128, num_experts=8)
    ).to(device="cuda", dtype=torch.bfloat16)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    experts.requires_grad_(trainable)
    experts._wavelet_ep_local_experts = 8
    grouped = copy.deepcopy(experts)
    grouped._wavelet_ep_grouped_mm = True
    hidden = torch.randn(
        153, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    other_hidden = hidden.detach().clone().requires_grad_()
    # Deliberately uneven groups and two empty experts.
    selected = torch.randint(0, 6, (153,), device="cuda")
    expected = _run_local_experts(experts, hidden, selected)
    actual = _run_local_experts(grouped, other_hidden, selected)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(other_hidden.grad, hidden.grad, rtol=0.02, atol=0.002)
    for left, right in zip(experts.parameters(), grouped.parameters(), strict=True):
        if trainable:
            torch.testing.assert_close(right.grad, left.grad, rtol=0.02, atol=0.004)
        else:
            assert left.grad is None and right.grad is None


def test_moe_metrics_measure_maximum_load_violation() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(num_experts_per_tok=1),
    )
    outputs = SimpleNamespace(
        router_logits=(
            torch.tensor(
                [
                    [9.0, 0.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0, 0.0],
                    [7.0, 0.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0, 0.0],
                ]
            ),
        )
    )

    metrics = moe_load_balance_metrics(model, outputs)

    assert metrics["moe/max_vio"] == pytest.approx(3.0)
    assert metrics["moe/max_vio/max"] == pytest.approx(3.0)


def test_freeze_moe_router_rejects_dense_model() -> None:
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace()

    with pytest.raises(ValueError, match="Qwen3-MoE or GPT-OSS"):
        configure_hf_moe_routers(
            model,
            ModelConfig(freeze_moe_router=True),
        )


def test_fsdp1_mixed_precision_exempts_fp32_router_class(monkeypatch) -> None:
    class Router(torch.nn.Module):
        pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    policy = _fsdp_mixed_precision(
        ModelConfig(torch_dtype="float32"),
        module_classes_to_ignore=(Router,),
    )

    assert policy is not None
    assert Router in policy._module_classes_to_ignore


def test_fsdp2_shards_fp32_router_with_separate_policy(monkeypatch) -> None:
    model = _tiny_qwen_moe().to(dtype=torch.bfloat16)
    config = ModelConfig(moe_router_dtype="float32")
    configure_hf_moe_routers(model, config)
    calls: list[tuple[torch.nn.Module, dict[str, object]]] = []

    class _ParallelDims:
        ep_enabled = False
        cp_enabled = False

        def get_mesh(self, name: str) -> object:
            assert name == "hsdp"
            return object()

    monkeypatch.setattr(
        model_utils,
        "fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )

    model_utils._wrap_fsdp2(
        model,
        model_config=config,
        fsdp_config=FSDPConfig(enabled=True, impl="fsdp2"),
        parallel_dims=_ParallelDims(),  # type: ignore[arg-type]
    )

    router = hf_moe_routers(model)[0]
    router_call = next(kwargs for module, kwargs in calls if module is router)
    assert router_call["mp_policy"].param_dtype is torch.float32
    assert calls[-1][0] is model


def test_sft_moe_metric_accumulation_uses_max_for_maximum() -> None:
    trainer = SFTTrainer(SFTConfig())
    trainer._sft_moe_metric_accum = [
        {"moe/max_vio": 1.0, "moe/max_vio/max": 2.0},
        {"moe/max_vio": 3.0, "moe/max_vio/max": 4.0},
    ]

    assert trainer._aggregate_sft_moe_metrics() == {
        "moe/max_vio": 2.0,
        "moe/max_vio/max": 4.0,
    }
    assert trainer._sft_moe_metric_accum == []

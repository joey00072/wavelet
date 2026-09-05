from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import GptOssConfig, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from wavelet.configs.sft import FSDPConfig, ModelConfig, SFTConfig
from wavelet.trainer import model as model_utils
from wavelet.trainer.model import _fsdp_mixed_precision
from wavelet.trainer.moe import (
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


def test_hf_moe_router_controls_freeze_and_run_gate_in_fp32() -> None:
    model = _tiny_qwen_moe().to(dtype=torch.bfloat16)
    config = ModelConfig(
        freeze_moe_router=True,
        moe_router_dtype="float32",
    )

    assert configure_hf_moe_routers(model, config) == 1

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


def test_gpt_oss_local_expert_compute_preserves_hf_weight_layout() -> None:
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
    hidden = torch.randn(6, 8)
    selected = torch.tensor([0, 3, 1, 2, 0, 3])

    expected = experts(
        hidden,
        selected.unsqueeze(-1),
        torch.ones(6, 1),
    )
    actual = _run_local_experts(experts, hidden, selected)

    torch.testing.assert_close(actual, expected)


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

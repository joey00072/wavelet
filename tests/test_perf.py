from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from wavelet.configs.rl_config import RLConfig
from wavelet.monitor import emit_perf
from wavelet.trainer.distributed import World
from wavelet.trainer.perf import (
    estimate_active_matmul_parameters,
    estimate_attention_flops_per_token,
    estimate_training_flops_per_token,
    peak_flops_per_second,
    training_flop_metrics,
)
from wavelet.trainer.rl import RLTrainer


def test_emit_perf_preserves_line_shape(monkeypatch, capsys) -> None:
    monkeypatch.setenv("WAVELET_PERF_LOG", "1")

    emit_perf("example", step=2, seconds=0.125, mode="async")

    assert capsys.readouterr().out == (
        "WAVELET_PERF example step=2 seconds=0.125 mode=async\n"
    )


def _dense_config() -> PretrainedConfig:
    return PretrainedConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )


class _ConfiguredModel(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(1))


def test_dense_training_flop_estimate_matches_component_formula() -> None:
    config = _dense_config()
    model = _ConfiguredModel(config)

    active_parameters = estimate_active_matmul_parameters(config)
    attention_flops = estimate_attention_flops_per_token(config, seq_len=10)

    assert active_parameters == 1_408
    assert attention_flops == 1_920
    assert estimate_training_flops_per_token(model, seq_len=10) == 10_368


def test_moe_estimate_counts_only_active_experts() -> None:
    config = PretrainedConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=12,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        first_k_dense_replace=1,
        num_experts_per_tok=2,
        num_experts=8,
        shared_expert_intermediate_size=12,
    )

    active_parameters = estimate_active_matmul_parameters(config)

    assert active_parameters == 4_192
    all_expert_mlp_parameters = 3 * 8 * 3 * 12 * 8
    assert active_parameters is not None
    assert active_parameters < all_expert_mlp_parameters + 2_048


def test_lora_estimate_uses_frozen_base_backward_cost() -> None:
    class LoraModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _dense_config()
            self.base = nn.Parameter(torch.ones(10), requires_grad=False)
            self.lora_A = nn.Parameter(torch.ones(3))
            self.lora_B = nn.Parameter(torch.ones(4))

    assert estimate_training_flops_per_token(LoraModel(), seq_len=10) == 7_594


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("NVIDIA A100-SXM4-80GB", 312e12),
        ("NVIDIA H100 PCIe", 756e12),
        ("NVIDIA H100 NVL", 835e12),
        ("NVIDIA H200", 989e12),
        ("NVIDIA B200", 2.25e15),
        ("NVIDIA GB200", 2.5e15),
        ("AMD Instinct MI325X", 1307.4e12),
    ],
)
def test_peak_flops_table(device_name: str, expected: float) -> None:
    assert peak_flops_per_second(device_name, dtype=torch.bfloat16) == expected


def test_training_flop_metrics_reports_percent_mfu() -> None:
    metrics = training_flop_metrics(
        flops_per_token=1_000,
        model_tokens=2_000,
        elapsed_seconds=0.5,
        world_size=2,
        dtype=torch.bfloat16,
        device_name="NVIDIA A100",
    )

    assert metrics["perf/model_flops_per_token"] == 1_000
    assert metrics["perf/peak_flops_per_second_per_gpu"] == 312e12
    assert metrics["perf/mfu"] == pytest.approx(100 * 4e6 / (2 * 312e12))


def test_training_flop_metrics_omits_mfu_for_unknown_hardware() -> None:
    metrics = training_flop_metrics(
        flops_per_token=1_000,
        model_tokens=2_000,
        elapsed_seconds=0.5,
        world_size=1,
        dtype=torch.bfloat16,
        device_name="Unknown Accelerator",
    )

    assert metrics == {"perf/model_flops_per_token": 1_000.0}


def test_rl_step_performance_uses_accumulated_compute_time(monkeypatch) -> None:
    trainer = RLTrainer(RLConfig())
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
    )
    trainer._step_compute_seconds = 0.5
    trainer._step_model_tokens = 10
    trainer._model_flops_per_token = 1_000
    trainer._model_compute_dtype = torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    metrics = trainer._finish_step_performance_metrics()

    assert metrics["perf/tokens_per_second"] == pytest.approx(40.0)
    assert metrics["perf/model_flops_per_token"] == 1_000.0
    assert trainer._step_compute_seconds == 0.0
    assert trainer._step_model_tokens == 0

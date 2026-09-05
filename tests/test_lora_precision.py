from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel

from wavelet.configs.config import LoRAConfig, OptimizerConfig
from wavelet.configs.rl_config import RLConfig
from wavelet.trainer.model import apply_lora, save_lora_adapter_snapshot
from wavelet.trainer.optim import setup_optimizer
from wavelet.trainer.trainer import BaseTrainer, _lora_dtype


def _tiny_lora_model(*, lora_dtype: torch.dtype) -> torch.nn.Module:
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=16,
            vocab_size=16,
        )
    ).to(dtype=torch.bfloat16)
    return apply_lora(
        model,
        LoRAConfig(rank=2, alpha=4, target_modules=["c_attn"]),
        lora_dtype=lora_dtype,
    )


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def test_lora_optimization_dtype_overrides_base_parameter_dtype() -> None:
    assert _lora_dtype("bfloat16", "model") == torch.bfloat16
    assert _lora_dtype("bfloat16", "float32") == torch.float32


def test_fp32_lora_parameters_remain_distinct_from_bfloat16_base() -> None:
    model = _tiny_lora_model(lora_dtype=torch.float32)

    trainable = _trainable_parameters(model)
    frozen = [
        parameter for parameter in model.parameters() if not parameter.requires_grad
    ]

    assert trainable
    assert all(parameter.dtype == torch.float32 for parameter in trainable)
    assert frozen
    assert all(parameter.dtype == torch.bfloat16 for parameter in frozen)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(input_ids=torch.tensor([[1, 2, 3]])).logits
    logits.float().sum().backward()

    assert logits.dtype == torch.bfloat16
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.float32
        for parameter in trainable
    )


def test_existing_adapter_is_cast_to_requested_optimization_dtype() -> None:
    model = _tiny_lora_model(lora_dtype=torch.bfloat16)

    result = apply_lora(
        model,
        LoRAConfig(rank=2, alpha=4, target_modules=["c_attn"]),
        lora_dtype=torch.float32,
    )

    assert result is model
    assert all(
        parameter.dtype == torch.float32 for parameter in _trainable_parameters(model)
    )


def test_adam_updates_and_state_keep_fp32_lora_fidelity() -> None:
    model = _tiny_lora_model(lora_dtype=torch.float32)
    optimizer = setup_optimizer(
        OptimizerConfig(
            type="adamw",
            implementation="for-loop",
            lr=1e-5,
            weight_decay=0.0,
        ),
        model.named_parameters(),
    )
    parameter = _trainable_parameters(model)[0]
    parameter.data.fill_(1.0)
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    assert parameter.dtype == torch.float32
    assert not torch.equal(parameter, torch.ones_like(parameter))
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32


def test_policy_snapshot_preserves_fp32_adapter_tensors(tmp_path: Path) -> None:
    model = _tiny_lora_model(lora_dtype=torch.float32)

    adapter_dir = save_lora_adapter_snapshot(model, tmp_path)
    state = load_file(adapter_dir / "adapter_model.safetensors")

    assert state
    assert all(tensor.dtype == torch.float32 for tensor in state.values())


def test_fp32_lora_uses_base_dtype_cuda_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = BaseTrainer(
        RLConfig(
            model={"torch_dtype": "bfloat16"},
            lora={"optimization_dtype": "float32"},
        )
    )
    trainer.world = SimpleNamespace(device=torch.device("cuda"))
    calls: list[tuple[str, torch.dtype]] = []

    def autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return contextlib.nullcontext()

    monkeypatch.setattr(torch, "autocast", autocast)

    with trainer._model_forward_context():
        pass

    assert calls == [("cuda", torch.bfloat16)]


def test_fused_lora_rejects_distinct_optimization_dtype() -> None:
    with pytest.raises(ValueError, match="distinct LoRA optimization dtype"):
        RLConfig(
            model={"fused_lora_mlp": True},
            lora={"optimization_dtype": "float32"},
        )

import pytest
import torch

from wavelet.configs.sft import OptimizerConfig
from wavelet.trainer.optim import setup_optimizer


def test_optimizer_rejects_multiple_trainable_lora_adapters() -> None:
    params = [
        (
            "base.q_proj.lora_A.default.weight",
            torch.nn.Parameter(torch.ones(2, 2)),
        ),
        (
            "base.q_proj.lora_A.policy_old.weight",
            torch.nn.Parameter(torch.ones(2, 2)),
        ),
    ]

    with pytest.raises(RuntimeError, match="exactly one LoRA adapter"):
        setup_optimizer(OptimizerConfig(type="adamw"), params)


def test_optimizer_allows_single_trainable_lora_adapter() -> None:
    params = [
        (
            "base.q_proj.lora_A.default.weight",
            torch.nn.Parameter(torch.ones(2, 2)),
        ),
        (
            "base.q_proj.lora_B.default.weight",
            torch.nn.Parameter(torch.ones(2, 2)),
        ),
    ]

    optimizer = setup_optimizer(OptimizerConfig(type="adamw"), params)

    assert len(optimizer.param_groups[0]["params"]) == 2


def test_optimizer_accepts_named_parameter_iterator() -> None:
    module = torch.nn.Linear(2, 1)

    optimizer = setup_optimizer(OptimizerConfig(type="adamw"), module.named_parameters())

    assert len(optimizer.param_groups[0]["params"]) == 2

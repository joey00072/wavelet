import itertools

import pytest
import torch

from wavelet.configs.sft import OptimizerConfig
from wavelet.trainer.optim import SignSGD, setup_linear_scheduler, setup_optimizer


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

    optimizer = setup_optimizer(
        OptimizerConfig(type="adamw"), module.named_parameters()
    )

    assert len(optimizer.param_groups[0]["params"]) == 2


def test_sign_sgd_applies_sign_update_and_decoupled_weight_decay() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    parameter.grad = torch.tensor([0.5, -0.25, 0.0])

    optimizer = setup_optimizer(
        OptimizerConfig(type="sign_sgd", lr=0.1, weight_decay=0.2),
        [("weight", parameter)],
    )
    optimizer.step()

    assert isinstance(optimizer, SignSGD)
    assert parameter.tolist() == pytest.approx([0.88, -1.86, 2.94])
    assert not optimizer.state


def test_linear_scheduler_decays_when_decay_would_start_before_warmup_ends() -> None:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = setup_linear_scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=50,
        decay_steps=80,
        lr=1.0,
        min_lr=0.1,
    )

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(99):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == pytest.approx(0.1)
    assert lrs[50] == pytest.approx(1.0)
    assert max(lrs) == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1)
    assert all(later <= earlier for earlier, later in itertools.pairwise(lrs[50:]))

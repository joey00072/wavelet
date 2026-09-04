import itertools

import pytest
import torch

from wavelet.configs.sft import OptimizerConfig, TrainerConfig
from wavelet.trainer.optim import (
    OptimizerStateOffloader,
    SignSGD,
    enable_optimizer_state_offload,
    setup_linear_scheduler,
    setup_optimizer,
)
from wavelet.trainer.trainer import BaseTrainer


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


def test_optimizer_state_offload_preserves_updates_and_cpu_state() -> None:
    baseline_parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    offloaded_parameter = torch.nn.Parameter(baseline_parameter.detach().clone())
    baseline = torch.optim.AdamW([baseline_parameter], lr=0.1)
    offloaded = torch.optim.AdamW([offloaded_parameter], lr=0.1)
    controller = enable_optimizer_state_offload(offloaded)

    for gradient in (torch.tensor([0.5, -0.25]), torch.tensor([-0.1, 0.2])):
        baseline_parameter.grad = gradient.clone()
        offloaded_parameter.grad = gradient.clone()
        baseline.step()
        offloaded.step()
        assert all(
            value.device.type == "cpu"
            for state in offloaded.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )

    assert isinstance(controller, OptimizerStateOffloader)
    assert offloaded_parameter.tolist() == pytest.approx(baseline_parameter.tolist())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimizer_state_offload_pins_cuda_optimizer_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, foreach=False)
    enable_optimizer_state_offload(optimizer)
    parameter.grad = torch.tensor([0.5], device="cuda")

    optimizer.step()

    state_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    ]
    assert state_tensors
    assert all(value.device.type == "cpu" for value in state_tensors)
    assert all(value.is_pinned() for value in state_tensors)


def test_optimizer_state_offload_survives_state_dict_round_trip() -> None:
    source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    source = torch.optim.AdamW([source_parameter], lr=0.1)
    enable_optimizer_state_offload(source)
    source_parameter.grad = torch.tensor([0.5])
    source.step()
    state_dict = source.state_dict()

    target_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    target = torch.optim.AdamW([target_parameter], lr=0.1)
    enable_optimizer_state_offload(target)
    target.load_state_dict(state_dict)

    assert target.state
    assert all(
        value.device.type == "cpu"
        for state in target.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def test_trainer_enables_configured_optimizer_state_offload() -> None:
    trainer = BaseTrainer(
        TrainerConfig(optim={"cpu_offload": True, "implementation": "for-loop"})
    )
    trainer.model = torch.nn.Linear(2, 1)  # type: ignore[assignment]

    trainer._setup_optimizer()

    assert trainer.optimizer is not None
    assert isinstance(
        getattr(trainer.optimizer, "_wavelet_state_offloader", None),
        OptimizerStateOffloader,
    )


@pytest.mark.parametrize(
    "optimizer_type",
    ["sign_sgd", "adamw_8bit", "paged_adamw_8bit", "adam_8bit"],
)
def test_optimizer_state_offload_rejects_bitsandbytes_state(
    optimizer_type: str,
) -> None:
    with pytest.raises(ValueError, match="cpu_offload"):
        OptimizerConfig(type=optimizer_type, cpu_offload=True)


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

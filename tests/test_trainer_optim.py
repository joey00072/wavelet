import itertools
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from wavelet.configs.config import OptimizerConfig, TrainerConfig
from wavelet.trainer.optim import (
    Muon,
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


def test_muon_updates_matrix_and_uses_adam_style_fallback_for_vector() -> None:
    matrix = torch.nn.Parameter(torch.eye(2))
    vector = torch.nn.Parameter(torch.ones(2))
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer = Muon([matrix, vector], lr=0.1)
    optimizer.step()
    assert torch.isfinite(matrix).all() and torch.isfinite(vector).all()
    assert "momentum" in optimizer.state[matrix]


def test_muon_vector_fallback_matches_adamw() -> None:
    muon_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    adam_parameter = torch.nn.Parameter(muon_parameter.detach().clone())
    muon_parameter.grad = torch.tensor([0.5, -0.25])
    adam_parameter.grad = muon_parameter.grad.clone()
    muon = Muon([muon_parameter], lr=0.1, weight_decay=0.2)
    adam = torch.optim.AdamW([adam_parameter], lr=0.1, weight_decay=0.2)
    muon.step()
    adam.step()
    assert muon_parameter.tolist() == pytest.approx(adam_parameter.tolist())
    assert "exp_avg" in muon.state[muon_parameter]


def test_muon_orthogonalization_handles_tall_and_batched_matrices() -> None:
    matrix = torch.randn(3, 20)
    result = Muon._orthogonalize(matrix)
    assert result.shape == matrix.shape
    batched = torch.randn(4, 3, 20)
    assert Muon._orthogonalize(batched).shape == batched.shape


def _dtensor_muon_worker(rank: int, world_size: int, rendezvous: str, dim: int) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    mesh = init_device_mesh("cpu", (world_size,))
    base = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    reference = torch.nn.Parameter(base.clone())
    sharded = torch.nn.Parameter(distribute_tensor(base, mesh, [Shard(dim)]))
    optimizer = Muon([sharded], lr=0.01, momentum=0.8, nesterov=True)
    expected = Muon([reference], lr=0.01, momentum=0.8, nesterov=True)
    torch.manual_seed(101 + dim)
    for _ in range(2):
        grad = torch.randn_like(base)
        reference.grad = grad
        sharded.grad = distribute_tensor(grad, mesh, [Shard(dim)])
        expected.step()
        optimizer.step()
    torch.testing.assert_close(sharded.full_tensor(), reference)
    state = optimizer.state_dict()
    resumed = Muon([sharded], lr=0.01, momentum=0.8, nesterov=True)
    resumed.load_state_dict(state)
    grad = torch.randn_like(base)
    reference.grad = grad
    sharded.grad = distribute_tensor(grad, mesh, [Shard(dim)])
    expected.step()
    resumed.step()
    torch.testing.assert_close(sharded.full_tensor(), reference)
    dist.destroy_process_group()


@pytest.mark.parametrize("dim", [0, 1])
def test_muon_dtensor_cpu_gloo_matches_unsharded_and_resumes(dim: int) -> None:
    with tempfile.NamedTemporaryFile() as rendezvous:
        mp.spawn(
            _dtensor_muon_worker, args=(2, rendezvous.name, dim), nprocs=2, join=True
        )


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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimizer_state_offload_reuses_pinned_buffers_across_steps() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, foreach=False)
    offloader = enable_optimizer_state_offload(optimizer)
    pointers = []
    for value in (0.5, -0.25, 0.75):
        parameter.grad = torch.tensor([value], device="cuda")
        optimizer.step()
        state = optimizer.state[parameter]
        pointers.append(
            {
                key: tensor.data_ptr()
                for key, tensor in state.items()
                if torch.is_tensor(tensor)
            }
        )
    assert pointers[0] == pointers[1] == pointers[2]
    assert all(
        tensor.is_pinned()
        for tensor in optimizer.state[parameter].values()
        if torch.is_tensor(tensor)
    )
    assert offloader._cpu_buffers


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

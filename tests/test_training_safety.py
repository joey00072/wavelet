from __future__ import annotations

import contextlib
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from wavelet.configs.config import RLConfig, SFTConfig
from wavelet.trainer.distributed import ParallelDims, World
from wavelet.trainer.rl import RLTrainer
from wavelet.trainer.trainer import SFTTrainer
from wavelet.trainer.types import LossOutput


class _NonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, gradient: float) -> torch.Tensor:
        ctx.gradient = gradient
        return value.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad * ctx.gradient, None


@pytest.mark.parametrize("kind", ["sft", "rl"])
@pytest.mark.parametrize("max_grad_norm", [0.0, 1.0])
@pytest.mark.parametrize("gradient", [float("inf"), float("nan")])
def test_nonfinite_backward_never_updates_weights_or_step(
    kind: str,
    max_grad_norm: float,
    gradient: float,
) -> None:
    if kind == "sft":
        trainer = SFTTrainer(SFTConfig(max_grad_norm=max_grad_norm))
    else:
        trainer = RLTrainer(RLConfig(max_grad_norm=max_grad_norm))
        trainer._set_optimizer_batch_loss_scales({"rl": 1, "ce": 1, "ref_kl": 1})
        trainer._record_micro_batch_metrics = Mock()
    model = torch.nn.Linear(1, 1, bias=False)
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters())
    trainer.scheduler = SimpleNamespace(step=Mock())
    initial = model.weight.detach().clone()

    def loss(*_args: object) -> LossOutput:
        return LossOutput(loss=_NonfiniteBackward.apply(model.weight, gradient).sum())

    if kind == "sft":
        trainer._forward_loss = loss
    else:
        trainer._forward_rl_loss = loss
    batch = {
        "input_ids": torch.ones((1, 1), dtype=torch.long),
        "labels": torch.ones((1, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 1), dtype=torch.long),
        "position_ids": torch.zeros((1, 1), dtype=torch.long),
        "loss_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    with pytest.raises(FloatingPointError, match="Non-finite gradients"):
        trainer._train_step(batch)

    torch.testing.assert_close(model.weight, initial)
    assert trainer.step == 0
    assert not trainer.optimizer.state
    assert model.weight.grad is None
    trainer.scheduler.step.assert_not_called()


@pytest.mark.parametrize("kind", ["sft", "rl"])
@pytest.mark.parametrize("fails", [True, False])
def test_final_export_precedes_completion_and_records_failure(
    kind: str,
    fails: bool,
) -> None:
    trainer = SFTTrainer(SFTConfig()) if kind == "sft" else RLTrainer(RLConfig())
    trainer.monitor = Mock()
    events: list[str] = []

    def export() -> None:
        events.append("export")
        if fails:
            raise OSError("disk full")

    trainer._save_model = export
    trainer.monitor.finish.side_effect = lambda **kwargs: events.append(
        kwargs["status"]
    )

    def finish() -> None:
        if kind == "sft":
            trainer._finish_if_requested(True, status="completed")
        else:
            trainer.finalize(status="completed")

    if fails:
        with pytest.raises(OSError, match="disk full"):
            finish()
    else:
        finish()

    assert events == ["export", "failed" if fails else "completed"]
    assert trainer._run_closed
    trainer.monitor.finish.assert_called_once_with(
        status="failed" if fails else "completed", step=0
    )


def test_final_checkpoint_failure_records_failed_run() -> None:
    trainer = RLTrainer(RLConfig())
    trainer.monitor = Mock()
    trainer.ckpt_manager = Mock()
    trainer.ckpt_manager.wait_for_pending_save.side_effect = OSError(
        "checkpoint failed"
    )
    trainer._save_model = Mock()

    with pytest.raises(OSError, match="checkpoint failed"):
        trainer.finalize()

    trainer.monitor.finish.assert_called_once_with(status="failed", step=0)
    trainer._save_model.assert_not_called()


def test_finite_gradients_with_overflowing_norm_abort_before_update() -> None:
    trainer = RLTrainer(RLConfig(max_grad_norm=1.0))
    trainer._set_optimizer_batch_loss_scales({"rl": 1, "ce": 1, "ref_kl": 1})
    trainer.model = torch.nn.Linear(2, 1, bias=False)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.scheduler = SimpleNamespace(step=Mock())
    initial = trainer.model.weight.detach().clone()
    trainer.model.weight.grad = torch.full_like(trainer.model.weight, 1e30)

    with pytest.raises(FloatingPointError, match="Non-finite gradient norm"):
        trainer._apply_optimizer_step()

    torch.testing.assert_close(trainer.model.weight, initial)
    assert trainer.model.weight.grad is None
    assert not trainer.optimizer.state
    trainer.scheduler.step.assert_not_called()


class _ReferenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.3, 0.1], [0.2, 0.4]]))
        self.register_buffer("offset", torch.ones(2))
        self.calls = 0

    def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(logits=self.weight[input_ids] + self.offset)

    def disable_adapter(self) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()


def _distributed_safety_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        model = _ReferenceModel()
        trainer = RLTrainer(RLConfig())
        trainer.world = World(
            rank=rank,
            local_rank=rank,
            world_size=2,
            local_world_size=2,
            device=torch.device("cpu"),
        )
        trainer.model = DistributedDataParallel(model)
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batch = {
            "input_ids": torch.tensor([[0, 1]]),
            "target_ids": torch.tensor([[1, 0]]),
            "position_ids": torch.tensor([[0, 1]]),
            "temperatures": torch.ones((1, 2)),
            "loss_mask": torch.ones((1, 2), dtype=torch.bool),
            "advantages": torch.ones((1, 2)),
            "inference_logprobs": torch.full((1, 2), -7.0),
            "has_inference_logprobs": torch.tensor([rank == 1]),
            "teacher_logprobs": torch.zeros((1, 2)),
            "has_teacher_logprobs": torch.tensor([False]),
        }
        references = trainer._inference_logprobs(batch, None)
        assert model.calls == 1
        if rank == 1:
            torch.testing.assert_close(references, batch["inference_logprobs"])
        else:
            assert not torch.equal(references, batch["inference_logprobs"])
        assert trainer.model.training

        # A padding-only rank must still participate in reference forwards.
        batch["loss_mask"][:] = rank == 0
        model.calls = 0
        output = trainer._forward_rl_loss(batch, None)
        assert model.calls == 2
        assert torch.isfinite(output.loss)
        output.loss.backward()
        trainer.optimizer.zero_grad(set_to_none=True)

        model.weight.grad = torch.full_like(
            model.weight, float("nan") if rank == 0 else 1
        )
        with pytest.raises(FloatingPointError, match="Non-finite gradients"):
            trainer._require_finite_gradients()
        assert model.weight.grad is None

        trainer.monitor = Mock()
        trainer._save_model = Mock(
            side_effect=OSError("disk full") if rank == 0 else None
        )
        with pytest.raises((OSError, RuntimeError)):
            trainer._finish_run(status="completed")
        trainer.monitor.finish.assert_called_once_with(status="failed", step=0)

        # Mixed CE/RL ranks must all enter component normalization.
        mixed = RLTrainer(RLConfig())
        mixed.world = trainer.world
        mixed.parallel_dims = ParallelDims(world_size=2)
        mixed.model = model
        mixed.optimizer = trainer.optimizer
        mixed.accumulation_steps = 1
        mixed._forward_rl_loss = Mock(return_value=LossOutput(loss=torch.tensor(0.0)))
        mixed._backward_rl_loss = Mock()
        mixed._record_micro_batch_metrics = Mock()
        mixed._apply_optimizer_step = Mock(return_value=None)
        mixed._complete_optimizer_step = Mock(return_value={})
        mixed._finish_step_performance_metrics = dict
        batch["loss_mask"][:] = True
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        batch["ce_weights"] = torch.full((1, 2), float(rank == 0))
        assert mixed._train_step(batch).stepped
        assert mixed._optimizer_batch_loss_scales == {
            "rl": 2.0,
            "ce": 1.0,
            "ref_kl": 1.0,
        }
    finally:
        dist.destroy_process_group()


def test_distributed_reference_padding_gradient_and_export_safety(
    tmp_path: Path,
) -> None:
    workers = mp.spawn(
        _distributed_safety_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Distributed training safety collectives did not finish")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

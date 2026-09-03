from __future__ import annotations

from concurrent.futures import Future

import pytest
import torch

from wavelet.configs.config import CheckpointConfig, TrainerConfig
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.distributed import World
from wavelet.trainer.trainer import BaseTrainer


def test_async_checkpoint_uses_threads_without_shared_memory(
    monkeypatch, tmp_path
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    manager = CheckpointManager(
        model,
        optimizer,
        None,
        CheckpointConfig(mode="async", interval=1),
        tmp_path,
        world,
    )
    captured = {}

    def async_save(**kwargs):
        captured.update(kwargs)
        response: Future[None] = Future()
        response.set_result(None)
        return response

    monkeypatch.setattr("wavelet.trainer.ckpt.dcp.async_save", async_save)

    assert manager.save(TrainerState(step=1, micro_step=1))
    manager.wait_for_pending_save()

    assert str(captured["async_checkpointer_type"].value) == "thread"
    assert captured["async_stager"]._config.use_shared_memory is False


def test_async_checkpoint_round_trip_restores_model_and_optimizer(tmp_path) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    manager = CheckpointManager(
        model,
        optimizer,
        None,
        CheckpointConfig(mode="async", interval=1),
        tmp_path,
        world,
    )
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected_weight = model.weight.detach().clone()

    assert manager.save(TrainerState(step=1, micro_step=3))
    manager.wait_for_pending_save()
    with torch.no_grad():
        model.weight.fill_(99.0)

    state = manager.load(tmp_path / "checkpoint-1")

    assert state == TrainerState(step=1, micro_step=3)
    assert torch.equal(model.weight, expected_weight)
    assert optimizer.state


def test_fixed_accumulation_trainer_still_rejects_misaligned_resume_state() -> None:
    trainer = BaseTrainer(TrainerConfig())
    trainer.accumulation_steps = 2

    with pytest.raises(ValueError, match="micro_step does not match"):
        trainer._validate_resume_state(TrainerState(step=3, micro_step=7))

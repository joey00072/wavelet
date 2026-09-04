from __future__ import annotations

import json
from concurrent.futures import Future

import pytest
import torch

from wavelet.configs.config import CheckpointConfig, TrainerConfig
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.distributed import World
from wavelet.trainer.optim import enable_optimizer_state_offload
from wavelet.trainer.trainer import BaseTrainer
from wavelet.utils.pathing import STABLE_CHECKPOINT_MARKER


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


def test_forced_checkpoint_saves_step_between_intervals(monkeypatch, tmp_path) -> None:
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
        CheckpointConfig(mode="async", interval=10),
        tmp_path,
        world,
    )
    response: Future[None] = Future()
    response.set_result(None)
    monkeypatch.setattr(
        "wavelet.trainer.ckpt.dcp.async_save", lambda **kwargs: response
    )

    assert not manager.save(TrainerState(step=3, micro_step=3))
    assert manager.save(TrainerState(step=3, micro_step=3), force=True)
    manager.wait_for_pending_save()

    assert (tmp_path / "checkpoint-3" / STABLE_CHECKPOINT_MARKER).exists()


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

    expected_state = TrainerState(
        step=1,
        micro_step=3,
        total_tokens=123,
        total_samples=7,
    )
    assert manager.save(expected_state)
    manager.wait_for_pending_save()
    with torch.no_grad():
        model.weight.fill_(99.0)

    state = manager.load(tmp_path / "checkpoint-1")

    assert state == expected_state
    assert torch.equal(model.weight, expected_weight)
    assert optimizer.state

    meta_path = tmp_path / "checkpoint-1" / "meta.json"
    legacy_metadata = json.loads(meta_path.read_text())
    legacy_metadata.pop("total_tokens")
    legacy_metadata.pop("total_samples")
    meta_path.write_text(json.dumps(legacy_metadata))

    legacy_state = manager.load(tmp_path / "checkpoint-1")

    assert legacy_state.total_tokens == 0
    assert legacy_state.total_samples == 0


def test_trainer_progress_counts_global_tokens_and_logical_samples() -> None:
    trainer = BaseTrainer(TrainerConfig())
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
    )

    trainer._record_progress({"input_ids": torch.ones((2, 3), dtype=torch.long)})
    trainer._record_progress(
        {
            "input_ids": torch.ones((2, 4), dtype=torch.long),
            "sample_counts": torch.tensor([1, 3]),
        }
    )

    assert trainer.total_tokens == 28
    assert trainer.total_samples == 12
    assert trainer._progress_metrics() == {
        "progress/total_tokens": 28.0,
        "progress/total_samples": 12.0,
    }


def test_trainer_rejects_negative_checkpoint_progress() -> None:
    trainer = BaseTrainer(TrainerConfig())

    with pytest.raises(ValueError, match="progress counters"):
        trainer._validate_resume_state(
            TrainerState(step=0, micro_step=0, total_tokens=-1)
        )


def test_checkpoint_round_trip_with_cpu_offloaded_optimizer_state(tmp_path) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    enable_optimizer_state_offload(optimizer)
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
    expected_exp_avg = optimizer.state[model.weight]["exp_avg"].clone()

    assert manager.save(TrainerState(step=1, micro_step=1))
    manager.wait_for_pending_save()
    with torch.no_grad():
        model.weight.fill_(99.0)
        optimizer.state[model.weight]["exp_avg"].fill_(99.0)

    manager.load(tmp_path / "checkpoint-1")

    assert torch.equal(model.weight, expected_weight)
    assert torch.equal(optimizer.state[model.weight]["exp_avg"], expected_exp_avg)
    assert optimizer.state[model.weight]["exp_avg"].device.type == "cpu"


def test_checkpoint_cleanup_preserves_interval_and_recent_steps(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
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
        CheckpointConfig(
            mode="async",
            interval=1,
            keep_last=2,
            keep_interval=3,
        ),
        tmp_path,
        world,
    )
    for step in range(1, 7):
        checkpoint_dir = tmp_path / f"checkpoint-{step}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()

    manager._maybe_clean()

    remaining = sorted(path.name for path in tmp_path.glob("checkpoint-*"))
    assert remaining == ["checkpoint-3", "checkpoint-5", "checkpoint-6"]


def test_fixed_accumulation_trainer_still_rejects_misaligned_resume_state() -> None:
    trainer = BaseTrainer(TrainerConfig())
    trainer.accumulation_steps = 2

    with pytest.raises(ValueError, match="micro_step does not match"):
        trainer._validate_resume_state(TrainerState(step=3, micro_step=7))

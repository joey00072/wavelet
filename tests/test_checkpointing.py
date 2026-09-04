from __future__ import annotations

import json
from concurrent.futures import Future
from unittest.mock import Mock

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


def test_checkpoint_load_can_skip_optimizer_scheduler_and_progress(tmp_path) -> None:
    world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    source_model = torch.nn.Linear(2, 1, bias=False)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
    source_scheduler = torch.optim.lr_scheduler.StepLR(
        source_optimizer,
        step_size=1,
        gamma=0.5,
    )
    source_model(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()
    source_scheduler.step()
    source_optimizer.zero_grad(set_to_none=True)
    expected_weight = source_model.weight.detach().clone()
    source_manager = CheckpointManager(
        source_model,
        source_optimizer,
        source_scheduler,
        CheckpointConfig(mode="async", interval=1),
        tmp_path,
        world,
    )
    assert source_manager.save(
        TrainerState(step=4, micro_step=8, total_tokens=80, total_samples=8)
    )
    source_manager.wait_for_pending_save()

    target_model = torch.nn.Linear(2, 1, bias=False)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.02)
    target_scheduler = torch.optim.lr_scheduler.StepLR(
        target_optimizer,
        step_size=2,
        gamma=0.1,
    )
    target_model(torch.ones(1, 2)).sum().backward()
    target_optimizer.step()
    target_optimizer.zero_grad(set_to_none=True)
    target_optimizer.state[target_model.weight]["exp_avg"].fill_(77.0)
    expected_exp_avg = target_optimizer.state[target_model.weight]["exp_avg"].clone()
    expected_scheduler_state = target_scheduler.state_dict()
    target_manager = CheckpointManager(
        target_model,
        target_optimizer,
        target_scheduler,
        CheckpointConfig(mode="disabled"),
        tmp_path / "unused",
        world,
    )

    state = target_manager.load(
        tmp_path / "checkpoint-4",
        load_optimizer=False,
        load_scheduler=False,
        load_progress=False,
    )

    assert torch.equal(target_model.weight, expected_weight)
    assert torch.equal(
        target_optimizer.state[target_model.weight]["exp_avg"], expected_exp_avg
    )
    assert target_scheduler.state_dict() == expected_scheduler_state
    assert state == TrainerState(step=0, micro_step=0)


def test_checkpoint_load_allows_new_world_size_without_dataloader(tmp_path) -> None:
    source_world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    source_model = torch.nn.Linear(2, 1, bias=False)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
    source_manager = CheckpointManager(
        source_model,
        source_optimizer,
        None,
        CheckpointConfig(mode="async", interval=1),
        tmp_path,
        source_world,
    )
    expected_weight = source_model.weight.detach().clone()
    assert source_manager.save(TrainerState(step=1, micro_step=1))
    source_manager.wait_for_pending_save()

    target_world = World(
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
    )
    target_model = torch.nn.Linear(2, 1, bias=False)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.01)
    target_manager = CheckpointManager(
        target_model,
        target_optimizer,
        None,
        CheckpointConfig(mode="disabled"),
        tmp_path / "unused",
        target_world,
    )

    state = target_manager.load(tmp_path / "checkpoint-1", dataloader=None)

    assert state.step == 1
    assert torch.equal(target_model.weight, expected_weight)


def test_checkpoint_load_rejects_new_world_size_with_rank_local_dataloader(
    tmp_path,
) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    source_manager = CheckpointManager(
        model,
        optimizer,
        None,
        CheckpointConfig(mode="async", interval=1),
        tmp_path,
        World(0, 0, 1, 1, torch.device("cpu")),
    )
    assert source_manager.save(TrainerState(step=1, micro_step=1))
    source_manager.wait_for_pending_save()

    target_manager = CheckpointManager(
        model,
        optimizer,
        None,
        CheckpointConfig(mode="disabled"),
        tmp_path / "unused",
        World(0, 0, 2, 2, torch.device("cpu")),
    )

    with pytest.raises(ValueError, match="skip_dataloader=true"):
        target_manager.load(tmp_path / "checkpoint-1", dataloader=Mock())


def test_trainer_wires_checkpoint_resume_skip_controls(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_dir = tmp_path / "source" / "checkpoint-4"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()
    trainer = BaseTrainer(
        TrainerConfig(
            output_dir=tmp_path / "target",
            max_steps=10,
            ckpt={
                "resume_dir": checkpoint_dir,
                "skip_optimizer": True,
                "skip_scheduler": True,
                "skip_dataloader": True,
                "skip_progress": True,
            },
        )
    )
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.01)
    trainer.scheduler = torch.optim.lr_scheduler.ConstantLR(
        trainer.optimizer,
        factor=1.0,
    )
    trainer.dataloader = object()
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    load = Mock(return_value=TrainerState(step=0, micro_step=0))
    monkeypatch.setattr(CheckpointManager, "load", load)

    trainer._setup_run()

    assert trainer.resume_checkpoint_dir == checkpoint_dir
    load.assert_called_once_with(
        checkpoint_dir,
        dataloader=None,
        load_optimizer=False,
        load_scheduler=False,
        load_progress=False,
    )
    assert trainer.ckpt_manager is not None
    assert trainer.ckpt_manager.scheduler is trainer.scheduler


def test_skipped_scheduler_restarts_over_remaining_steps() -> None:
    trainer = BaseTrainer(TrainerConfig(max_steps=10))
    trainer.optimizer = torch.optim.SGD(
        torch.nn.Linear(1, 1).parameters(),
        lr=0.25,
    )
    trainer.scheduler = Mock()
    trainer.ckpt_manager = Mock()
    setup_scheduler = Mock()
    trainer._setup_scheduler = setup_scheduler

    trainer._reset_scheduler(completed_steps=4)

    setup_scheduler.assert_called_once_with(total_steps=6)
    assert trainer.optimizer.param_groups[0]["lr"] == trainer.config.optim.lr


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

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

import wavelet.trainer.policy_export as policy_export_module
from wavelet.configs.config import RLConfig
from wavelet.contracts.schedule import (
    latest_exported_policy_step_at_or_before,
    next_exported_policy_step,
    policy_step_to_load,
    required_policy_step,
    retained_policy_snapshots,
)
from wavelet.orchestrator.scheduler import (
    _load_policy_and_update_scheduler,
    _VerifierChunkPublisher,
)
from wavelet.trainer.distributed import World
from wavelet.trainer.policy_export import PolicyExporter
from wavelet.transport.rollouts.filesystem import (
    STABLE_BATCH_MARKER,
    FileSystemPolicyReceiver,
)


class _PolicyReceiver:
    def __init__(self, steps: list[int]) -> None:
        self.steps = steps

    def available_steps(self) -> list[int]:
        return self.steps


class _PolicyExporter(PolicyExporter):
    pass


@pytest.mark.parametrize("interval", [1, 2, 4])
def test_export_retains_policy_selected_before_request_drain(tmp_path, interval):
    config = RLConfig(
        output_dir=tmp_path,
        orchestrator={"max_async_level": 9, "max_off_policy_steps": 8},
        policy_transfer={"keep_last": 2, "export_every_steps": interval},
    )
    trainer = SimpleNamespace(config=config, world=Mock(is_main=True))
    exporter = _PolicyExporter(trainer)
    exporter._barrier = Mock()
    receiver = FileSystemPolicyReceiver(tmp_path, config.policy_transfer)
    receiver.policy_dir.mkdir(parents=True)
    # Select the oldest allowed snapshot before draining. The trainer can finish
    # the previous batch and publish another export while the loader is paused.
    selected_step = 8
    for step in range(0, selected_step + 8 + 2, interval):
        temporary = receiver.policy_dir / f"pending-{step}"
        temporary.mkdir()
        destination = receiver.policy_dir / f"step-{step:06d}"
        exporter._publish_export_directory(temporary, destination, export_step=step)
    assert selected_step in receiver.available_steps()
    assert (receiver.policy_dir / "step-000008" / STABLE_BATCH_MARKER).exists()
    assert len(receiver.available_steps()) <= retained_policy_snapshots(config)


def test_retention_preserves_larger_explicit_limit():
    config = RLConfig(policy_transfer={"keep_last": 20})
    assert retained_policy_snapshots(config) == 20


def _config() -> RLConfig:
    return RLConfig(orchestrator={"max_async_level": 1, "max_off_policy_steps": 8})


def test_required_policy_step_uses_stricter_async_window() -> None:
    config = _config()

    assert required_policy_step(config, 0) == 0
    assert required_policy_step(config, 1) == 1
    assert required_policy_step(config, 2) == 2


def test_required_policy_step_uses_off_policy_window_when_stricter() -> None:
    config = RLConfig(orchestrator={"max_async_level": 8, "max_off_policy_steps": 2})

    assert required_policy_step(config, 0) == 0
    assert required_policy_step(config, 2) == 0
    assert required_policy_step(config, 3) == 1


def test_zero_off_policy_window_requires_current_policy() -> None:
    config = RLConfig(orchestrator={"max_async_level": 4, "max_off_policy_steps": 0})

    assert required_policy_step(config, 0) == 0
    assert required_policy_step(config, 1) == 1
    assert required_policy_step(config, 2) == 2


def test_policy_selection_does_not_wait_for_current_rollout_step() -> None:
    config = RLConfig(orchestrator={"max_async_level": 2, "max_off_policy_steps": 8})
    policy_step = policy_step_to_load(
        config,
        _PolicyReceiver([0, 1]),  # type: ignore[arg-type]
        rollout_step=2,
        loaded_policy_step=0,
    )

    assert policy_step == 1


def test_policy_selection_loads_newest_available_policy() -> None:
    policy_step = policy_step_to_load(
        _config(),
        _PolicyReceiver([0, 1, 2, 3]),  # type: ignore[arg-type]
        rollout_step=3,
        loaded_policy_step=1,
    )

    assert policy_step == 3


def test_policy_selection_does_not_load_policy_newer_than_rollout_step() -> None:
    policy_step = policy_step_to_load(
        _config(),
        _PolicyReceiver([0, 1, 2, 3]),  # type: ignore[arg-type]
        rollout_step=2,
        loaded_policy_step=0,
    )

    assert policy_step == 2


def test_policy_selection_reuses_loaded_policy_inside_async_window() -> None:
    config = RLConfig(orchestrator={"max_async_level": 2, "max_off_policy_steps": 8})
    policy_step = policy_step_to_load(
        config,
        _PolicyReceiver([0]),  # type: ignore[arg-type]
        rollout_step=1,
        loaded_policy_step=0,
    )

    assert policy_step is None


def test_policy_selection_waits_for_next_exported_step() -> None:
    config = RLConfig(
        orchestrator={"max_async_level": 2, "max_off_policy_steps": 8},
        policy_transfer={"export_every_steps": 2},
    )

    assert next_exported_policy_step(config, 1) == 2
    assert latest_exported_policy_step_at_or_before(config, 3) == 2
    assert (
        policy_step_to_load(
            config,
            _PolicyReceiver([0]),  # type: ignore[arg-type]
            rollout_step=3,
            loaded_policy_step=0,
        )
        == 2
    )


def test_policy_selection_uses_initial_export_when_allowed() -> None:
    config = RLConfig(
        orchestrator={"max_async_level": 0, "max_off_policy_steps": 0},
        policy_transfer={"export_initial": True, "export_every_steps": 4},
    )

    assert next_exported_policy_step(config, 0) == 0
    assert latest_exported_policy_step_at_or_before(config, 0) == 0
    assert (
        policy_step_to_load(
            config,
            _PolicyReceiver([0]),  # type: ignore[arg-type]
            rollout_step=0,
            loaded_policy_step=None,
        )
        == 0
    )


def test_checkpoint_resume_can_force_export_between_intervals() -> None:
    config = RLConfig(policy_transfer={"export_every_steps": 4})
    config = config.model_copy(
        update={
            "policy_transfer": config.policy_transfer.model_copy(
                update={"type": "nccl"}
            )
        }
    )
    trainer = SimpleNamespace(
        config=config,
        model=object(),
        tokenizer=object(),
        world=World(
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            device=torch.device("cpu"),
        ),
        output_dir=Path("outputs/run"),
    )
    exporter = _PolicyExporter(trainer)
    exporter._export_nccl_policy = Mock(return_value=Path("policy"))

    assert exporter.export_policy(step=7) is None
    assert exporter.export_policy(step=7, force=True) == Path("policy")
    exporter._export_nccl_policy.assert_called_once_with(7)


def test_async_policy_load_updates_scheduler_before_return(monkeypatch) -> None:
    calls: list[tuple[str, int | str | None]] = []

    async def fake_load_policy_async(
        config,
        inference_engine,
        policy_receiver,
        policy_step: int,
    ) -> int:
        calls.append(("load", policy_step))
        return 5

    class Scheduler:
        def begin_policy_update(self) -> None:
            calls.append(("begin", 0))

        async def drain_policy_update_requests(self) -> None:
            calls.append(("drain", 0))

        def set_policy_step(
            self,
            policy_step: int,
            *,
            model_name: str | None = None,
        ) -> None:
            calls.append(("set", policy_step))
            calls.append(("model", model_name))

        async def mark_policy_update(self) -> int:
            calls.append(("mark", 0))
            return 0

        def finish_policy_update(self) -> None:
            calls.append(("finish", 0))

    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler._load_policy_async",
        fake_load_policy_async,
    )

    scheduler = Scheduler()
    scheduler.begin_policy_update()
    loaded_step = asyncio.run(
        _load_policy_and_update_scheduler(
            _config(),
            inference_engine=type("Engine", (), {"policy_model_name": "policy"})(),
            policy_receiver=object(),  # type: ignore[arg-type]
            policy_step=4,
            scheduler=scheduler,
        )
    )

    assert loaded_step == 5
    assert calls == [
        ("begin", 0),
        ("load", 4),
        ("set", 5),
        ("model", "policy"),
        ("mark", 0),
        ("finish", 0),
    ]


def test_foreground_policy_refresh_marks_pending_work_stale(monkeypatch) -> None:
    calls: list[tuple[str, int | str | None]] = []

    async def fake_load_policy_async(*_args, **_kwargs) -> int:
        return 5

    class Scheduler:
        def begin_policy_update(self) -> None:
            calls.append(("begin", 0))

        async def drain_policy_update_requests(self) -> None:
            calls.append(("drain", 0))

        def set_policy_step(
            self,
            policy_step: int,
            *,
            model_name: str | None = None,
        ) -> None:
            calls.append(("set", policy_step))
            calls.append(("model", model_name))

        async def mark_policy_update(self) -> int:
            calls.append(("mark", 0))
            return 0

        def finish_policy_update(self) -> None:
            calls.append(("finish", 0))

    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler._load_policy_async",
        fake_load_policy_async,
    )
    context = object.__new__(_VerifierChunkPublisher)
    context.config = _config()
    context.inference_engine = type("Engine", (), {"policy_model_name": "policy"})()
    context.policy_receiver = object()
    context.scheduler = Scheduler()
    context.loaded_policy_step = 3
    context.state = None
    context.last_eval_steps = {}
    context.orchestrator = Mock()

    asyncio.run(context._load_now(5, optimizer_step=5))

    assert calls == [
        ("begin", 0),
        ("set", 5),
        ("model", "policy"),
        ("mark", 0),
        ("finish", 0),
    ]


def test_background_policy_refresh_closes_submission_gate_immediately(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Scheduler:
        def begin_policy_update(self) -> None:
            calls.append("begin")

        def finish_policy_update(self) -> None:
            calls.append("finish")

    async def fake_update(*_args, **_kwargs) -> int:
        calls.append("task")
        return 4

    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler._load_policy_and_update_scheduler",
        fake_update,
    )
    context = object.__new__(_VerifierChunkPublisher)
    context.config = _config()
    context.inference_engine = object()
    context.policy_receiver = object()
    context.scheduler = Scheduler()
    context.state = None

    async def run() -> None:
        context._start_background_load(4)
        assert calls == ["begin"]
        await context.pending_policy_update

    asyncio.run(run())

    assert calls == ["begin", "task"]


@pytest.mark.parametrize("lora", [None, {"rank": 4}])
def test_http_policy_update_does_not_wait_for_agent_episode(monkeypatch, lora):
    config = RLConfig(lora=lora)
    scheduler = Mock()
    scheduler.drain_policy_update_requests = AsyncMock()
    scheduler.mark_policy_update = AsyncMock()
    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler._load_policy_async", AsyncMock(return_value=1)
    )
    asyncio.run(_load_policy_and_update_scheduler(config, Mock(), Mock(), 1, scheduler))
    scheduler.drain_policy_update_requests.assert_not_awaited()
    scheduler.set_policy_step.assert_called_once()
    scheduler.finish_policy_update.assert_called_once()


def test_policy_watch_runs_while_agent_is_waiting():
    async def run():
        policy_loaded = asyncio.Event()
        config = RLConfig(transport={"poll_interval_seconds": 0.01})

        async def generate(**kwargs):
            await asyncio.wait_for(policy_loaded.wait(), timeout=1)
            return ["completed"]

        async def refresh(step):
            assert step == 1
            policy_loaded.set()

        context = object.__new__(_VerifierChunkPublisher)
        context.config = config
        context.scheduler = Mock(generate_batch=generate)
        context.prepare_policy = refresh
        assert await context._generate_batch(rollout_step=1) == ["completed"]

    asyncio.run(run())


def test_policy_watch_failure_cancels_pending_generation():
    async def run():
        cancelled = asyncio.Event()

        async def generate(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        context = object.__new__(_VerifierChunkPublisher)
        context.config = RLConfig(transport={"poll_interval_seconds": 0.01})
        context.scheduler = Mock(generate_batch=generate)
        context.prepare_policy = AsyncMock(side_effect=RuntimeError("transfer failed"))
        with pytest.raises(RuntimeError, match="transfer failed"):
            await context._generate_batch(rollout_step=1)
        assert cancelled.is_set()

    asyncio.run(run())


def test_shutdown_finishes_in_progress_policy_transfer(monkeypatch):
    async def run():
        release = asyncio.Event()
        context = object.__new__(_VerifierChunkPublisher)
        context.pending_policy_update = asyncio.create_task(release.wait())
        context.scheduler = Mock(aclose=AsyncMock())
        monkeypatch.setattr(
            "wavelet.orchestrator.scheduler._teardown_cached_verifier_envs", AsyncMock()
        )
        closing = asyncio.create_task(context.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert not context.pending_policy_update.cancelled()
        context.scheduler.aclose.assert_not_awaited()
        release.set()
        await closing
        context.scheduler.aclose.assert_awaited_once()

    asyncio.run(run())


def test_nccl_initial_policy_rendezvous_precedes_rank_gathers(tmp_path, monkeypatch):
    trainer = SimpleNamespace(world=Mock(is_main=True), model=object())
    exporter = _PolicyExporter(trainer)
    calls = []
    exporter._wait_for_nccl_ready = lambda path: calls.append("ready")
    broadcaster = Mock()
    exporter._nccl_broadcaster = lambda: broadcaster
    exporter._barrier = lambda: calls.append("barrier")
    monkeypatch.setattr(
        policy_export_module,
        "broadcast_model",
        lambda broadcaster, model: calls.append("broadcast"),
    )
    exporter._broadcast_nccl_export(tmp_path, export_step=0)
    assert calls == ["ready", "barrier", "broadcast"]

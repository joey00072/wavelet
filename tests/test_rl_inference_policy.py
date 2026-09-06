from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.schedule import (
    latest_exported_policy_step_at_or_before,
    next_exported_policy_step,
    policy_step_to_load,
    required_policy_step,
)
from wavelet.orchestrator.scheduler import (
    _load_policy_and_update_scheduler,
    _VerifierChunkPublisher,
)
from wavelet.trainer.distributed import World
from wavelet.transport.policy import PolicyExportMixin


class _PolicyReceiver:
    def __init__(self, steps: list[int]) -> None:
        self.steps = steps

    def available_steps(self) -> list[int]:
        return self.steps


class _PolicyExporter(PolicyExportMixin):
    pass


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
    exporter = _PolicyExporter()
    exporter.config = config
    exporter.model = object()
    exporter.tokenizer = object()
    exporter.world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    exporter.output_dir = Path("outputs/run")
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
        ("drain", 0),
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
        ("drain", 0),
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

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import wavelet.orchestrator.envs as verifier_envs
from wavelet.configs.rl_config import RLConfig, RLEvalEnvConfig
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.runtime import (
    _load_policy_for_step,
    _pipelined_rollouts,
    _policy_step_for_trainer_step,
    _publish_rollout_timed,
)
from wavelet.orchestrator.scheduler import (
    _MAX_GROUP_RETRIES,
    VerifierRolloutScheduler,
    _PendingVerifierRequest,
    _VerifierGroupState,
)
from wavelet.transport.queue import FileSystemRolloutSender


def _scheduler(
    config: RLConfig,
    *,
    policy_step: int | None,
    rollout_step: int | None = None,
    rollout_count: int = 1,
    requires_group_scoring: bool = False,
) -> VerifierRolloutScheduler:
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {}
    scheduler.ready_groups = []
    scheduler.ready_group_off_policy_steps = []
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_step = policy_step
    scheduler.rollout_step = rollout_step
    scheduler.rollout_count = rollout_count
    scheduler.requires_group_scoring = requires_group_scoring
    scheduler.env_name = "env"
    return scheduler


async def _finished_task(outputs: list[dict[str, Any]]) -> asyncio.Task:
    async def rollout() -> list[dict[str, Any]]:
        return outputs

    task = asyncio.create_task(rollout())
    await task
    return task


def _consume(
    scheduler: VerifierRolloutScheduler,
    outputs: list[dict[str, Any]],
    *,
    request: _PendingVerifierRequest,
) -> tuple[int, int, int]:
    async def run() -> tuple[int, int, int]:
        task = await _finished_task(outputs)
        scheduler.pending[task] = request
        scheduler.pending_clients[task] = request.client_index
        return scheduler._consume_completed_task(
            task,
            target_groups=1,
            outputs=[],
            accepted_groups=0,
        )

    return asyncio.run(run())


# ── freshness window matches the trainer contract ─────────────────────────────


def test_ready_group_outside_trainer_window_is_dropped_for_next_step() -> None:
    # max_async_level=2 lets the loaded policy lag by one step, so groups that
    # were only one step behind the LOADED policy can be two steps behind the
    # rollout step; the trainer rejects those with required_policy_step.
    config = RLConfig(orchestrator={"max_async_level": 2, "max_off_policy_steps": 1})
    step = 10
    scheduler = _scheduler(config, policy_step=step - 1, rollout_step=step)
    scheduler.ready_groups = [
        [{"_wavelet_policy_step": step - 2, "reward": 1.0}],
        [{"_wavelet_policy_step": step - 1, "reward": 1.0}],
    ]
    scheduler.ready_group_off_policy_steps = [1, 0]

    dropped = scheduler._prune_ready_groups(advance_age=False)

    assert dropped == 1
    assert [group[0]["_wavelet_policy_step"] for group in scheduler.ready_groups] == [
        step - 1
    ]
    assert scheduler.ready_group_off_policy_steps == [0]


def test_completed_group_outside_trainer_window_is_cancelled() -> None:
    config = RLConfig(orchestrator={"max_async_level": 2, "max_off_policy_steps": 1})
    step = 10
    scheduler = _scheduler(config, policy_step=step - 1, rollout_step=step)
    scheduler.groups = {
        0: _VerifierGroupState(example={"example_id": 0}, rollouts_to_schedule=0)
    }

    result = _consume(
        scheduler,
        [{"example_id": 0, "reward": 1.0, "error": None}],
        request=_PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
            off_policy_steps=1,
            policy_step=step - 2,
        ),
    )

    assert result == (0, 0, 0)
    assert scheduler.groups == {}
    assert scheduler.cancelled_rollouts_count == 1


def test_group_within_trainer_window_is_admitted() -> None:
    config = RLConfig(
        orchestrator={
            "max_async_level": 2,
            "max_off_policy_steps": 1,
            "filter_zero_advantage": False,
        }
    )
    step = 10
    scheduler = _scheduler(config, policy_step=step - 1, rollout_step=step)
    scheduler.groups = {
        0: _VerifierGroupState(example={"example_id": 0}, rollouts_to_schedule=0)
    }

    result = _consume(
        scheduler,
        [
            {
                "example_id": 0,
                "reward": 1.0,
                "error": None,
                "trajectory": [],
                "prompt": [],
                "completion": [],
            }
        ],
        request=_PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
            policy_step=step - 1,
        ),
    )

    assert result[1] == 1
    assert scheduler.cancelled_rollouts_count == 0


# ── group scoring and retry budget ────────────────────────────────────────────


def test_group_scoring_partial_failure_discards_partial_outputs() -> None:
    scheduler = _scheduler(
        RLConfig(),
        policy_step=0,
        rollout_step=0,
        rollout_count=2,
        requires_group_scoring=True,
    )
    group = _VerifierGroupState(example={"example_id": 0}, rollouts_to_schedule=0)
    scheduler.groups = {0: group}

    result = _consume(
        scheduler,
        [{"example_id": 0, "reward": 1.0, "error": None}],
        request=_PendingVerifierRequest(
            group_id=0, client_index=0, rollout_count=2, policy_step=0
        ),
    )

    assert result == (0, 0, 0)
    assert group.completed_outputs == []
    assert group.rollouts_to_schedule == 2
    assert group.failed_rollouts == 1
    assert 0 in scheduler.groups


def test_persistently_failing_group_is_dropped_after_retry_budget() -> None:
    scheduler = _scheduler(RLConfig(), policy_step=0, rollout_step=0)
    group = _VerifierGroupState(example={"example_id": 0}, rollouts_to_schedule=0)
    scheduler.groups = {0: group}
    request = _PendingVerifierRequest(
        group_id=0, client_index=0, rollout_count=1, policy_step=0
    )

    for attempt in range(_MAX_GROUP_RETRIES):
        assert _consume(scheduler, [], request=request) == (0, 0, 0)
        assert 0 in scheduler.groups
        assert group.rollouts_to_schedule == attempt + 1

    assert _consume(scheduler, [], request=request) == (0, 1, 1)
    assert scheduler.groups == {}


# ── integrated launcher ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("max_async_level", "max_off_policy_steps", "expected"),
    [(0, 0, False), (1, 1, False), (2, 0, False), (2, 1, True)],
)
def test_pipelined_rollouts_require_a_lag_one_contract(
    max_async_level: int, max_off_policy_steps: int, expected: bool
) -> None:
    config = RLConfig(
        orchestrator={
            "max_async_level": max_async_level,
            "max_off_policy_steps": max_off_policy_steps,
        }
    )

    assert _pipelined_rollouts(config) is expected


class _FakePolicyReceiver:
    def __init__(self, steps: list[int]) -> None:
        self.steps = steps
        self.waited: list[int] = []

    def available_steps(self) -> set[int]:
        return set(self.steps)

    def wait_for_step(self, step: int) -> SimpleNamespace:
        self.waited.append(step)
        return SimpleNamespace(step=step, step_dir=Path(f"policies/step-{step}"))


class _FakeEngine:
    def __init__(self) -> None:
        self.policy_step: int | None = None
        self.loaded: list[int] = []

    def load_policy(self, step_dir: Path, *, step: int) -> None:
        del step_dir
        self.loaded.append(step)
        self.policy_step = step


def test_integrated_policy_load_waits_only_for_exported_steps() -> None:
    config = RLConfig(policy_transfer={"export_every_steps": 2})
    receiver = _FakePolicyReceiver([0])
    engine = _FakeEngine()

    assert _policy_step_for_trainer_step(config, receiver, step=1) == 0
    _load_policy_for_step(config, receiver, engine, step=1)
    assert receiver.waited == [0]
    assert engine.loaded == [0]

    # Already loaded: no second wait.
    _load_policy_for_step(config, receiver, engine, step=1)
    assert receiver.waited == [0]

    receiver.steps.append(2)
    _load_policy_for_step(config, receiver, engine, step=3)
    assert receiver.waited == [0, 2]
    assert engine.policy_step == 2


def test_integrated_policy_load_prefers_forced_exports() -> None:
    config = RLConfig(policy_transfer={"export_every_steps": 4})
    receiver = _FakePolicyReceiver([0, 3])
    engine = _FakeEngine()

    _load_policy_for_step(config, receiver, engine, step=3)

    assert receiver.waited == [3]


def test_integrated_policy_load_skips_when_nothing_was_exported() -> None:
    config = RLConfig(policy_transfer={"export_initial": False})
    receiver = _FakePolicyReceiver([])
    engine = _FakeEngine()

    assert _load_policy_for_step(config, receiver, engine, step=0) == 0.0
    assert receiver.waited == []


def test_integrated_publish_labels_unloaded_engine_as_policy_zero() -> None:
    published: dict[str, object] = {}

    def publish(**kwargs: object) -> object:
        published.update(kwargs)
        return object()

    orchestrator = type("Orchestrator", (), {"publish": staticmethod(publish)})()
    engine = type("InferenceEngine", (), {"policy_step": None})()

    _publish_rollout_timed(orchestrator, step=0, inference_engine=engine)  # type: ignore[arg-type]

    assert published["policy_step"] == 0


def test_integrated_publish_reuses_existing_stable_batch(tmp_path: Path) -> None:
    config = RLConfig(output_dir=tmp_path / "run")
    orchestrator = RLOrchestrator(config)
    source = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps({"prompt": "p", "advantage": 1.0}) + "\n")
    sender = FileSystemRolloutSender(config.output_dir, config.transport)
    existing = sender.publish(source, step=3, optimizer_step=3, policy_step=3, rows=1)

    def fail(**kwargs: object) -> Path:
        raise AssertionError("resume must not regenerate a stable batch")

    orchestrator.materialize = fail  # type: ignore[method-assign]

    batch = orchestrator.publish(step=3, policy_step=3)

    assert batch.step == 3
    assert batch.path == existing.path


# ── evals use the served policy model ─────────────────────────────────────────


def test_eval_requests_route_to_the_served_policy_model(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class _Env:
        def get_eval_dataset(self, n: int) -> SimpleNamespace:
            del n
            return SimpleNamespace(to_list=list)

    async def run_examples(vf, env, examples, *, model, **kwargs):  # type: ignore[no-untyped-def]
        del vf, env, examples
        captured["model"] = model
        captured["max_inflight_rollouts"] = kwargs["max_inflight_rollouts"]
        return []

    monkeypatch.setattr(verifier_envs, "_load_verifiers", lambda feature: object())
    monkeypatch.setattr(
        verifier_envs, "_load_cached_env", lambda *args, **kwargs: (_Env(), False)
    )
    monkeypatch.setattr(
        verifier_envs, "_verifier_clients", lambda *args, **kwargs: [object()]
    )
    monkeypatch.setattr(verifier_envs, "_run_eval_examples", run_examples)
    monkeypatch.setattr(verifier_envs, "_scale_verifier_executors", lambda n: None)

    config = RLConfig(
        output_dir=tmp_path / "run",
        lora={"rank": 4},
        eval={"env": [{"id": "math-env"}], "max_inflight_rollouts": 7},
    )
    orchestrator = SimpleNamespace(config=config)
    engine = SimpleNamespace(policy_model_name="policy-adapter")

    metrics = asyncio.run(
        verifier_envs._evaluate_env_async(
            orchestrator,  # type: ignore[arg-type]
            RLEvalEnvConfig(id="math-env"),
            step=4,
            policy_step=4,
            inference_engine=engine,
        )
    )

    assert captured["model"] == "policy-adapter"
    assert captured["max_inflight_rollouts"] == 7
    assert metrics["progress/policy_step"] == 4.0

    asyncio.run(
        verifier_envs._evaluate_env_async(
            orchestrator,  # type: ignore[arg-type]
            RLEvalEnvConfig(id="math-env"),
            step=4,
            policy_step=4,
        )
    )
    assert captured["model"] == config.model.name

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from wavelet.configs.rl_config import RLConfig, RLEvalConfig
from wavelet.orchestrator.eval_utils import compute_eval_policy_step, pass_at_k
from wavelet.orchestrator.schedule import select_due_eval_envs, target_steps
from wavelet.orchestrator.scheduler import (
    _final_eval_policy_step,
    _initial_eval_steps,
    _run_evals,
    _run_evals_async,
    _run_final_evals,
    _select_final_eval_envs,
    _VerifierChunkPublisher,
)
from wavelet.orchestrator.verifiers import _eval_metrics, _run_eval_examples


def test_compute_eval_policy_step_runs_base_and_intervals() -> None:
    assert (
        compute_eval_policy_step(
            policy_step=0,
            last_eval_step=-1,
            interval=4,
            eval_base_model=True,
        )
        == 0
    )
    assert (
        compute_eval_policy_step(
            policy_step=4,
            last_eval_step=0,
            interval=4,
        )
        == 4
    )
    assert (
        compute_eval_policy_step(
            policy_step=5,
            last_eval_step=4,
            interval=4,
        )
        is None
    )
    assert (
        compute_eval_policy_step(
            policy_step=9,
            last_eval_step=4,
            interval=4,
        )
        == 8
    )


def test_compute_eval_policy_step_can_skip_base_model() -> None:
    assert (
        compute_eval_policy_step(
            policy_step=0,
            last_eval_step=-1,
            interval=4,
            eval_base_model=False,
        )
        is None
    )


def test_every_scheduler_starts_with_base_eval_due() -> None:
    config = RLConfig(
        eval={
            "eval_base_model": True,
            "env": [{"id": "aime", "name": "aime2024"}],
        }
    )

    last_eval_steps = _initial_eval_steps(config)

    assert last_eval_steps == {"aime2024": -1}
    assert [
        env.resolved_name
        for env in select_due_eval_envs(
            config,
            policy_step=0,
            last_eval_steps=last_eval_steps,
        )
    ] == ["aime2024"]


def test_resumed_scheduler_restores_completed_evals(tmp_path: Path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        eval={
            "interval": 100,
            "eval_base_model": True,
            "env": [{"id": "aime", "name": "aime2024"}],
        },
    )
    (tmp_path / "eval_metrics.jsonl").write_text(
        '{"progress/policy_step": 500, "eval/aime2024/avg@8": 0.5}\n',
        encoding="utf-8",
    )

    last_eval_steps = _initial_eval_steps(config, start_step=553)

    assert last_eval_steps == {"aime2024": 500}
    assert (
        select_due_eval_envs(
            config,
            policy_step=553,
            last_eval_steps=last_eval_steps,
        )
        == []
    )


def test_resumed_scheduler_does_not_assume_missing_eval_completed(
    tmp_path: Path,
) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        eval={"env": [{"id": "aime", "name": "aime2024"}]},
    )

    assert _initial_eval_steps(config, start_step=553) == {"aime2024": -1}


def test_pass_at_k_for_binary_rewards() -> None:
    metrics = pass_at_k([0.0, 1.0, 0.0, 1.0])

    assert metrics["pass@1"] == pytest.approx(0.5)
    assert metrics["pass@2"] == pytest.approx(5 / 6)
    assert metrics["pass@4"] == pytest.approx(1.0)
    assert metrics["pass^1"] == pytest.approx(0.5)
    assert metrics["pass^2"] == pytest.approx(1 / 6)
    assert metrics["pass^4"] == pytest.approx(0.0)


def test_eval_config_inherits_group_defaults() -> None:
    config = RLEvalConfig(
        interval=8,
        num_examples=16,
        rollouts_per_example=4,
        env=[{"id": "alphabet-sort", "name": "alphabet"}],
    )

    env = config.env[0]
    assert env.interval == 8
    assert env.num_examples == 16
    assert env.rollouts_per_example == 4
    assert env.resolved_name == "alphabet"
    assert config.max_inflight_rollouts == 64


def test_select_due_eval_envs_updates_each_env_independently() -> None:
    config = RLConfig(
        eval={
            "eval_base_model": False,
            "env": [
                {"id": "alphabet-sort", "name": "alphabet", "interval": 4},
                {"id": "reverse-text", "name": "reverse", "interval": 8},
            ],
        },
    )
    last_eval_steps = {"alphabet": -1, "reverse": -1}

    assert (
        select_due_eval_envs(
            config,
            policy_step=0,
            last_eval_steps=last_eval_steps,
        )
        == []
    )

    due = select_due_eval_envs(
        config,
        policy_step=8,
        last_eval_steps=last_eval_steps,
    )

    assert [env.resolved_name for env in due] == ["alphabet", "reverse"]
    assert last_eval_steps == {"alphabet": 8, "reverse": 8}


def test_due_eval_tracks_actual_policy_when_interval_was_skipped() -> None:
    config = RLConfig(
        eval={
            "eval_base_model": False,
            "env": [{"id": "aime", "name": "aime2024", "interval": 100}],
        },
    )
    last_eval_steps = {"aime2024": 0}

    due = select_due_eval_envs(
        config,
        policy_step=101,
        last_eval_steps=last_eval_steps,
    )

    assert [env.resolved_name for env in due] == ["aime2024"]
    assert last_eval_steps == {"aime2024": 101}


def test_final_eval_policy_step_uses_last_exported_step() -> None:
    config = RLConfig()
    config.policy_transfer.export_every_steps = 4

    assert _final_eval_policy_step(config, 100) == 100
    assert _final_eval_policy_step(config, 99) == 96


def test_final_eval_skips_policy_already_evaluated_at_interval() -> None:
    config = RLConfig(
        eval={
            "interval": 100,
            "env": [{"id": "aime", "name": "aime2024"}],
        }
    )
    last_eval_steps = {"aime2024": 100}

    envs = _select_final_eval_envs(
        config,
        policy_step=100,
        last_eval_steps=last_eval_steps,
    )

    assert envs == []
    assert last_eval_steps == {"aime2024": 100}


def test_final_eval_runs_policy_not_seen_at_an_interval() -> None:
    config = RLConfig(
        eval={
            "interval": 100,
            "env": [{"id": "aime", "name": "aime2024"}],
        }
    )
    last_eval_steps = {"aime2024": 100}

    envs = _select_final_eval_envs(
        config,
        policy_step=137,
        last_eval_steps=last_eval_steps,
    )

    assert [env.resolved_name for env in envs] == ["aime2024"]
    assert last_eval_steps == {"aime2024": 137}


def test_sync_final_eval_does_not_wake_for_duplicate_policy(
    monkeypatch, tmp_path: Path
) -> None:
    config = RLConfig.model_validate(
        {
            "max_steps": 100,
            "output_dir": tmp_path,
            "eval": {"env": [{"id": "aime", "name": "aime2024"}]},
        }
    )
    inference_engine = Mock()
    wake = Mock()
    run_evals = Mock()
    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler._wake_for_colocated_sleep", wake
    )
    monkeypatch.setattr("wavelet.orchestrator.scheduler._run_evals", run_evals)

    loaded_step = _run_final_evals(
        config,
        Mock(),
        inference_engine,
        Mock(),
        target_step=100,
        loaded_policy_step=100,
        last_eval_steps={"aime2024": 100},
    )

    assert loaded_step == 100
    wake.assert_not_called()
    run_evals.assert_not_called()


def test_async_final_eval_cancels_unused_scheduler_work(
    monkeypatch, tmp_path: Path
) -> None:
    config = RLConfig.model_validate(
        {
            "max_steps": 101,
            "output_dir": tmp_path,
            "eval": {"env": [{"id": "aime", "name": "aime2024"}]},
        }
    )
    scheduler = Mock()
    scheduler.aclose = AsyncMock()
    run_evals = AsyncMock()
    monkeypatch.setattr("wavelet.orchestrator.scheduler._run_evals_async", run_evals)
    context = object.__new__(_VerifierChunkPublisher)
    context.config = config
    context.orchestrator = Mock()
    context.inference_engine = Mock()
    context.policy_receiver = Mock()
    context.scheduler = scheduler
    context.last_eval_steps = {"aime2024": 100}
    context.loaded_policy_step = 101

    asyncio.run(context.run_final_evals(101))

    scheduler.aclose.assert_awaited_once_with()
    run_evals.assert_awaited_once()
    assert run_evals.await_args.kwargs["policy_step"] == 101


def test_eval_only_base_model_does_not_wait_for_policy_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    config = RLConfig.model_validate(
        {
            "max_steps": 0,
            "output_dir": tmp_path,
            "orchestrator": {
                "custom_rollout_function": (
                    "wavelet.orchestrator.verifiers:generate_rollouts"
                )
            },
            "eval": {"env": [{"id": "test-eval"}]},
            "policy_transfer": {"export_initial": True},
        }
    )
    receiver = Mock()
    inference_engine = Mock()
    monkeypatch.setattr("wavelet.orchestrator.scheduler._run_evals", Mock())

    loaded_step = _run_final_evals(
        config,
        Mock(),
        inference_engine,
        receiver,
        target_step=0,
        loaded_policy_step=None,
    )

    assert loaded_step == 0
    receiver.wait_for_step.assert_not_called()
    inference_engine.load_policy.assert_not_called()


def test_rl_config_allows_zero_step_eval_only_runs() -> None:
    config = RLConfig.model_validate(
        {"max_steps": 0, "policy_transfer": {"export_initial": True}}
    )

    assert target_steps(config) == 0
    assert _final_eval_policy_step(config, 0) == 0


def test_run_evals_publishes_metrics_to_monitor(monkeypatch) -> None:
    config = RLConfig.model_validate(
        {
            "orchestrator": {
                "custom_rollout_function": (
                    "wavelet.orchestrator.verifiers:generate_rollouts"
                )
            }
        }
    )
    env = SimpleNamespace(resolved_name="aime")
    metrics = {"eval/aime/avg@8": 0.25}
    evaluate = Mock(return_value=metrics)
    log_metrics = Mock()
    monkeypatch.setattr("wavelet.orchestrator.scheduler.evaluate_env", evaluate)
    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler.log_eval_metrics",
        log_metrics,
    )

    _run_evals(
        config,
        Mock(),
        policy_step=96,
        rollout_step=100,
        envs=[env],
    )

    log_metrics.assert_called_once_with(
        config,
        metrics,
        step=100,
        policy_step=96,
    )


def test_run_evals_async_publishes_metrics_to_monitor(monkeypatch) -> None:
    config = RLConfig.model_validate(
        {
            "orchestrator": {
                "custom_rollout_function": (
                    "wavelet.orchestrator.verifiers:generate_rollouts"
                )
            }
        }
    )
    env = SimpleNamespace(resolved_name="aime")
    metrics = {"eval/aime/pass@8": 0.5}
    evaluate = AsyncMock(return_value=metrics)
    log_metrics = Mock()
    monkeypatch.setattr("wavelet.orchestrator.scheduler.evaluate_env_async", evaluate)
    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler.log_eval_metrics",
        log_metrics,
    )

    asyncio.run(
        _run_evals_async(
            config,
            Mock(),
            policy_step=96,
            rollout_step=100,
            envs=[env],
        )
    )

    log_metrics.assert_called_once_with(
        config,
        metrics,
        step=100,
        policy_step=96,
    )


def test_background_eval_does_not_block_rollout_publishing(monkeypatch) -> None:
    async def run() -> None:
        config = RLConfig.model_validate(
            {
                "launcher": {"mode": "process"},
                "orchestrator": {
                    "custom_rollout_function": (
                        "wavelet.orchestrator.verifiers:generate_rollouts"
                    ),
                    "examples_per_step": 1,
                    "max_async_level": 1,
                },
                "eval": {
                    "background": True,
                    "env": [{"id": "aime", "interval": 1}],
                },
            }
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def evaluate(*_args, **_kwargs) -> None:
            started.set()
            await release.wait()

        monkeypatch.setattr(
            "wavelet.orchestrator.scheduler._run_evals_async",
            evaluate,
        )
        context = object.__new__(_VerifierChunkPublisher)
        context.config = config
        context.orchestrator = Mock()
        context.inference_engine = Mock()
        context.policy_receiver = Mock()
        context.scheduler = Mock()
        context.state = None
        context.loaded_policy_step = 1
        context.last_eval_steps = {"aime": -1}
        context.pending_eval_task = None
        context.pending_eval_policy_step = None

        await context._record_loaded_policy(1)
        await started.wait()

        assert context.pending_eval_task is not None
        assert not context.pending_eval_task.done()
        release.set()
        await context._settle_pending_eval(for_policy_update=False)

    asyncio.run(run())


def test_background_eval_is_cancelled_before_new_policy(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        config = RLConfig.model_validate(
            {
                "output_dir": tmp_path,
                "launcher": {"mode": "process"},
                "orchestrator": {
                    "custom_rollout_function": (
                        "wavelet.orchestrator.verifiers:generate_rollouts"
                    ),
                    "examples_per_step": 1,
                    "max_async_level": 1,
                },
                "eval": {"background": True, "cancel_on_new_policy": True},
            }
        )
        cancelled = asyncio.Event()

        async def evaluate() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        context = object.__new__(_VerifierChunkPublisher)
        context.config = config
        context.pending_eval_task = asyncio.create_task(evaluate())
        context.pending_eval_policy_step = 3

        await asyncio.sleep(0)
        await context._settle_pending_eval(for_policy_update=True)

        assert cancelled.is_set()
        assert context.pending_eval_task is None
        events = (tmp_path / "events" / "queue.jsonl").read_text(encoding="utf-8")
        assert "eval_cancelled_for_policy_update" in events

    asyncio.run(run())


def test_background_eval_can_delay_policy_update_instead_of_cancelling(
    tmp_path,
) -> None:
    async def run() -> None:
        config = RLConfig.model_validate(
            {
                "output_dir": tmp_path,
                "launcher": {"mode": "process"},
                "orchestrator": {
                    "custom_rollout_function": (
                        "wavelet.orchestrator.verifiers:generate_rollouts"
                    ),
                    "examples_per_step": 1,
                    "max_async_level": 1,
                },
                "eval": {"background": True, "cancel_on_new_policy": False},
            }
        )
        release = asyncio.Event()
        context = object.__new__(_VerifierChunkPublisher)
        context.config = config
        context.pending_eval_task = asyncio.create_task(release.wait())
        context.pending_eval_policy_step = 3

        waiter = asyncio.create_task(
            context._settle_pending_eval(for_policy_update=True)
        )
        await asyncio.sleep(0)
        assert not waiter.done()
        release.set()
        await waiter

        assert context.pending_eval_task is None

    asyncio.run(run())


def test_eval_metrics_include_avg_and_pass_at_k() -> None:
    metrics = _eval_metrics(
        "alphabet",
        [
            {
                "example_id": "a",
                "reward": 0.0,
                "completion": [{"role": "assistant", "content": "x"}],
                "is_truncated": False,
            },
            {
                "example_id": "a",
                "reward": 1.0,
                "completion": [{"role": "assistant", "content": "y"}],
                "is_truncated": False,
            },
        ],
        total_rollouts=2,
        elapsed_seconds=1.5,
        rollouts_per_example=2,
    )

    assert metrics["eval/alphabet/avg@2"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/pass@1"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/pass@2"] == pytest.approx(1.0)
    assert metrics["eval/alphabet/pass^1"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/pass^2"] == pytest.approx(0.0)
    assert metrics["eval/alphabet/failed_rollouts"] == pytest.approx(0.0)


def test_eval_metrics_treat_missing_reward_as_failed_rollout() -> None:
    metrics = _eval_metrics(
        "alphabet",
        [
            {
                "example_id": "a",
                "reward": 1.0,
                "completion": [{"role": "assistant", "content": "x"}],
                "is_truncated": False,
            },
            {
                "example_id": "a",
                "error": "timeout",
                "completion": [],
                "is_truncated": True,
            },
        ],
        total_rollouts=2,
        elapsed_seconds=1.5,
        rollouts_per_example=2,
    )

    assert metrics["eval/alphabet/avg@2"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/pass@1"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/pass@2"] == pytest.approx(1.0)
    assert metrics["eval/alphabet/pass^1"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/pass^2"] == pytest.approx(0.0)
    assert metrics["eval/alphabet/effective/avg@2"] == pytest.approx(1.0)
    assert metrics["eval/alphabet/effective/pass@1"] == pytest.approx(1.0)
    assert metrics["eval/alphabet/effective/pass^1"] == pytest.approx(1.0)
    assert metrics["eval/alphabet/failed_rollouts"] == pytest.approx(0.5)


def test_eval_rollouts_preserve_failed_attempts_and_example_identity() -> None:
    vf = SimpleNamespace(RolloutInput=lambda **kwargs: kwargs)
    env = SimpleNamespace(
        run_rollout=AsyncMock(
            side_effect=[
                {"reward": 1.0, "completion": ["ok"]},
                RuntimeError("generation failed"),
            ]
        )
    )

    outputs = asyncio.run(
        _run_eval_examples(
            vf,
            env,
            [{"question": "q"}],
            clients=[object()],
            model="model",
            sampling_args={},
            rollouts_per_example=2,
            max_retries=0,
        )
    )

    assert outputs[0]["example_id"] == "0"
    assert outputs[1] == {
        "example_id": "0",
        "error": "generation failed",
        "completion": [],
    }


def test_eval_rollouts_treat_verifier_caught_errors_as_failed_attempts() -> None:
    vf = SimpleNamespace(RolloutInput=lambda **kwargs: kwargs)
    env = SimpleNamespace(
        run_rollout=AsyncMock(
            side_effect=[
                {"reward": 1.0, "completion": ["ok"], "error": None},
                {"reward": 0.0, "completion": [], "error": RuntimeError("boom")},
            ]
        )
    )

    outputs = asyncio.run(
        _run_eval_examples(
            vf,
            env,
            [{"question": "q", "example_id": "a"}],
            clients=[object()],
            model="model",
            sampling_args={},
            rollouts_per_example=2,
            max_retries=0,
        )
    )
    metrics = _eval_metrics(
        "alphabet",
        outputs,
        total_rollouts=2,
        elapsed_seconds=1.0,
        rollouts_per_example=2,
    )

    assert outputs[0]["reward"] == 1.0
    assert "reward" not in outputs[1]
    assert outputs[1]["error"] == "boom"
    assert metrics["eval/alphabet/failed_rollouts"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/avg@2"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/pass@1"] == pytest.approx(0.5)
    assert metrics["eval/alphabet/effective/avg@2"] == pytest.approx(1.0)
    assert metrics["eval/alphabet/effective/pass@1"] == pytest.approx(1.0)


def test_eval_rollouts_offset_fixed_seed_per_rollout() -> None:
    vf = SimpleNamespace(RolloutInput=lambda **kwargs: kwargs)
    run_rollout = AsyncMock(return_value={"reward": 1.0, "completion": ["ok"]})
    env = SimpleNamespace(run_rollout=run_rollout)
    sampling_args = {"seed": 7, "extra_body": {"cache_salt": "3"}}

    asyncio.run(
        _run_eval_examples(
            vf,
            env,
            [{"question": "q"}, {"question": "r"}],
            clients=[object()],
            model="model",
            sampling_args=sampling_args,
            rollouts_per_example=3,
            max_retries=0,
        )
    )

    seeds = [
        call.kwargs["sampling_args"]["seed"] for call in run_rollout.await_args_list
    ]
    assert seeds == [7, 8, 9, 7, 8, 9]
    assert all(
        call.kwargs["sampling_args"]["extra_body"] == {"cache_salt": "3"}
        for call in run_rollout.await_args_list
    )
    assert sampling_args == {"seed": 7, "extra_body": {"cache_salt": "3"}}


def test_eval_rollouts_leave_unseeded_sampling_args_untouched() -> None:
    vf = SimpleNamespace(RolloutInput=lambda **kwargs: kwargs)
    run_rollout = AsyncMock(return_value={"reward": 1.0, "completion": ["ok"]})
    env = SimpleNamespace(run_rollout=run_rollout)
    sampling_args = {"temperature": 1.0}

    asyncio.run(
        _run_eval_examples(
            vf,
            env,
            [{"question": "q"}],
            clients=[object()],
            model="model",
            sampling_args=sampling_args,
            rollouts_per_example=2,
            max_retries=0,
        )
    )

    assert all(
        call.kwargs["sampling_args"] == {"temperature": 1.0}
        for call in run_rollout.await_args_list
    )


def test_eval_rollouts_bound_inflight_requests() -> None:
    vf = SimpleNamespace(RolloutInput=lambda **kwargs: kwargs)
    active = 0
    peak = 0

    async def run_rollout(*_args, **_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return {"reward": 1.0, "completion": ["ok"]}

    outputs = asyncio.run(
        _run_eval_examples(
            vf,
            SimpleNamespace(run_rollout=run_rollout),
            [{"question": str(index)} for index in range(4)],
            clients=[object()],
            model="model",
            sampling_args={},
            rollouts_per_example=2,
            max_retries=0,
            max_inflight_rollouts=3,
        )
    )

    assert len(outputs) == 8
    assert peak == 3

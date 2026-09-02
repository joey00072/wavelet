from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from wavelet.configs.rl_config import RLConfig, RLEvalConfig
from wavelet.orchestrator.eval_utils import compute_eval_policy_step, pass_at_k
from wavelet.orchestrator.rollout_worker import _final_eval_policy_step
from wavelet.orchestrator.schedule import select_due_eval_envs, target_steps
from wavelet.orchestrator.scheduler import _run_final_evals
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


def test_pass_at_k_for_binary_rewards() -> None:
    metrics = pass_at_k([0.0, 1.0, 0.0, 1.0])

    assert metrics["pass@1"] == pytest.approx(0.5)
    assert metrics["pass@2"] == pytest.approx(5 / 6)
    assert metrics["pass@4"] == pytest.approx(1.0)


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


def test_final_eval_policy_step_uses_last_exported_step() -> None:
    config = RLConfig()
    config.policy_transfer.export_every_steps = 4

    assert _final_eval_policy_step(config, 100) == 100
    assert _final_eval_policy_step(config, 99) == 96


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
    assert metrics["eval/alphabet/effective/avg@2"] == pytest.approx(1.0)
    assert metrics["eval/alphabet/effective/pass@1"] == pytest.approx(1.0)
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

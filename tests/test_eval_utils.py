from __future__ import annotations

import pytest

from wavelet.configs.rl_config import RLEvalConfig
from wavelet.configs.rl_config import RLConfig
from wavelet.entrypoints.rl_inference import _final_eval_policy_step
from wavelet.orchestrator.eval_utils import compute_eval_policy_step, pass_at_k
from wavelet.orchestrator.verifiers import _eval_metrics


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


def test_final_eval_policy_step_uses_last_exported_step() -> None:
    config = RLConfig()
    config.policy_transfer.export_every_steps = 4

    assert _final_eval_policy_step(config, 100) == 100
    assert _final_eval_policy_step(config, 99) == 96


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

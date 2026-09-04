from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from unittest.mock import Mock

import pytest

import wavelet.monitor as monitor_module
from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.metrics import (
    RolloutMetricInputs,
    log_eval_metrics,
    log_rollout_metrics,
    policy_staleness,
    rollout_metrics,
)


def test_rollout_metrics_match_reference_style_grouping() -> None:
    rows = [
        {
            "env_name": "reverse-text",
            "example_id": "a",
            "reward": 1.0,
            "advantage": 0.5,
            "input_ids": [1, 2, 3, 4],
            "loss_mask": [False, False, True, True],
            "metadata": {"stop_condition": "max_turns_reached", "turn_count": 1},
        },
        {
            "env_name": "reverse-text",
            "example_id": "a",
            "reward": 0.0,
            "advantage": -0.5,
            "input_ids": [1, 2, 3],
            "loss_mask": [False, True, True],
            "metadata": {"stop_condition": "max_turns_reached", "turn_count": 1},
        },
        {
            "env_name": "reverse-text",
            "example_id": "b",
            "reward": 1.0,
            "advantage": 0.25,
            "input_ids": [1, 2],
            "loss_mask": [False, True],
            "inference_logprobs": [-0.1],
            "metadata": {
                "stop_condition": "length",
                "is_truncated": True,
                "turn_count": 1,
            },
        },
        {
            "env_name": "reverse-text",
            "example_id": "c",
            "reward": 0.0,
            "advantage": 0.0,
            "input_ids": [1, 2],
            "loss_mask": [False, False],
            "teacher_logprobs": [],
            "metadata": {
                "_wavelet_filtered_rollout": True,
                "error": "parser",
                "turn_count": 1,
            },
        },
    ]

    metrics = rollout_metrics(
        RolloutMetricInputs(
            rows=rows,
            rollouts_per_example=2,
            step=3,
            policy_step=2,
            timings={"generate_completions": 1.5},
        )
    )

    serialized = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "ed7453ef43f587296ac12d2e4923c6f8a82d6e58f33278b41368c05cbb909337"
    )

    assert metrics["progress/samples"] == 4
    assert metrics["progress/problems"] == 3
    assert metrics["progress/tokens"] == 11
    assert metrics["progress/decode_tokens"] == 5
    assert metrics["reward/all/mean"] == pytest.approx(0.5)
    assert metrics["solve_none/all"] == pytest.approx(1 / 3)
    assert metrics["solve_all/all"] == 0.0
    assert metrics["effective_batch_size/all"] == pytest.approx(2 / 3)
    assert metrics["policy/lag"] == 1
    assert metrics["time/generate_completions"] == 1.5
    assert metrics["stop_condition/all/max_turns_reached"] == pytest.approx(2 / 3)
    assert metrics["fate/all/produced"] == 4
    assert metrics["fate/all/trainable"] == 3
    assert metrics["fate/all/zero_loss"] == 1
    assert metrics["fate/all/filtered"] == 1
    assert metrics["fate/all/errored"] == 1
    assert metrics["fate/all/truncated"] == 1
    assert metrics["fate/all/with_inference_logprobs"] == 1
    assert metrics["fate/all/with_teacher_logprobs"] == 1
    assert metrics["fate/reverse-text/filtered_rate"] == pytest.approx(0.25)
    assert metrics["fate/errors/parser"] == 1.0


def test_policy_staleness_splits_generation_and_queue_lag() -> None:
    assert policy_staleness(
        policy_start_step=2,
        policy_end_step=4,
        training_step=5,
    ) == (3, 2, 1)
    assert policy_staleness(
        policy_start_step=7,
        policy_end_step=6,
        training_step=5,
    ) == (0, 0, 0)


def test_rollout_metrics_report_off_policy_components_once_per_rollout() -> None:
    rows = [
        {
            "metadata": {
                "policy_step": 2,
                "policy_end_step": 4,
                "_wavelet_rollout_count": 1,
            }
        },
        {
            "metadata": {
                "policy_step": 3,
                "policy_end_step": 3,
                "_wavelet_rollout_count": 1,
            }
        },
        {
            "metadata": {
                "policy_step": 0,
                "policy_end_step": 5,
                "_wavelet_rollout_count": 0,
            }
        },
    ]

    metrics = rollout_metrics(
        RolloutMetricInputs(rows=rows, rollouts_per_example=1, step=5)
    )

    assert metrics["off_policy/mean"] == 2.5
    assert metrics["off_policy/max"] == 3.0
    assert metrics["off_policy/in_flight/mean"] == 1.0
    assert metrics["off_policy/in_flight/max"] == 2.0
    assert metrics["off_policy/in_queue/mean"] == 1.5
    assert metrics["off_policy/in_queue/max"] == 2.0


def test_rollout_metrics_report_per_phase_timing() -> None:
    rows = [
        {
            "metadata": {
                "_wavelet_rollout_count": 1,
                "rollout": {
                    "timing_seconds": {
                        "generation": 0.8,
                        "scoring": 0.2,
                        "total": 1.1,
                    }
                },
            }
        },
        {
            "metadata": {
                "_wavelet_rollout_count": 1,
                "rollout": {
                    "timing_seconds": {
                        "generation": 1.2,
                        "scoring": 0.4,
                        "total": 1.8,
                    }
                },
            }
        },
    ]

    metrics = rollout_metrics(
        RolloutMetricInputs(rows=rows, rollouts_per_example=1, step=0)
    )

    assert metrics["time/rollout/generation/mean"] == 1.0
    assert metrics["time/rollout/generation/max"] == 1.2
    assert metrics["time/rollout/scoring/mean"] == pytest.approx(0.3)
    assert metrics["time/rollout/total/max"] == 1.8


def test_dropped_and_materialized_error_counts_are_combined() -> None:
    metrics = rollout_metrics(
        RolloutMetricInputs(
            rows=[{"metadata": {"error": {"type": "TimeoutError"}}}],
            rollouts_per_example=1,
            step=0,
            extra_metrics={"fate/errors/timeout_error": 2.0},
        )
    )

    assert metrics["fate/errors/timeout_error"] == 3.0


def test_log_rollout_metrics_writes_per_step_trace(tmp_path) -> None:
    rollout_path = tmp_path / "rollouts.jsonl"
    row = {
        "env_name": "reverse-text",
        "example_id": "a",
        "reward": 1.0,
        "advantage": 0.5,
        "input_ids": [1, 2],
        "loss_mask": [False, True],
        "metadata": {
            "task": {"name": "reverse-text", "example_id": "a"},
            "harness": {"name": "tiny-agent", "type": "agent", "version": "1"},
        },
    }
    rollout_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    metrics = log_rollout_metrics(
        RLConfig(output_dir=tmp_path),
        rollout_path,
        step=4,
        queue_step=5,
        optimizer_step=6,
        policy_step=3,
    )

    trace_path = tmp_path / "traces" / "step-000004.jsonl"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    metrics_row = json.loads(
        (tmp_path / "orchestrator_metrics.jsonl").read_text(encoding="utf-8")
    )
    with (tmp_path / "orchestrator_metrics.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert metrics["progress/samples"] == 1
    assert metrics_row["step"] == 4
    assert metrics_row["progress/queue_step"] == 5.0
    assert csv_rows[0]["step"] == "4.0"
    assert csv_rows[0]["progress/queue_step"] == "5.0"
    assert trace["event"] == "rollout_metrics_logged"
    assert trace["subsystem"] == "orchestrator"
    assert trace["task"] == "reverse-text"
    assert trace["harness"] == "tiny-agent"
    assert trace["queue_step"] == 5
    assert trace["optimizer_step"] == 6
    assert trace["policy_step"] == 3
    assert trace["details"]["trainable"] == 1.0


def test_log_eval_metrics_uses_orchestrator_wandb_run(monkeypatch, tmp_path) -> None:
    metrics = {
        "eval/aime/avg@8": 0.25,
        "eval/aime/pass@8": 0.75,
        "progress/policy_step": 96.0,
    }
    wandb_log = Mock()
    monkeypatch.setattr("wavelet.monitor._wandb_log", wandb_log)

    log_eval_metrics(
        RLConfig(output_dir=tmp_path),
        metrics,
        step=100,
        policy_step=96,
    )

    wandb_log.assert_called_once()
    assert wandb_log.call_args.args[1] == metrics
    assert wandb_log.call_args.kwargs == {"step": 100}
    trace = json.loads(
        (tmp_path / "traces" / "step-000100.jsonl").read_text(encoding="utf-8")
    )
    assert trace["event"] == "eval_metrics_logged"
    assert trace["policy_step"] == 96
    assert trace["details"] == {
        "eval/aime/avg@8": 0.25,
        "eval/aime/pass@8": 0.75,
    }


def test_orchestrator_wandb_config_redacts_secrets(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    run = Mock()

    def fake_init(**kwargs):
        captured.update(kwargs)
        return run

    fake_wandb = types.SimpleNamespace(
        init=fake_init,
        define_metric=Mock(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(monitor_module, "_WANDB_RUN", None)
    config = RLConfig(
        output_dir=tmp_path,
        orchestrator={"verifier_env_args": {"api_token": "secret"}},
        monitor={"wandb": {"enabled": True, "mode": "offline"}},
    )

    monitor_module._wandb_log(config, {"reward": 1.0}, step=1)

    wandb_config = captured["config"]
    assert isinstance(wandb_config, dict)
    assert wandb_config["orchestrator"]["verifier_env_args"] == {
        "api_token": "<redacted>"
    }


def test_rollout_metrics_include_non_overlapping_generation_metrics() -> None:
    metrics = rollout_metrics(
        RolloutMetricInputs(
            rows=[],
            rollouts_per_example=8,
            step=1,
            extra_metrics={"generation/reward/mean": 0.25},
        )
    )

    assert metrics["generation/reward/mean"] == 0.25


def test_rollout_metrics_keep_interleaved_group_values_aligned() -> None:
    rows = [
        {
            "example_id": "a",
            "input_ids": [1, 2],
            "loss_mask": [False, True],
        },
        {
            "example_id": "b",
            "input_ids": list(range(10)),
            "loss_mask": [False] * 9 + [True],
        },
        {
            "example_id": "a",
            "input_ids": [1, 2, 3, 4],
            "loss_mask": [False, False, True, True],
        },
    ]

    metrics = rollout_metrics(
        RolloutMetricInputs(rows=rows, rollouts_per_example=2, step=1)
    )

    assert metrics["seq_len/all/mean"] == pytest.approx(6.5)
    assert metrics["decode_len/all/mean"] == pytest.approx(1.25)


def test_rollout_metrics_prefer_exact_dispatch_group_identity() -> None:
    rows = [
        {
            "env_name": "math",
            "example_id": "duplicate-public-id",
            "reward": 1.0,
            "metadata": {"group_key": "dispatch-a"},
        },
        {
            "env_name": "math",
            "example_id": "duplicate-public-id",
            "reward": 0.0,
            "metadata": {"group_key": "dispatch-b"},
        },
    ]

    metrics = rollout_metrics(
        RolloutMetricInputs(rows=rows, rollouts_per_example=1, step=1)
    )

    assert metrics["progress/problems"] == 2
    assert metrics["solve_none/all"] == pytest.approx(0.5)
    assert metrics["solve_all/all"] == pytest.approx(0.5)


def test_rollout_metrics_do_not_count_extra_trajectory_branches_as_rollouts() -> None:
    rows = [
        {
            "example_id": "a",
            "reward": 1.0,
            "metadata": {"_wavelet_rollout_count": 1},
        },
        {
            "example_id": "a",
            "reward": 1.0,
            "metadata": {"_wavelet_rollout_count": 0},
        },
        {
            "example_id": "a",
            "reward": 0.0,
            "metadata": {"_wavelet_rollout_count": 1},
        },
    ]

    metrics = rollout_metrics(
        RolloutMetricInputs(rows=rows, rollouts_per_example=2, step=1)
    )

    assert metrics["reward/all/mean"] == pytest.approx(0.5)
    assert metrics["solve_none/all"] == 0.0
    assert metrics["solve_all/all"] == 0.0
    assert metrics["effective_batch_size/all"] == 1.0


def test_rollout_metrics_reject_extra_metric_overrides() -> None:
    with pytest.raises(ValueError, match="replace core metrics: step"):
        rollout_metrics(
            RolloutMetricInputs(
                rows=[],
                rollouts_per_example=8,
                step=1,
                extra_metrics={"step": 99.0},
            )
        )

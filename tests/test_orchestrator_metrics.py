from __future__ import annotations

import csv
import hashlib
import json

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.metrics import (
    RolloutMetricInputs,
    log_rollout_metrics,
    rollout_metrics,
)


def test_rollout_metrics_match_reference_style_grouping() -> None:
    rows = [
        {
            "source": "policy",
            "env_name": "reverse-text",
            "example_id": "a",
            "reward": 1.0,
            "advantage": 0.5,
            "input_ids": [1, 2, 3, 4],
            "loss_mask": [False, False, True, True],
            "metadata": {"stop_condition": "max_turns_reached", "turn_count": 1},
        },
        {
            "source": "policy",
            "env_name": "reverse-text",
            "example_id": "a",
            "reward": 0.0,
            "advantage": -0.5,
            "input_ids": [1, 2, 3],
            "loss_mask": [False, True, True],
            "metadata": {"stop_condition": "max_turns_reached", "turn_count": 1},
        },
        {
            "source": "distill",
            "env_name": "reverse-text",
            "example_id": "b",
            "reward": 1.0,
            "advantage": 0.25,
            "input_ids": [1, 2],
            "loss_mask": [False, True],
            "inference_logprobs": [-0.1],
            "ref_logprobs": [-0.2],
            "rl_weights": 0.0,
            "ref_kl_weights": 1.0,
            "metadata": {
                "stop_condition": "length",
                "is_truncated": True,
                "turn_count": 1,
            },
        },
        {
            "source": "distill",
            "env_name": "reverse-text",
            "example_id": "c",
            "reward": 0.0,
            "advantage": 0.0,
            "input_ids": [1, 2],
            "loss_mask": [False, False],
            "teacher_logprobs": [],
            "ref_logprobs": [],
            "rl_weights": 0.0,
            "ref_kl_weights": 1.0,
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
        "f7f42faf78623927c0d6d194d7ae45de98f504b031ae8eea1bca8ff06f8a4d31"
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
    assert metrics["fate/all/with_ref_logprobs"] == 2
    assert metrics["fate/all/with_rl_loss"] == 2
    assert metrics["fate/all/with_ref_kl_loss"] == 1
    assert metrics["fate/reverse-text/filtered_rate"] == pytest.approx(0.25)
    assert metrics["batch/source/policy"] == pytest.approx(0.5)
    assert metrics["batch/source/distill"] == pytest.approx(0.5)
    assert metrics["fate/source/distill/with_ref_logprobs_rate"] == 1.0
    assert metrics["fate/source/distill/with_ref_kl_loss_rate"] == 0.5


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
    metrics_row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    with (tmp_path / "metrics.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert metrics["progress/samples"] == 1
    assert metrics_row["step"] == 4
    assert metrics_row["subsystem"] == "orchestrator"
    assert metrics_row["progress/queue_step"] == 5.0
    assert csv_rows[0]["step"] == "4"
    assert csv_rows[0]["subsystem"] == "orchestrator"
    assert csv_rows[0]["progress/queue_step"] == "5.0"
    assert trace["event"] == "rollout_metrics_logged"
    assert trace["subsystem"] == "orchestrator"
    assert trace["task"] == "reverse-text"
    assert trace["harness"] == "tiny-agent"
    assert trace["queue_step"] == 5
    assert trace["optimizer_step"] == 6
    assert trace["policy_step"] == 3
    assert trace["details"]["trainable"] == 1.0

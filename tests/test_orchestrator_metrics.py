from __future__ import annotations

import json

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.metrics import RolloutMetricInputs, rollout_metrics
from wavelet.orchestrator.metrics import log_rollout_metrics


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
    assert metrics["progress/samples"] == 1
    assert trace["event"] == "rollout_metrics_logged"
    assert trace["subsystem"] == "orchestrator"
    assert trace["task"] == "reverse-text"
    assert trace["harness"] == "tiny-agent"
    assert trace["queue_step"] == 5
    assert trace["optimizer_step"] == 6
    assert trace["policy_step"] == 3
    assert trace["details"]["trainable"] == 1.0

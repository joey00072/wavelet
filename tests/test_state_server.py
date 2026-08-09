import json

import pytest
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.queue import FileSystemRolloutSender
from wavelet.orchestrator.state_server import OrchestratorRunState, _build_state_app


def test_orchestrator_state_tracks_rollout_and_policy_updates(tmp_path) -> None:
    config = RLConfig(output_dir=tmp_path)
    state = OrchestratorRunState(config, target_step=10)

    state.update_policy(
        loaded_step=3,
        pending_load=False,
        requested_step=None,
        available_tail=[0, 1, 2, 3],
    )
    state.mark_submitted(queue_step=4, optimizer_step=1, chunk_index=0, pending_count=1)
    state.mark_completed(
        queue_step=4,
        optimizer_step=1,
        chunk_index=0,
        pending_count=0,
        completed_count=1,
    )
    state.mark_published(
        queue_step=4,
        optimizer_step=1,
        chunk_index=0,
        path="rollouts/step-000004/rollouts.jsonl",
        next_queue_step_to_publish=5,
        completed_count=0,
    )

    snapshot = state.snapshot()

    assert snapshot["target_step"] == 10
    assert snapshot["policy"]["loaded_step"] == 3
    assert snapshot["rollouts"]["pending_count"] == 0
    assert snapshot["rollouts"]["next_queue_step_to_publish"] == 5
    assert snapshot["rollouts"]["published_tail"][-1]["queue_step"] == 4
    assert snapshot["events"][-3]["type"] == "submitted"
    assert snapshot["events"][-1]["type"] == "published"
    assert state.events(limit=1)[0]["type"] == "published"


def test_orchestrator_state_config_redacts_secrets(tmp_path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        orchestrator={
            "verifier_api_key_var": "MY_API_KEY",
            "verifier_env_args": {"api_key": "secret", "safe": "value"},
        },
    )
    state = OrchestratorRunState(config, target_step=1)

    sanitized = state.sanitized_config()

    assert sanitized["orchestrator"]["verifier_api_key_var"] == "<redacted>"
    assert sanitized["orchestrator"]["verifier_env_args"]["api_key"] == "<redacted>"
    assert sanitized["orchestrator"]["verifier_env_args"]["safe"] == "value"


def test_orchestrator_state_describes_mixed_algorithms_and_observations(
    tmp_path,
) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        algo={"type": "grpo"},
        orchestrator={
            "verifier_env_id": "reverse-text",
            "train_sources": [
                {"name": "policy", "weight": 2},
                {
                    "name": "distill",
                    "weight": 1,
                    "algo": {
                        "type": "opd",
                        "teacher": {
                            "name": "teacher-a",
                            "base_url": [
                                "http://teacher-a:8001/v1",
                                "http://teacher-a:8002/v1",
                            ],
                            "api_key_var": "TEACHER_API_KEY",
                        },
                    },
                },
                {
                    "name": "distill-b",
                    "weight": 1,
                    "algo": {
                        "type": "opd",
                        "teacher": {
                            "name": "teacher-b",
                            "base_url": "http://teacher-b:8001/v1",
                        },
                    },
                },
            ],
        },
    )
    (tmp_path / "orchestrator_metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 4,
                "batch/source/distill": 1 / 3,
                "reward/source/distill/mean": 0.75,
                "fate/source/distill/produced": 2,
                "fate/source/distill/trainable_rate": 1.0,
                "fate/source/distill/with_ref_logprobs_rate": 1.0,
                "fate/source/distill/with_ref_kl_loss_rate": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = OrchestratorRunState(config, target_step=10)
    algorithms = state.algorithm_snapshot()
    snapshot = state.snapshot()

    assert algorithms == snapshot["algorithms"]
    assert algorithms["loss_components"] == ["ref_kl", "rl"]
    assert algorithms["teacher_count"] == 2
    assert algorithms["multi_teacher"] is True
    assert algorithms["student"]["adapter_count"] == 1
    assert algorithms["observed_step"] == 4.0
    assert algorithms["sources"][0]["algorithm"]["type"] == "grpo"
    distill = algorithms["sources"][1]
    assert distill["algorithm"]["type"] == "opd"
    assert distill["algorithm"]["teacher"] == {
        "name": "teacher-a",
        "base_urls": [
            "http://teacher-a:8001/v1",
            "http://teacher-a:8002/v1",
        ],
        "replica_count": 2,
    }
    assert "api_key_var" not in distill["algorithm"]["teacher"]
    assert distill["observed"]["batch_fraction"] == pytest.approx(1 / 3)
    assert distill["observed"]["ref_logprobs_rate"] == 1.0


def test_state_api_exposes_algorithm_snapshot(tmp_path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        algo={"type": "opd", "teacher": {"name": "t", "base_url": "http://t:1"}},
    )
    state = OrchestratorRunState(config, target_step=1)
    app = _build_state_app(
        state,
        fastapi=FastAPI,
        query=Query,
        cors_middleware=CORSMiddleware,
    )

    response = TestClient(app).get("/algorithms")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default"]["type"] == "opd"
    assert payload["sources"][0]["algorithm"]["teacher"]["name"] == "t"
    assert payload["student"]["adapter_count"] == 1


def test_orchestrator_state_includes_queue_summary(tmp_path) -> None:
    config = RLConfig(output_dir=tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    sender = FileSystemRolloutSender(tmp_path, config.transport)
    sender.publish(source, step=0, optimizer_step=0, rows=1)
    state = OrchestratorRunState(config, target_step=1)

    snapshot = state.snapshot()
    queues = state.queue_snapshot(detail=True, limit=10)

    assert snapshot["queue_summary"]["ready_count"] == 1
    assert queues is not None
    assert queues["summary"]["ready_count"] == 1
    assert queues["items"][0]["queue_step"] == 0


def test_orchestrator_state_inspects_latest_stable_rollout_batch(tmp_path) -> None:
    config = RLConfig(output_dir=tmp_path)
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "prompt": [{"role": "user", "content": "low"}],
            "completion": [{"role": "assistant", "content": "bad"}],
            "reward": 0.0,
            "advantage": -1.0,
            "example_id": "a",
            "metadata": {
                "group_key": "g-a",
                "rollout_key": "g-a:0",
                "completion_token_count": 3,
            },
            "input_ids": [1, 2, 3],
            "rl_weights": 1.0,
        },
        {
            "prompt": [{"role": "user", "content": "mid"}],
            "completion": [{"role": "assistant", "content": "ok"}],
            "reward": 0.5,
            "advantage": 0.0,
            "example_id": "b",
            "metadata": {"group_key": "g-b", "rollout_key": "g-b:0"},
        },
        {
            "prompt": [{"role": "user", "content": "high"}],
            "completion": [{"role": "assistant", "content": "good"}],
            "reward": 1.0,
            "advantage": 1.0,
            "example_id": "c",
            "metadata": {
                "group_key": "g-c",
                "rollout_key": "g-c:0",
                "completion_token_count": 1,
            },
            "ref_logprobs": [-0.2],
            "rl_weights": 0.0,
            "ref_kl_weights": 1.0,
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    sender = FileSystemRolloutSender(tmp_path, config.transport)
    sender.publish(source, step=0, optimizer_step=0, rows=3)
    sender.publish(source, step=2, optimizer_step=2, rows=3)
    state = OrchestratorRunState(config, target_step=3)

    inspection = state.inspect_rollouts(
        step=None,
        random_count=2,
        seed=1,
        max_scan_rows=100,
        max_text_chars=1000,
    )

    assert inspection["available"] is True
    assert inspection["queue_step"] == 2
    assert inspection["scanned_rows"] == 3
    assert inspection["stats"]["reward"]["mean"] == pytest.approx(0.5)
    assert inspection["samples"]["min_reward"]["reward"] == 0.0
    assert inspection["samples"]["max_reward"]["reward"] == 1.0
    assert inspection["samples"]["near_mean_reward"]["reward"] == 0.5
    assert inspection["samples"]["min_reward"]["loss_components"] == ["rl"]
    assert inspection["samples"]["max_reward"]["loss_components"] == ["ref_kl"]
    assert inspection["samples"]["max_reward"]["has_ref_logprobs"] is True
    assert len(inspection["samples"]["random"]) == 2
    assert "input_ids" not in inspection["samples"]["min_reward"]


def test_orchestrator_state_rollout_inspection_is_bounded(tmp_path) -> None:
    config = RLConfig(output_dir=tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"reward": index}) + "\n" for index in range(5)),
        encoding="utf-8",
    )
    FileSystemRolloutSender(tmp_path, config.transport).publish(
        source,
        step=0,
        optimizer_step=0,
        rows=5,
    )
    state = OrchestratorRunState(config, target_step=1)

    inspection = state.inspect_rollouts(
        step=0,
        random_count=10,
        seed=None,
        max_scan_rows=2,
        max_text_chars=1000,
    )

    assert inspection["scanned_rows"] == 2
    assert inspection["truncated"] is True
    assert inspection["stats"]["reward"]["max"] == 1.0
    assert len(inspection["samples"]["random"]) == 2

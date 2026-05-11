from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.queue import FileSystemRolloutSender
from wavelet.orchestrator.state_server import OrchestratorRunState


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

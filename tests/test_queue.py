from __future__ import annotations

import json
from pathlib import Path

import pytest

from wavelet.configs.rl_config import RLPolicyTransferConfig, RLTransportConfig
from wavelet.orchestrator.queue import (
    FileSystemPolicyReceiver,
    FileSystemRolloutReceiver,
    FileSystemRolloutSender,
    RolloutBatch,
    RolloutManifest,
    read_manifest,
    tail_events,
)


def _write_source(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_rollout_receiver_wait_available_skips_missing_step(tmp_path: Path) -> None:
    config = RLTransportConfig(poll_interval_seconds=0.001)
    sender = FileSystemRolloutSender(tmp_path, config)
    source = _write_source(tmp_path / "rollouts.jsonl", "{}\n")

    sender.publish(source, step=2)
    sender.publish(source, step=1)

    receiver = FileSystemRolloutReceiver(tmp_path, config)

    assert receiver.wait_available().step == 1
    assert receiver.wait_available().step == 2


def test_queue_public_imports_are_compatible() -> None:
    assert RolloutBatch.__name__ == "RolloutBatch"
    assert FileSystemPolicyReceiver.__name__ == "FileSystemPolicyReceiver"


def test_publish_does_not_mark_batch_stable_when_manifest_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = RLTransportConfig()
    sender = FileSystemRolloutSender(tmp_path, config)
    source = _write_source(tmp_path / "source.jsonl", "{}\n")

    def fail_manifest(*_args, **_kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("wavelet.transport.queue.write_manifest", fail_manifest)

    with pytest.raises(OSError, match="disk full"):
        sender.publish(source, step=0, optimizer_step=0)

    assert not (tmp_path / "rollouts" / "step-000000" / "STABLE").exists()


def test_rollout_receiver_wait_available_advances_past_late_gap(
    tmp_path: Path,
) -> None:
    config = RLTransportConfig(poll_interval_seconds=0.001)
    sender = FileSystemRolloutSender(tmp_path, config)
    source = _write_source(tmp_path / "rollouts.jsonl", "{}\n")

    sender.publish(source, step=1)
    receiver = FileSystemRolloutReceiver(tmp_path, config)

    assert receiver.wait_available().step == 1
    sender.publish(source, step=0)

    assert receiver.wait_available().step == 0
    assert receiver.next_step == 2


def test_publish_without_metadata_preserves_old_artifacts(tmp_path: Path) -> None:
    config = RLTransportConfig(poll_interval_seconds=0.001)
    sender = FileSystemRolloutSender(tmp_path, config)
    source = _write_source(tmp_path / "rollouts.jsonl", "{}\n")

    batch = sender.publish(source, step=0)

    assert sorted(path.name for path in batch.step_dir.iterdir()) == [
        "STABLE",
        "rollouts.jsonl",
    ]


def test_publish_with_metadata_writes_manifest(tmp_path: Path) -> None:
    config = RLTransportConfig(poll_interval_seconds=0.001)
    sender = FileSystemRolloutSender(tmp_path, config)
    source = _write_source(tmp_path / "rollouts.jsonl", "{}\n")

    batch = sender.publish(
        source,
        step=0,
        optimizer_step=0,
        policy_step=0,
        rows=1,
        events_dir=tmp_path / "events",
    )

    manifest = read_manifest(batch.step_dir)
    assert isinstance(manifest, RolloutManifest)
    assert manifest.queue_step == 0
    assert manifest.optimizer_step == 0
    assert manifest.policy_step == 0
    assert manifest.rows == 1
    assert manifest.payload_bytes == 3
    assert manifest.transfer_seconds is not None
    assert manifest.transfer_seconds >= 0

    events, parse_errors = tail_events(tmp_path / "events", limit=1)
    assert parse_errors == 0
    assert len(events) == 1
    assert events[0].details is not None
    assert events[0].details["payload_bytes"] == 3
    assert events[0].details["transfer_seconds"] >= 0


def test_stable_rollout_batch_cannot_be_overwritten(tmp_path: Path) -> None:
    config = RLTransportConfig(poll_interval_seconds=0.001)
    sender = FileSystemRolloutSender(tmp_path, config)
    first = _write_source(tmp_path / "first.jsonl", '{"value": 1}\n')
    second = _write_source(tmp_path / "second.jsonl", '{"value": 2}\n')
    batch = sender.publish(first, step=0)

    with pytest.raises(FileExistsError, match="already stable"):
        sender.publish(second, step=0)

    assert sender.stable_batch(0) == batch
    assert batch.path.read_text(encoding="utf-8") == '{"value": 1}\n'


def test_rollout_receiver_records_wait_metrics(tmp_path: Path) -> None:
    config = RLTransportConfig(poll_interval_seconds=0.001)
    sender = FileSystemRolloutSender(tmp_path, config)
    source = _write_source(tmp_path / "rollouts.jsonl", "{}\n")
    sender.publish(
        source,
        step=0,
        optimizer_step=0,
        policy_step=0,
        rows=1,
    )
    receiver = FileSystemRolloutReceiver(
        tmp_path,
        config,
        events_dir=tmp_path / "events",
        consumer_id="trainer",
    )

    assert receiver.wait().step == 0

    events, parse_errors = tail_events(tmp_path / "events", limit=1)
    assert parse_errors == 0
    assert len(events) == 1
    assert events[0].kind == "rollout_received"
    assert events[0].queue_step == 0
    assert events[0].optimizer_step == 0
    assert events[0].policy_step == 0
    assert events[0].consumer_id == "trainer"
    assert events[0].details is not None
    assert events[0].details["mode"] == "wait"
    assert events[0].details["payload_bytes"] == 3
    assert events[0].details["wait_seconds"] >= 0
    trace = json.loads(
        (tmp_path / "traces" / "step-000000.jsonl").read_text(encoding="utf-8")
    )
    assert trace["subsystem"] == "trainer"
    assert trace["event"] == "rollout_received"
    assert trace["queue_step"] == 0
    assert trace["optimizer_step"] == 0
    assert trace["policy_step"] == 0
    assert trace["details"]["consumer_id"] == "trainer"


def test_policy_receiver_records_wait_metrics(tmp_path: Path) -> None:
    config = RLPolicyTransferConfig(poll_interval_seconds=0.001)
    policy_dir = tmp_path / "policies" / "step-000000"
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy.json").write_text(
        json.dumps({"artifact": {"bytes": len(b"weights")}}),
        encoding="utf-8",
    )
    (policy_dir / "adapter").mkdir()
    (policy_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (policy_dir / "STABLE").touch()
    receiver = FileSystemPolicyReceiver(
        tmp_path,
        config,
        events_dir=tmp_path / "events",
        consumer_id="inference",
    )

    assert receiver.wait().step == 0

    events, parse_errors = tail_events(tmp_path / "events", limit=1)
    assert parse_errors == 0
    assert len(events) == 1
    assert events[0].kind == "policy_received"
    assert events[0].policy_step == 0
    assert events[0].consumer_id == "inference"
    assert events[0].details is not None
    assert events[0].details["mode"] == "wait"
    assert events[0].details["payload_bytes"] == len(b"weights")
    assert events[0].details["wait_seconds"] >= 0
    trace = json.loads(
        (tmp_path / "traces" / "step-000000.jsonl").read_text(encoding="utf-8")
    )
    assert trace["subsystem"] == "inference"
    assert trace["event"] == "policy_received"
    assert trace["policy_step"] == 0
    assert trace["details"]["consumer_id"] == "inference"

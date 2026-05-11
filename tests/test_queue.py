from __future__ import annotations

from pathlib import Path

from wavelet.configs.rl_config import RLTransportConfig
from wavelet.orchestrator.queue import (
    FileSystemPolicyReceiver,
    FileSystemRolloutReceiver,
    FileSystemRolloutSender,
    RolloutBatch,
    RolloutManifest,
    read_manifest,
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

    batch = sender.publish(source, step=0, optimizer_step=0, policy_step=0, rows=1)

    manifest = read_manifest(batch.step_dir)
    assert isinstance(manifest, RolloutManifest)
    assert manifest.queue_step == 0
    assert manifest.optimizer_step == 0
    assert manifest.policy_step == 0
    assert manifest.rows == 1

from __future__ import annotations

from pathlib import Path

from wavelet.configs.rl_config import RLTransportConfig
from wavelet.orchestrator.queue import FileSystemRolloutReceiver, FileSystemRolloutSender


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

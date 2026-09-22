from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from wavelet.contracts.queue_records import RolloutBatch
from wavelet.trainer.loop import run_rl_training


@dataclass
class _FakeTrainer:
    step: int = 0
    events: list[str] = field(default_factory=list)

    def validate_rollout_batch(self, _batch, *, row_count: int) -> None:
        self.events.append(f"validate:{row_count}")

    def record_rollout_claim(self, _batch, *, trainer_step_before: int) -> None:
        self.events.append(f"claim:{trainer_step_before}")

    def load_rollout_path(self, _path: Path) -> None:
        self.events.append("load")

    def prepare_for_training(self) -> None:
        self.events.append("prepare")

    def train_until(self, target_step: int) -> None:
        self.events.append("train")
        self.step = target_step

    def record_rollout_consumed(
        self,
        _batch,
        *,
        trainer_step_before: int,
        optimizer_step_completed: bool,
    ) -> None:
        assert optimizer_step_completed
        self.events.append(f"consumed:{trainer_step_before}")

    def export_policy(self, *, step: int) -> Path | None:
        self.events.append(f"export:{step}")
        return None

    def offload_after_refit(self) -> None:
        self.events.append("offload")


class _Receiver:
    def __init__(self, batch: RolloutBatch) -> None:
        self.batch = batch

    def wait(self) -> RolloutBatch:
        return self.batch


def test_shared_rl_loop_preserves_training_order(tmp_path: Path) -> None:
    payload = tmp_path / "rollouts.jsonl"
    payload.write_text("{}\n", encoding="utf-8")
    batch = RolloutBatch(step=0, path=payload, step_dir=tmp_path)
    trainer = _FakeTrainer()

    run_rl_training(
        trainer,
        _Receiver(batch),
        target_step=1,
    )

    assert trainer.events == [
        "validate:1",
        "claim:0",
        "load",
        "prepare",
        "train",
        "consumed:0",
        "export:1",
        "offload",
    ]

"""Backend contract used by the shared RL training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from wavelet.contracts.queue_records import RolloutBatch


class TrainerBackend(Protocol):
    step: int

    def validate_rollout_batch(
        self, batch: RolloutBatch, *, row_count: int
    ) -> None: ...

    def record_rollout_claim(
        self, batch: RolloutBatch, *, trainer_step_before: int
    ) -> None: ...

    def load_rollout_path(self, path: Path) -> None: ...

    def prepare_for_training(self) -> None: ...

    def train_until(self, target_step: int) -> None: ...

    def record_rollout_consumed(
        self,
        batch: RolloutBatch,
        *,
        trainer_step_before: int,
        optimizer_step_completed: bool,
    ) -> None: ...

    def export_policy(self, *, step: int) -> Path | None: ...

    def offload_after_refit(self) -> None: ...


__all__ = ["TrainerBackend"]

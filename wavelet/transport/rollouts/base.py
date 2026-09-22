"""Backend-independent rollout transport protocols."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from wavelet.contracts.queue_records import RolloutBatch, RolloutManifest


class RolloutSender(Protocol):
    def publish(
        self,
        source_path: Path,
        *,
        step: int,
        optimizer_step: int | None = None,
        chunk_index: int | None = None,
        policy_step: int | None = None,
        rows: int | None = None,
        tokens: int | None = None,
        training_records: Iterable[dict[str, object]] | None = None,
    ) -> RolloutBatch: ...

    def stable_batch(self, step: int) -> RolloutBatch | None: ...


class RolloutReceiver(Protocol):
    def wait(self) -> RolloutBatch: ...

    def wait_available(self) -> RolloutBatch: ...

    def available_steps(self) -> list[int]: ...


__all__ = ["RolloutManifest", "RolloutReceiver", "RolloutSender"]

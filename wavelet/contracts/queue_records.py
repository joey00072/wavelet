"""Shared immutable records used by rollout and policy transports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINING_PAYLOAD_SUFFIX = ".train.msgpack"


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    step: int
    path: Path
    step_dir: Path

    @property
    def training_path(self) -> Path:
        binary = self.path.with_name(self.path.name + TRAINING_PAYLOAD_SUFFIX)
        return binary if binary.is_file() else self.path


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    step: int
    step_dir: Path

    @property
    def meta_path(self) -> Path:
        return self.step_dir / "policy.json"


@dataclass(frozen=True, slots=True)
class RolloutManifest:
    format_version: int
    queue_step: int
    optimizer_step: int | None
    chunk_index: int | None
    policy_step: int | None
    rows: int | None
    tokens: int | None
    reward_mean: float | None
    producer_id: str | None
    created_at: str
    payload_bytes: int | None = None
    transfer_seconds: float | None = None
    environment_record_cursors: dict[str, list[int]] | None = None
    environment_next_record_cursors: dict[str, int] | None = None
    environment_selection_cursor: int | None = None
    curriculum_state: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    format_version: int
    queue_step: int
    consumer_id: str
    trainer_step_before: int
    claimed_at: str


@dataclass(frozen=True, slots=True)
class ConsumedRecord:
    format_version: int
    queue_step: int
    consumer_id: str
    trainer_step_before: int
    trainer_step_after: int
    optimizer_step_completed: bool
    consumed_at: str


@dataclass(frozen=True, slots=True)
class QueueEvent:
    time: str
    kind: str
    queue_step: int | None = None
    optimizer_step: int | None = None
    policy_step: int | None = None
    producer_id: str | None = None
    consumer_id: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class QueueItemSnapshot:
    queue_step: int
    status: str
    stable: bool
    manifest: RolloutManifest | None
    claim: ClaimRecord | None
    consumed: ConsumedRecord | None
    age_seconds: float | None
    parse_errors: list[str]


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    ready_count: int
    claimed_count: int
    consumed_count: int
    incomplete_count: int
    unknown_count: int
    stale_ready_count: int
    abandoned_claim_count: int
    oldest_ready_age_seconds: float | None
    latest_queue_step: int | None
    latest_consumed_queue_step: int | None
    next_expected_trainer_queue_step: int
    event_parse_error_count: int
    items: list[QueueItemSnapshot]


@dataclass(frozen=True, slots=True)
class PolicyQueueSnapshot:
    latest_exported_step: int | None
    steps: list[int]
    incomplete_steps: list[int]

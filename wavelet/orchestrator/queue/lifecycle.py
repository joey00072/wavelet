from __future__ import annotations

import json
import logging
import os
import socket
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from wavelet.orchestrator.queue.events import append_event_best_effort
from wavelet.orchestrator.queue.types import (
    CLAIM_FILENAME,
    CONSUMED_FILENAME,
    MANIFEST_FILENAME,
    ClaimRecord,
    ConsumedRecord,
    QueueEvent,
    RolloutBatch,
    RolloutManifest,
)
from wavelet.orchestrator.trace import append_trace_event_best_effort, make_trace_event


logger = logging.getLogger(__name__)

RecordT = TypeVar("RecordT")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def process_identity(role: str) -> str:
    return f"{role}:{socket.gethostname()}:{os.getpid()}"


def write_manifest(step_dir: Path, manifest: RolloutManifest) -> Path:
    return _write_record(step_dir / MANIFEST_FILENAME, manifest)


def read_manifest(step_dir: Path) -> RolloutManifest | None:
    return _read_record(step_dir / MANIFEST_FILENAME, RolloutManifest)


def write_claim(step_dir: Path, claim: ClaimRecord) -> Path:
    return _write_record(step_dir / CLAIM_FILENAME, claim)


def read_claim(step_dir: Path) -> ClaimRecord | None:
    return _read_record(step_dir / CLAIM_FILENAME, ClaimRecord)


def write_consumed(step_dir: Path, consumed: ConsumedRecord) -> Path:
    return _write_record(step_dir / CONSUMED_FILENAME, consumed)


def read_consumed(step_dir: Path) -> ConsumedRecord | None:
    return _read_record(step_dir / CONSUMED_FILENAME, ConsumedRecord)


def record_rollout_claim(
    batch: RolloutBatch,
    *,
    trainer_step_before: int,
    consumer_id: str | None = None,
    events_dir: Path | None = None,
) -> ClaimRecord | None:
    consumer_id = consumer_id or process_identity("rl-trainer")
    claim = ClaimRecord(
        format_version=1,
        queue_step=batch.step,
        consumer_id=consumer_id,
        trainer_step_before=trainer_step_before,
        claimed_at=utc_now(),
    )
    try:
        write_claim(batch.step_dir, claim)
    except Exception as exc:  # pragma: no cover - defensive observability guard
        logger.warning("Failed to write rollout claim for step %s: %s", batch.step, exc)
        return None
    append_event_best_effort(
        events_dir,
        QueueEvent(
            time=claim.claimed_at,
            kind="rollout_claimed",
            queue_step=batch.step,
            consumer_id=claim.consumer_id,
        ),
    )
    append_trace_event_best_effort(
        _trace_output_dir(events_dir),
        make_trace_event(
            subsystem="trainer",
            event="rollout_claimed",
            step=trainer_step_before,
            queue_step=batch.step,
            details={
                "consumer_id": claim.consumer_id,
                "path": str(batch.path),
            },
        ),
    )
    return claim


def record_rollout_consumed(
    batch: RolloutBatch,
    *,
    trainer_step_before: int,
    trainer_step_after: int,
    optimizer_step_completed: bool,
    consumer_id: str | None = None,
    events_dir: Path | None = None,
) -> ConsumedRecord | None:
    consumer_id = consumer_id or process_identity("rl-trainer")
    consumed = ConsumedRecord(
        format_version=1,
        queue_step=batch.step,
        consumer_id=consumer_id,
        trainer_step_before=trainer_step_before,
        trainer_step_after=trainer_step_after,
        optimizer_step_completed=optimizer_step_completed,
        consumed_at=utc_now(),
    )
    try:
        write_consumed(batch.step_dir, consumed)
    except Exception as exc:  # pragma: no cover - defensive observability guard
        logger.warning(
            "Failed to write rollout consumed record for step %s: %s",
            batch.step,
            exc,
        )
        return None
    append_event_best_effort(
        events_dir,
        QueueEvent(
            time=consumed.consumed_at,
            kind="rollout_consumed",
            queue_step=batch.step,
            consumer_id=consumed.consumer_id,
        ),
    )
    append_trace_event_best_effort(
        _trace_output_dir(events_dir),
        make_trace_event(
            subsystem="trainer",
            event="rollout_consumed",
            step=trainer_step_after,
            queue_step=batch.step,
            optimizer_step=trainer_step_after,
            details={
                "consumer_id": consumed.consumer_id,
                "trainer_step_before": trainer_step_before,
                "optimizer_step_completed": optimizer_step_completed,
            },
        ),
    )
    return consumed


def _trace_output_dir(events_dir: Path | None) -> Path | None:
    return events_dir.parent if events_dir is not None else None


def _write_record(path: Path, record: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(asdict(record), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _read_record(path: Path, record_type: type[RecordT]) -> RecordT | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError(f"Expected object in {path}.")
    return record_type(**row)

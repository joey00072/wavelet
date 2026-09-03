"""Filesystem rollout queue, lifecycle, metrics, and inspection."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from wavelet.configs.rl_config import RLPolicyTransferConfig, RLTransportConfig
from wavelet.monitor import tail_jsonl
from wavelet.orchestrator.trace import append_trace_event_best_effort, make_trace_event

STEP_DIR_PREFIX = "step-"


STABLE_BATCH_MARKER = "STABLE"


POLICY_META_FILENAME = "policy.json"


MANIFEST_FILENAME = "manifest.json"


CLAIM_FILENAME = "claim.json"


CONSUMED_FILENAME = "consumed.json"


QUEUE_EVENT_FILENAME = "queue.jsonl"


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    step: int
    path: Path
    step_dir: Path


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    step: int
    step_dir: Path

    @property
    def adapter_dir(self) -> Path:
        return self.step_dir / "adapter"

    @property
    def model_dir(self) -> Path:
        return self.step_dir / "model"

    @property
    def meta_path(self) -> Path:
        return self.step_dir / POLICY_META_FILENAME


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


logger = logging.getLogger(__name__)


def append_event(events_dir: Path, event: QueueEvent) -> Path:
    events_dir.mkdir(parents=True, exist_ok=True)
    path = events_dir / QUEUE_EVENT_FILENAME
    line = json.dumps(asdict(event), sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    return path


def append_event_best_effort(events_dir: Path | None, event: QueueEvent) -> None:
    if events_dir is None:
        return
    try:
        append_event(events_dir, event)
    except Exception as exc:  # pragma: no cover - defensive observability guard
        logger.warning("Failed to append queue event: %s", exc)


def tail_events(events_dir: Path, *, limit: int) -> tuple[list[QueueEvent], int]:
    if limit <= 0:
        return [], 0
    path = events_dir / QUEUE_EVENT_FILENAME
    if not path.exists():
        return [], 0
    rows, parse_errors = tail_jsonl(path, limit=limit)
    events: list[QueueEvent] = []
    for row in rows:
        try:
            events.append(QueueEvent(**row))
        except TypeError:
            parse_errors += 1
    return events, parse_errors


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


def validate_rollout_manifest(
    batch: RolloutBatch,
    *,
    queue_step: int,
    optimizer_step: int,
    chunk_index: int | None,
    rows: int,
    minimum_policy_step: int,
    maximum_policy_step: int,
) -> RolloutManifest:
    """Validate immutable rollout provenance before reuse or training."""
    manifest = read_manifest(batch.step_dir)
    if manifest is None:
        raise ValueError(f"Rollout queue step {batch.step} is missing manifest.json.")
    expected = {
        "queue_step": queue_step,
        "optimizer_step": optimizer_step,
        "chunk_index": chunk_index,
        "rows": rows,
    }
    actual = {
        "queue_step": manifest.queue_step,
        "optimizer_step": manifest.optimizer_step,
        "chunk_index": manifest.chunk_index,
        "rows": manifest.rows,
    }
    invalid = [
        f"{name}={actual[name]!r} (expected {value!r})"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if invalid:
        raise ValueError(
            f"Rollout queue step {batch.step} has invalid manifest metadata: "
            + ", ".join(invalid)
        )

    policy_step = manifest.policy_step
    if (
        policy_step is None
        or policy_step < minimum_policy_step
        or policy_step > maximum_policy_step
    ):
        raise ValueError(
            f"Rollout queue step {batch.step} has policy_step={policy_step!r}; "
            f"optimizer step {optimizer_step} requires a policy step in "
            f"[{minimum_policy_step}, {maximum_policy_step}]."
        )
    return manifest


def write_claim(step_dir: Path, claim: ClaimRecord) -> Path:
    return _write_record(step_dir / CLAIM_FILENAME, claim)


def read_claim(step_dir: Path) -> ClaimRecord | None:
    return _read_record(step_dir / CLAIM_FILENAME, ClaimRecord)


def write_consumed(step_dir: Path, consumed: ConsumedRecord) -> Path:
    return _write_record(step_dir / CONSUMED_FILENAME, consumed)


def read_consumed(step_dir: Path) -> ConsumedRecord | None:
    return _read_record(step_dir / CONSUMED_FILENAME, ConsumedRecord)


def prune_consumed_rollout_batches(
    output_dir: Path,
    config: RLTransportConfig,
    *,
    keep_last: int | None,
) -> list[Path]:
    """Remove old consumed queue batches while preserving recent audit samples."""
    if keep_last is None:
        return []
    queue_dir = resolve_queue_dir(output_dir, config)
    if not queue_dir.exists():
        return []

    consumed: list[tuple[int, Path]] = []
    for candidate in queue_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(STEP_DIR_PREFIX):
            continue
        try:
            step = int(candidate.name.removeprefix(STEP_DIR_PREFIX))
        except ValueError:
            continue
        if (candidate / CONSUMED_FILENAME).exists():
            consumed.append((step, candidate))

    removed: list[Path] = []
    for step, path in sorted(consumed)[:-keep_last]:
        shutil.rmtree(path)
        materialized = queue_dir / f"materialized-step-{step:06d}.jsonl"
        materialized.unlink(missing_ok=True)
        removed.append(path)
    return removed


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
    manifest = _read_manifest_best_effort(batch)
    append_event_best_effort(
        events_dir,
        QueueEvent(
            time=claim.claimed_at,
            kind="rollout_claimed",
            queue_step=batch.step,
            optimizer_step=(None if manifest is None else manifest.optimizer_step),
            policy_step=None if manifest is None else manifest.policy_step,
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
            optimizer_step=(None if manifest is None else manifest.optimizer_step),
            policy_step=None if manifest is None else manifest.policy_step,
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
    manifest = _read_manifest_best_effort(batch)
    append_event_best_effort(
        events_dir,
        QueueEvent(
            time=consumed.consumed_at,
            kind="rollout_consumed",
            queue_step=batch.step,
            optimizer_step=(None if manifest is None else manifest.optimizer_step),
            policy_step=None if manifest is None else manifest.policy_step,
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
            optimizer_step=(
                trainer_step_after if manifest is None else manifest.optimizer_step
            ),
            policy_step=None if manifest is None else manifest.policy_step,
            details={
                "consumer_id": consumed.consumer_id,
                "trainer_step_before": trainer_step_before,
                "optimizer_step_completed": optimizer_step_completed,
            },
        ),
    )
    return consumed


def _read_manifest_best_effort(batch: RolloutBatch) -> RolloutManifest | None:
    try:
        return read_manifest(batch.step_dir)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to read rollout manifest for lifecycle event step %s: %s",
            batch.step,
            exc,
        )
        return None


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


logger = logging.getLogger(__name__)


_COPY_CHUNK_BYTES = 1024 * 1024


ItemT = TypeVar("ItemT")


def resolve_queue_dir(output_dir: Path, config: RLTransportConfig) -> Path:
    if config.queue_dir is not None:
        return Path(config.queue_dir)
    return output_dir / "rollouts"


def get_step_dir(queue_dir: Path, step: int) -> Path:
    return queue_dir / f"{STEP_DIR_PREFIX}{step:06d}"


def resolve_policy_dir(output_dir: Path, config: RLPolicyTransferConfig) -> Path:
    if config.policy_dir is not None:
        return Path(config.policy_dir)
    return output_dir / "policies"


def get_policy_step_dir(policy_dir: Path, step: int) -> Path:
    return policy_dir / f"{STEP_DIR_PREFIX}{step:06d}"


def parse_step(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith(STEP_DIR_PREFIX):
        return None
    try:
        return int(path.name.removeprefix(STEP_DIR_PREFIX))
    except ValueError:
        return None


def _copy_payload(source_path: Path, target_path: Path) -> tuple[int, float]:
    started_at = time.monotonic()
    payload_bytes = 0
    with source_path.open("rb") as source, target_path.open("wb") as target:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            payload_bytes += len(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    return payload_bytes, time.monotonic() - started_at


def _wait_for_item(
    find_item: Callable[[], ItemT | None],
    *,
    poll_interval_seconds: float,
    idle_timeout_seconds: float | None,
    timeout_message: str,
) -> tuple[ItemT, float]:
    started_at = time.monotonic()
    deadline = (
        None if idle_timeout_seconds is None else started_at + idle_timeout_seconds
    )
    while (item := find_item()) is None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(timeout_message)
        time.sleep(poll_interval_seconds)
    return item, time.monotonic() - started_at


def _available_steps(root: Path, is_stable: Callable[[Path], bool]) -> list[int]:
    if not root.exists():
        return []
    return sorted(
        step
        for candidate in root.iterdir()
        if (step := parse_step(candidate)) is not None and is_stable(candidate)
    )


def _is_stable_dir(path: Path) -> bool:
    return path.is_dir() and (path / STABLE_BATCH_MARKER).exists()


def _record_received(
    events_dir: Path | None,
    *,
    kind: str,
    subsystem: str,
    step: int,
    consumer_id: str,
    mode: str,
    payload_bytes: int | None,
    wait_seconds: float,
    queue_step: int | None = None,
    optimizer_step: int | None = None,
    policy_step: int | None = None,
) -> None:
    if events_dir is None:
        return
    details = {
        "mode": mode,
        "payload_bytes": payload_bytes,
        "wait_seconds": wait_seconds,
    }
    append_event_best_effort(
        events_dir,
        QueueEvent(
            time=utc_now(),
            kind=kind,
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            policy_step=policy_step,
            consumer_id=consumer_id,
            details=details,
        ),
    )
    append_trace_event_best_effort(
        _trace_output_dir(events_dir),
        make_trace_event(
            subsystem=subsystem,
            event=kind,
            step=step,
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            policy_step=policy_step,
            details={"consumer_id": consumer_id, **details},
        ),
    )


class FileSystemRolloutSender:
    def __init__(self, output_dir: Path, config: RLTransportConfig) -> None:
        self.config = config
        self.output_dir = output_dir
        self.queue_dir = resolve_queue_dir(output_dir, config)

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
        reward_mean: float | None = None,
        producer_id: str | None = None,
        events_dir: Path | None = None,
    ) -> RolloutBatch:
        step_dir = get_step_dir(self.queue_dir, step)
        existing = self.stable_batch(step)
        if existing is not None:
            raise FileExistsError(
                f"Rollout queue step {step} is already stable at "
                f"'{existing.step_dir}' and cannot be overwritten."
            )
        step_dir.mkdir(parents=True, exist_ok=True)
        target_path = step_dir / self.config.rollout_filename
        tmp_path = step_dir / f"{self.config.rollout_filename}.tmp"
        payload_bytes, transfer_seconds = _copy_payload(Path(source_path), tmp_path)
        tmp_path.replace(target_path)
        metadata_provided = any(
            value is not None
            for value in (
                optimizer_step,
                chunk_index,
                policy_step,
                rows,
                tokens,
                reward_mean,
                producer_id,
                events_dir,
            )
        )
        if metadata_provided:
            created_at = utc_now()
            producer_id = producer_id or process_identity("rl-inference")
            manifest = RolloutManifest(
                format_version=1,
                queue_step=step,
                optimizer_step=optimizer_step,
                chunk_index=chunk_index,
                policy_step=policy_step,
                rows=rows,
                tokens=tokens,
                reward_mean=reward_mean,
                producer_id=producer_id,
                created_at=created_at,
                payload_bytes=payload_bytes,
                transfer_seconds=transfer_seconds,
            )
            try:
                write_manifest(step_dir, manifest)
            except Exception as exc:  # pragma: no cover - fail-open observability
                logger.warning(
                    "Failed to write rollout manifest for step %s: %s", step, exc
                )
            append_event_best_effort(
                events_dir or (self.output_dir / "events"),
                QueueEvent(
                    time=created_at,
                    kind="rollout_published",
                    queue_step=step,
                    optimizer_step=optimizer_step,
                    policy_step=policy_step,
                    producer_id=producer_id,
                    details={
                        "payload_bytes": payload_bytes,
                        "transfer_seconds": transfer_seconds,
                    },
                ),
            )
        (step_dir / STABLE_BATCH_MARKER).touch()
        return RolloutBatch(step=step, path=target_path, step_dir=step_dir)

    def stable_batch(self, step: int) -> RolloutBatch | None:
        """Return an immutable stable batch for idempotent scheduler resume."""
        step_dir = get_step_dir(self.queue_dir, step)
        target_path = step_dir / self.config.rollout_filename
        if not _is_stable_dir(step_dir) or not target_path.is_file():
            return None
        return RolloutBatch(step=step, path=target_path, step_dir=step_dir)


class FileSystemRolloutReceiver:
    def __init__(
        self,
        output_dir: Path,
        config: RLTransportConfig,
        *,
        start_step: int = 0,
        events_dir: Path | None = None,
        consumer_id: str | None = None,
    ) -> None:
        self.config = config
        self.queue_dir = resolve_queue_dir(output_dir, config)
        self.next_step = start_step
        self._consumed_steps: set[int] = set()
        self.events_dir = events_dir
        self.consumer_id = consumer_id

    def can_receive(self) -> bool:
        return self._stable_batch_for_step(self.next_step) is not None

    def receive(self) -> RolloutBatch:
        batch = self._stable_batch_for_step(self.next_step)
        if batch is None:
            raise FileNotFoundError(
                f"No stable rollout batch available for step {self.next_step}."
            )
        self._accept(batch, wait_seconds=0.0, mode="receive")
        return batch

    def wait(self) -> RolloutBatch:
        batch, wait_seconds = _wait_for_item(
            lambda: self._stable_batch_for_step(self.next_step),
            poll_interval_seconds=self.config.poll_interval_seconds,
            idle_timeout_seconds=self.config.idle_timeout_seconds,
            timeout_message=(
                f"Timed out waiting for rollout batch step {self.next_step} in "
                f"'{self.queue_dir}'."
            ),
        )
        self._accept(batch, wait_seconds=wait_seconds, mode="wait")
        return batch

    def wait_available(self) -> RolloutBatch:
        """Return the oldest currently stable unconsumed batch at or after next_step."""
        batch, wait_seconds = _wait_for_item(
            self._oldest_available_batch,
            poll_interval_seconds=self.config.poll_interval_seconds,
            idle_timeout_seconds=self.config.idle_timeout_seconds,
            timeout_message=(
                "Timed out waiting for any rollout batch at or after step "
                f"{self.next_step} in '{self.queue_dir}'."
            ),
        )
        self._mark_consumed(batch)
        self._record_received(
            batch,
            wait_seconds=wait_seconds,
            mode="wait_available",
        )
        return batch

    def available_steps(self) -> list[int]:
        return _available_steps(self.queue_dir, _is_stable_dir)

    def _stable_batch_for_step(self, step: int) -> RolloutBatch | None:
        step_dir = get_step_dir(self.queue_dir, step)
        if not _is_stable_dir(step_dir):
            return None
        batch_path = step_dir / self.config.rollout_filename
        if not batch_path.exists():
            return None
        return RolloutBatch(step=step, path=batch_path, step_dir=step_dir)

    def _oldest_available_batch(self) -> RolloutBatch | None:
        for step in self.available_steps():
            if step < self.next_step or step in self._consumed_steps:
                continue
            return self._stable_batch_for_step(step)
        return None

    def _mark_consumed(self, batch: RolloutBatch) -> None:
        self._consumed_steps.add(batch.step)
        while self.next_step in self._consumed_steps:
            self._consumed_steps.remove(self.next_step)
            self.next_step += 1
        if self.config.cleanup_consumed:
            marker = batch.step_dir / ".consumed"
            marker.touch()

    def _accept(self, batch: RolloutBatch, *, wait_seconds: float, mode: str) -> None:
        self.next_step += 1
        self._record_received(batch, wait_seconds=wait_seconds, mode=mode)
        if self.config.cleanup_consumed:
            (batch.step_dir / ".consumed").touch()

    def _record_received(
        self,
        batch: RolloutBatch,
        *,
        wait_seconds: float,
        mode: str,
    ) -> None:
        consumer_id = self.consumer_id or process_identity("rl-trainer")
        try:
            payload_bytes = batch.path.stat().st_size
        except OSError:
            payload_bytes = None
        manifest = _read_manifest_best_effort(batch)
        _record_received(
            self.events_dir,
            kind="rollout_received",
            subsystem="trainer",
            step=batch.step,
            queue_step=batch.step,
            optimizer_step=(None if manifest is None else manifest.optimizer_step),
            policy_step=None if manifest is None else manifest.policy_step,
            consumer_id=consumer_id,
            mode=mode,
            payload_bytes=payload_bytes,
            wait_seconds=wait_seconds,
        )


class FileSystemPolicyReceiver:
    def __init__(
        self,
        output_dir: Path,
        config: RLPolicyTransferConfig,
        *,
        start_step: int = 0,
        events_dir: Path | None = None,
        consumer_id: str | None = None,
    ) -> None:
        self.config = config
        self.policy_dir = resolve_policy_dir(output_dir, config)
        self.next_step = start_step
        self.events_dir = events_dir
        self.consumer_id = consumer_id

    def can_receive(self) -> bool:
        return self._stable_policy_for_step(self.next_step) is not None

    def receive(self) -> PolicySnapshot:
        snapshot = self._stable_policy_for_step(self.next_step)
        if snapshot is None:
            raise FileNotFoundError(
                f"No stable policy snapshot available for step {self.next_step}."
            )
        self._accept(snapshot, wait_seconds=0.0, mode="receive")
        return snapshot

    def wait(self) -> PolicySnapshot:
        snapshot, wait_seconds = _wait_for_item(
            lambda: self._stable_policy_for_step(self.next_step),
            poll_interval_seconds=self.config.poll_interval_seconds,
            idle_timeout_seconds=self.config.idle_timeout_seconds,
            timeout_message=(
                f"Timed out waiting for policy step {self.next_step} in "
                f"'{self.policy_dir}'."
            ),
        )
        self._accept(snapshot, wait_seconds=wait_seconds, mode="wait")
        return snapshot

    def wait_for_step(self, step: int) -> PolicySnapshot:
        self.next_step = step
        return self.wait()

    def available_steps(self) -> list[int]:
        return _available_steps(self.policy_dir, _is_stable_dir)

    def _stable_policy_for_step(self, step: int) -> PolicySnapshot | None:
        step_dir = get_policy_step_dir(self.policy_dir, step)
        if not _is_stable_dir(step_dir):
            return None
        return PolicySnapshot(step=step, step_dir=step_dir)

    def _record_received(
        self,
        snapshot: PolicySnapshot,
        *,
        wait_seconds: float,
        mode: str,
    ) -> None:
        consumer_id = self.consumer_id or process_identity("rl-inference")
        payload_bytes = _policy_artifact_bytes(snapshot.meta_path)
        _record_received(
            self.events_dir,
            kind="policy_received",
            subsystem="inference",
            step=snapshot.step,
            policy_step=snapshot.step,
            consumer_id=consumer_id,
            mode=mode,
            payload_bytes=payload_bytes,
            wait_seconds=wait_seconds,
        )

    def _accept(
        self,
        snapshot: PolicySnapshot,
        *,
        wait_seconds: float,
        mode: str,
    ) -> None:
        self.next_step += 1
        self._record_received(snapshot, wait_seconds=wait_seconds, mode=mode)


def _policy_artifact_bytes(metadata_path: Path) -> int | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    artifact = metadata.get("artifact") if isinstance(metadata, dict) else None
    value = artifact.get("bytes") if isinstance(artifact, dict) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _trace_output_dir(events_dir: Path | None) -> Path | None:
    return events_dir.parent if events_dir is not None else None


def publish_adapter_policy_snapshot(
    output_dir: Path,
    config: RLPolicyTransferConfig,
    adapter_path: Path,
    *,
    step: int = 0,
    metadata: dict[str, object] | None = None,
) -> Path:
    policy_dir = resolve_policy_dir(output_dir, config)
    step_dir = get_policy_step_dir(policy_dir, step)
    tmp_dir = step_dir.with_name(f"{step_dir.name}.tmp")
    adapter_path = Path(adapter_path)
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"Adapter snapshot '{adapter_path}' is missing adapter_model.safetensors."
        )
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if step_dir.exists():
        shutil.rmtree(step_dir)
    tmp_adapter_dir = tmp_dir / "adapter"
    tmp_adapter_dir.mkdir(parents=True, exist_ok=True)
    for source in adapter_path.iterdir():
        if source.is_file():
            shutil.copy2(source, tmp_adapter_dir / source.name)
    meta = {
        "format_version": 1,
        "step": step,
        "kind": "adapter",
        "source_adapter_path": str(adapter_path),
    }
    if metadata:
        meta.update(metadata)
    from wavelet.orchestrator.policy_metadata import adapter_artifact_metadata

    artifact = adapter_artifact_metadata(tmp_adapter_dir)
    if artifact is not None:
        meta["artifact"] = artifact
    (tmp_dir / POLICY_META_FILENAME).write_text(json.dumps(meta))
    (tmp_dir / STABLE_BATCH_MARKER).touch()
    tmp_dir.replace(step_dir)
    append_event_best_effort(
        output_dir / "events",
        QueueEvent(
            time=utc_now(),
            kind="policy_export_completed",
            policy_step=step,
        ),
    )
    return step_dir


def policy_lag(
    manifest: RolloutManifest,
    latest_policy_step: int | None,
) -> int | None:
    if latest_policy_step is None or manifest.policy_step is None:
        return None
    return latest_policy_step - manifest.policy_step


def event_rate(
    events_dir: Path,
    *,
    kind: str,
    window_seconds: float = 60.0,
    limit: int = 5000,
) -> tuple[float, int]:
    events, parse_errors = tail_events(events_dir, limit=limit)
    if not events:
        return 0.0, parse_errors
    latest_time = max((_parse_datetime(event.time) for event in events), default=None)
    if latest_time is None:
        return 0.0, parse_errors
    earliest_time = latest_time.timestamp() - window_seconds
    count = 0
    for event in events:
        parsed = _parse_datetime(event.time)
        if (
            event.kind == kind
            and parsed is not None
            and parsed.timestamp() >= earliest_time
        ):
            count += 1
    return count / window_seconds, parse_errors


def publish_rate(
    events_dir: Path, *, window_seconds: float = 60.0
) -> tuple[float, int]:
    return event_rate(
        events_dir,
        kind="rollout_published",
        window_seconds=window_seconds,
    )


def consume_rate(
    events_dir: Path, *, window_seconds: float = 60.0
) -> tuple[float, int]:
    return event_rate(
        events_dir,
        kind="rollout_consumed",
        window_seconds=window_seconds,
    )


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def scan_queue_dir(
    queue_dir: Path,
    *,
    events_dir: Path | None = None,
    latest_policy_step: int | None = None,
    stale_policy_lag: int | None = 2,
    abandoned_claim_age_seconds: float = 300.0,
    detail: bool = False,
    limit: int = 100,
) -> QueueSnapshot:
    now = datetime.now().astimezone()
    items: list[QueueItemSnapshot] = []
    candidates = queue_dir.iterdir() if queue_dir.exists() else ()
    for candidate in sorted(candidates, key=lambda path: path.name):
        queue_step = parse_step(candidate)
        if queue_step is None:
            continue
        items.append(
            _scan_queue_item(
                candidate,
                queue_step=queue_step,
                latest_policy_step=latest_policy_step,
                stale_policy_lag=stale_policy_lag,
                abandoned_claim_age_seconds=abandoned_claim_age_seconds,
                now=now,
            )
        )

    status_counts = Counter(item.status for item in items)
    known_consumed = {item.queue_step for item in items if item.status == "consumed"}
    next_expected = 0
    while next_expected in known_consumed:
        next_expected += 1

    event_parse_errors = (
        tail_events(events_dir, limit=5000)[1] if events_dir is not None else 0
    )
    visible_items = items[-max(limit, 0) :] if detail else []
    return QueueSnapshot(
        ready_count=status_counts["ready"],
        claimed_count=status_counts["claimed"],
        consumed_count=status_counts["consumed"],
        incomplete_count=status_counts["incomplete"],
        unknown_count=sum(1 for item in items if item.parse_errors),
        stale_ready_count=status_counts["stale"],
        abandoned_claim_count=status_counts["abandoned_claim"],
        oldest_ready_age_seconds=_oldest_age_seconds(
            item for item in items if item.status in {"ready", "stale"}
        ),
        latest_queue_step=max((item.queue_step for item in items), default=None),
        latest_consumed_queue_step=max(known_consumed, default=None),
        next_expected_trainer_queue_step=next_expected,
        event_parse_error_count=event_parse_errors,
        items=visible_items,
    )


def scan_policy_dir(policy_dir: Path) -> PolicyQueueSnapshot:
    if not policy_dir.exists():
        return PolicyQueueSnapshot(
            latest_exported_step=None,
            steps=[],
            incomplete_steps=[],
        )
    steps: list[int] = []
    incomplete_steps: list[int] = []
    for candidate in sorted(policy_dir.iterdir(), key=lambda path: path.name):
        step = parse_step(candidate)
        if step is None:
            continue
        if (candidate / STABLE_BATCH_MARKER).exists():
            steps.append(step)
        else:
            incomplete_steps.append(step)
    return PolicyQueueSnapshot(
        latest_exported_step=max(steps, default=None),
        steps=steps,
        incomplete_steps=incomplete_steps,
    )


def build_queue_report(
    *,
    queue_dir: Path,
    policy_dir: Path,
    events_dir: Path | None = None,
    detail: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    events_dir = events_dir or queue_dir.parent / "events"
    policy_snapshot = scan_policy_dir(policy_dir)
    queue_snapshot = scan_queue_dir(
        queue_dir,
        events_dir=events_dir,
        latest_policy_step=policy_snapshot.latest_exported_step,
        detail=detail,
        limit=limit,
    )
    publish_per_second, publish_parse_errors = publish_rate(events_dir)
    consume_per_second, consume_parse_errors = consume_rate(events_dir)
    summary = asdict(queue_snapshot)
    summary.pop("items", None)
    return {
        "summary": summary,
        "policy": asdict(policy_snapshot),
        "rates": {
            "rollouts_published_per_second": publish_per_second,
            "rollouts_consumed_per_second": consume_per_second,
        },
        "errors": {
            "event_parse_error_count": max(
                queue_snapshot.event_parse_error_count,
                publish_parse_errors,
                consume_parse_errors,
            ),
        },
        "items": [asdict(item) for item in queue_snapshot.items],
    }


def _scan_queue_item(
    step_dir: Path,
    *,
    queue_step: int,
    latest_policy_step: int | None,
    stale_policy_lag: int | None,
    abandoned_claim_age_seconds: float,
    now: datetime,
) -> QueueItemSnapshot:
    parse_errors: list[str] = []
    stable = (step_dir / STABLE_BATCH_MARKER).exists()
    manifest = _read_optional(read_manifest, step_dir, "manifest", parse_errors)
    claim = _read_optional(read_claim, step_dir, "claim", parse_errors)
    consumed = _read_optional(read_consumed, step_dir, "consumed", parse_errors)
    age_seconds = _age_seconds(step_dir, now=now)

    if consumed is not None:
        status = "consumed"
    elif claim is not None:
        status = "claimed"
        claim_age = _timestamp_age_seconds(claim.claimed_at, now=now)
        if claim_age is not None and claim_age > abandoned_claim_age_seconds:
            status = "abandoned_claim"
    elif stable:
        status = "ready"
        if (
            manifest is not None
            and stale_policy_lag is not None
            and (lag := policy_lag(manifest, latest_policy_step)) is not None
            and lag > stale_policy_lag
        ):
            status = "stale"
    else:
        status = "incomplete"

    return QueueItemSnapshot(
        queue_step=queue_step,
        status=status,
        stable=stable,
        manifest=manifest,
        claim=claim,
        consumed=consumed,
        age_seconds=age_seconds,
        parse_errors=parse_errors,
    )


def _read_optional(read_fn, step_dir: Path, label: str, errors: list[str]):
    try:
        return read_fn(step_dir)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"{label}: {exc}")
        return None


def _oldest_age_seconds(items: Iterable[QueueItemSnapshot]) -> float | None:
    ages = [item.age_seconds for item in items if item.age_seconds is not None]
    return max(ages) if ages else None


def _age_seconds(path: Path, *, now: datetime) -> float | None:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except OSError:
        return None
    return max(0.0, (now - modified).total_seconds())


def _timestamp_age_seconds(value: str, *, now: datetime) -> float | None:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.astimezone()
    return max(0.0, (now - timestamp).total_seconds())


@dataclass
class RolloutChunkAccumulator:
    """Accumulate streaming queue chunks until they are safe to load or step."""

    accumulated_rows: int = 0
    accumulated_chunks: int = 0
    accumulated_loss_scale: float = 0.0
    chunk_index: int = 0
    pending_paths: list[Path] = field(default_factory=list)
    pending_batches: list[RolloutBatch] = field(default_factory=list)  # noqa: F405
    pending_rows: int = 0

    def buffer(self, batch: RolloutBatch | Path, rows: int) -> None:  # noqa: F405
        self.chunk_index += 1
        if isinstance(batch, RolloutBatch):  # noqa: F405
            self.pending_paths.append(batch.path)
            self.pending_batches.append(batch)
        else:
            self.pending_paths.append(batch)
        self.pending_rows += rows

    def should_load(self, *, min_rows: int) -> bool:
        return self.pending_rows >= min_rows or (
            self.accumulated_rows > 0 and self.pending_rows > 0
        )

    def drain_pending_batches(
        self,
    ) -> tuple[list[Path], list[RolloutBatch], int]:  # noqa: F405
        paths = self.pending_paths
        batches = self.pending_batches
        loaded_chunks = len(paths)
        self.pending_paths = []
        self.pending_batches = []
        self.pending_rows = 0
        return paths, batches, loaded_chunks

    def drain_pending_paths(self) -> tuple[list[Path], int]:
        paths, _, loaded_chunks = self.drain_pending_batches()
        return paths, loaded_chunks

    def mark_loaded(
        self,
        *,
        rows: int,
        chunks: int,
        loss_scale: float | None,
    ) -> None:
        self.accumulated_rows += rows
        self.accumulated_chunks += chunks
        if loss_scale is not None:
            self.accumulated_loss_scale += float(loss_scale)

    def should_step(self, *, chunks_per_step: int) -> bool:
        return self.accumulated_chunks >= chunks_per_step

    def reset_after_optimizer_step(self) -> None:
        self.accumulated_rows = 0
        self.accumulated_chunks = 0
        self.accumulated_loss_scale = 0.0

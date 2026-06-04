from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

from wavelet.configs.rl_config import RLPolicyTransferConfig, RLTransportConfig
from wavelet.orchestrator.queue.events import append_event_best_effort
from wavelet.orchestrator.queue.lifecycle import (
    process_identity,
    utc_now,
    write_manifest,
)
from wavelet.orchestrator.queue.types import (
    POLICY_META_FILENAME,
    STABLE_BATCH_MARKER,
    STEP_DIR_PREFIX,
    PolicySnapshot,
    QueueEvent,
    RolloutBatch,
    RolloutManifest,
)


logger = logging.getLogger(__name__)
_COPY_CHUNK_BYTES = 1024 * 1024


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


def _parse_step(path: Path) -> int | None:
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
                logger.warning("Failed to write rollout manifest for step %s: %s", step, exc)
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


class FileSystemRolloutReceiver:
    def __init__(
        self,
        output_dir: Path,
        config: RLTransportConfig,
        *,
        start_step: int = 0,
    ) -> None:
        self.config = config
        self.queue_dir = resolve_queue_dir(output_dir, config)
        self.next_step = start_step
        self._consumed_steps: set[int] = set()

    def can_receive(self) -> bool:
        return self._stable_batch_for_step(self.next_step) is not None

    def receive(self) -> RolloutBatch:
        batch = self._stable_batch_for_step(self.next_step)
        if batch is None:
            raise FileNotFoundError(
                f"No stable rollout batch available for step {self.next_step}."
            )
        self.next_step += 1
        if self.config.cleanup_consumed:
            marker = batch.step_dir / ".consumed"
            marker.touch()
        return batch

    def wait(self) -> RolloutBatch:
        deadline = None
        if self.config.idle_timeout_seconds is not None:
            deadline = time.monotonic() + self.config.idle_timeout_seconds
        while True:
            batch = self._stable_batch_for_step(self.next_step)
            if batch is not None:
                self.next_step += 1
                if self.config.cleanup_consumed:
                    marker = batch.step_dir / ".consumed"
                    marker.touch()
                return batch
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for rollout batch step {self.next_step} in "
                    f"'{self.queue_dir}'."
                )
            time.sleep(self.config.poll_interval_seconds)

    def wait_available(self) -> RolloutBatch:
        """Return the oldest currently stable unconsumed batch at or after next_step."""
        deadline = None
        if self.config.idle_timeout_seconds is not None:
            deadline = time.monotonic() + self.config.idle_timeout_seconds
        while True:
            batch = self._oldest_available_batch()
            if batch is not None:
                self._mark_consumed(batch)
                return batch
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for any rollout batch at or after step "
                    f"{self.next_step} in '{self.queue_dir}'."
                )
            time.sleep(self.config.poll_interval_seconds)

    def available_steps(self) -> list[int]:
        steps: list[int] = []
        if not self.queue_dir.exists():
            return steps
        for candidate in self.queue_dir.iterdir():
            step = _parse_step(candidate)
            if step is None:
                continue
            if not self._is_stable_step_dir(candidate):
                continue
            steps.append(step)
        return sorted(steps)

    def _stable_batch_for_step(self, step: int) -> RolloutBatch | None:
        step_dir = get_step_dir(self.queue_dir, step)
        if not self._is_stable_step_dir(step_dir):
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

    @staticmethod
    def _is_stable_step_dir(step_dir: Path) -> bool:
        return step_dir.is_dir() and (step_dir / STABLE_BATCH_MARKER).exists()


class FileSystemPolicyReceiver:
    def __init__(
        self,
        output_dir: Path,
        config: RLPolicyTransferConfig,
        *,
        start_step: int = 0,
    ) -> None:
        self.config = config
        self.policy_dir = resolve_policy_dir(output_dir, config)
        self.next_step = start_step

    def can_receive(self) -> bool:
        return self._stable_policy_for_step(self.next_step) is not None

    def receive(self) -> PolicySnapshot:
        snapshot = self._stable_policy_for_step(self.next_step)
        if snapshot is None:
            raise FileNotFoundError(
                f"No stable policy snapshot available for step {self.next_step}."
            )
        self.next_step += 1
        return snapshot

    def wait(self) -> PolicySnapshot:
        deadline = None
        if self.config.idle_timeout_seconds is not None:
            deadline = time.monotonic() + self.config.idle_timeout_seconds
        while True:
            snapshot = self._stable_policy_for_step(self.next_step)
            if snapshot is not None:
                self.next_step += 1
                return snapshot
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for policy step {self.next_step} in "
                    f"'{self.policy_dir}'."
                )
            time.sleep(self.config.poll_interval_seconds)

    def wait_for_step(self, step: int) -> PolicySnapshot:
        self.next_step = step
        return self.wait()

    def available_steps(self) -> list[int]:
        steps: list[int] = []
        if not self.policy_dir.exists():
            return steps
        for candidate in self.policy_dir.iterdir():
            step = _parse_step(candidate)
            if step is None:
                continue
            if not self._is_stable_policy_dir(candidate):
                continue
            steps.append(step)
        return sorted(steps)

    def _stable_policy_for_step(self, step: int) -> PolicySnapshot | None:
        step_dir = get_policy_step_dir(self.policy_dir, step)
        if not self._is_stable_policy_dir(step_dir):
            return None
        return PolicySnapshot(step=step, step_dir=step_dir)

    @staticmethod
    def _is_stable_policy_dir(step_dir: Path) -> bool:
        return step_dir.is_dir() and (step_dir / STABLE_BATCH_MARKER).exists()


def publish_adapter_policy_snapshot(
    output_dir: Path,
    config: RLPolicyTransferConfig,
    adapter_path: Path,
    *,
    step: int = 0,
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

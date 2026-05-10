from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from wavelet.configs.rl_config import RLPolicyTransferConfig, RLTransportConfig


STEP_DIR_PREFIX = "step-"
STABLE_BATCH_MARKER = "STABLE"
POLICY_META_FILENAME = "policy.json"


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


class FileSystemRolloutSender:
    def __init__(self, output_dir: Path, config: RLTransportConfig) -> None:
        self.config = config
        self.queue_dir = resolve_queue_dir(output_dir, config)

    def publish(self, source_path: Path, *, step: int) -> RolloutBatch:
        step_dir = get_step_dir(self.queue_dir, step)
        step_dir.mkdir(parents=True, exist_ok=True)
        target_path = step_dir / self.config.rollout_filename
        tmp_path = step_dir / f"{self.config.rollout_filename}.tmp"
        tmp_path.write_bytes(Path(source_path).read_bytes())
        tmp_path.replace(target_path)
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
    return step_dir

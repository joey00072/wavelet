from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Self


def json_value(value: Any) -> Any:
    """Convert verifier data structurally; never persist object reprs."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, BaseException):
        return str(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Evaluation data contains unsupported {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        default=json_value,
        allow_nan=False,
        separators=(",", ":"),
    )


def evaluation_signature(**settings: Any) -> str:
    """Identify the exact policy, inputs, environment and sampling plan."""
    return hashlib.sha256(canonical_json(settings).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish a durable JSON object without exposing partial writes."""
    encoded = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EvaluationJournal:
    """Single-writer journal of atomically committed episode results.

    Use as a context manager. Uncommitted .tmp files are ignored on resume;
    malformed committed files stop the run rather than silently losing results.
    """

    def __init__(self, path: Path, *, signature: str, resume: bool = True) -> None:
        self.path = path
        self.signature = signature
        self.resume = resume
        self.rows: dict[str, dict[str, Any]] = {}
        self._lock: Any = None
        self.elapsed_seconds = 0.0
        self._started_at = 0.0
        self._previous_seconds = 0.0

    def __enter__(self) -> Self:
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = (self.path / "writer.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.path / "manifest.json"
            if manifest.exists():
                if not self.resume:
                    raise FileExistsError(
                        "Evaluation already exists; set eval.resume=true or use a "
                        "new output directory."
                    )
                if json.loads(manifest.read_text()) != {
                    "format_version": 1,
                    "signature": self.signature,
                }:
                    raise ValueError("Evaluation settings do not match journal")
            else:
                atomic_json(
                    manifest, {"format_version": 1, "signature": self.signature}
                )
            timing_path = self.path / "timing.json"
            if timing_path.exists():
                self._previous_seconds = float(
                    json.loads(timing_path.read_text())["seconds"]
                )
                if (
                    not math.isfinite(self._previous_seconds)
                    or self._previous_seconds < 0
                ):
                    raise ValueError("Invalid evaluation journal timing")
            self._started_at = time.monotonic()
            for record in self.path.glob("episode-*.json"):
                payload = json.loads(record.read_text(encoding="utf-8"))
                canonical_json(payload)
                if (
                    not isinstance(payload, dict)
                    or not isinstance(payload.get("key"), str)
                    or not isinstance(payload.get("output"), dict)
                ):
                    raise ValueError(  # noqa: TRY004 - malformed journal data
                        f"Invalid evaluation journal record: {record}"
                    )
                key = payload["key"]
                if record.name != self._record_name(key):
                    raise ValueError(f"Evaluation journal identity mismatch: {record}")
                self.rows[key] = payload["output"]
        except BaseException:
            self._lock.close()
            self._lock = None
            raise
        return self

    def __exit__(self, *_args: object) -> None:
        if self._lock is not None:
            try:
                self._save_timing()
            finally:
                self._lock.close()
                self._lock = None

    def completed(self, index: int | str) -> dict[str, Any] | None:
        row = self.rows.get(str(index))
        if row is None or row.get("error") is not None or "reward" not in row:
            return None
        return dict(row)

    def record(self, index: int | str, output: dict[str, Any]) -> dict[str, Any]:
        if self._lock is None:
            raise RuntimeError("Evaluation journal must be opened as a context manager")
        normalized = json.loads(canonical_json(output))
        key = str(index)
        atomic_json(
            self.path / self._record_name(key), {"key": key, "output": normalized}
        )
        self.rows[key] = normalized
        self._save_timing()
        return dict(normalized)

    def _save_timing(self) -> None:
        self.elapsed_seconds = (
            self._previous_seconds + time.monotonic() - self._started_at
        )
        atomic_json(self.path / "timing.json", {"seconds": self.elapsed_seconds})

    @staticmethod
    def _record_name(key: str) -> str:
        return "episode-" + hashlib.sha256(key.encode()).hexdigest() + ".json"


def evaluation_task_keys(examples: list[dict[str, Any]]) -> list[str]:
    """Stable content identity; duplicate tasks retain distinct occurrences."""
    occurrences: dict[str, int] = {}
    keys = []
    for index, example in enumerate(examples):
        identity = dict(example)
        identity.setdefault("example_id", identity.get("id", index))
        digest = evaluation_signature(example=identity)
        occurrence = occurrences.get(digest, 0)
        keys.append(f"{digest}:{occurrence}")
        occurrences[digest] = occurrence + 1
    return keys


def compute_eval_policy_step(
    *,
    policy_step: int,
    last_eval_step: int,
    interval: int,
    eval_base_model: bool = True,
) -> int | None:
    if policy_step <= last_eval_step:
        return None
    highest_interval_step = (policy_step // interval) * interval
    if highest_interval_step <= last_eval_step:
        return None
    if highest_interval_step == 0:
        if policy_step == 0 and eval_base_model and last_eval_step < 0:
            return 0
        return None
    return highest_interval_step


def pass_at_k(rewards: list[float]) -> dict[str, float]:
    """Return unbiased at-least-one and all-correct metrics for binary rewards."""
    n = len(rewards)
    c = sum(reward == 1.0 for reward in rewards)
    if n == 0:
        return {}
    ks = [2**index for index in range(n.bit_length())]
    return {
        key: value
        for k in ks
        for key, value in (
            (f"pass@{k}", _pass_at_k(n, c, k)),
            (f"pass^{k}", _pass_power_k(n, c, k)),
        )
    }


def _pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _pass_power_k(n: int, c: int, k: int) -> float:
    return math.comb(c, k) / math.comb(n, k)

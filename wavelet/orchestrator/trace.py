from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRACE_DIRNAME = "traces"
TRACE_FILENAME = "trace.jsonl"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    format_version: int
    timestamp: str
    subsystem: str
    event: str
    step: int | None = None
    queue_step: int | None = None
    optimizer_step: int | None = None
    policy_step: int | None = None
    task: str | None = None
    harness: str | None = None
    rollout_id: str | None = None
    details: dict[str, Any] | None = None


def make_trace_event(
    *,
    subsystem: str,
    event: str,
    step: int | None = None,
    queue_step: int | None = None,
    optimizer_step: int | None = None,
    policy_step: int | None = None,
    task: str | None = None,
    harness: str | None = None,
    rollout_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> TraceEvent:
    return TraceEvent(
        format_version=1,
        timestamp=datetime.now(UTC).isoformat(),
        subsystem=subsystem,
        event=event,
        step=step,
        queue_step=queue_step,
        optimizer_step=optimizer_step,
        policy_step=policy_step,
        task=task,
        harness=harness,
        rollout_id=rollout_id,
        details=details,
    )


def trace_path(output_dir: Path, *, step: int | None = None) -> Path:
    if step is None:
        return output_dir / TRACE_DIRNAME / TRACE_FILENAME
    return output_dir / TRACE_DIRNAME / f"step-{step:06d}.jsonl"


def append_trace_event(output_dir: Path, event: TraceEvent) -> Path:
    path = trace_path(output_dir, step=event.step)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(event), sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    return path


def append_trace_event_best_effort(output_dir: Path | None, event: TraceEvent) -> None:
    if output_dir is None:
        return
    try:
        append_trace_event(output_dir, event)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        logger.warning("Failed to append trace event: %s", exc)

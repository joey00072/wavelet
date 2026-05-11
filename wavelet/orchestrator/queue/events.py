from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict
from pathlib import Path

from wavelet.orchestrator.queue.types import QUEUE_EVENT_FILENAME, QueueEvent


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
    lines: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(line)

    events: list[QueueEvent] = []
    parse_errors = 0
    for line in lines:
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                parse_errors += 1
                continue
            events.append(QueueEvent(**row))
        except (json.JSONDecodeError, TypeError):
            parse_errors += 1
    return events, parse_errors

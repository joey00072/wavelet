from __future__ import annotations

from datetime import datetime
from pathlib import Path

from wavelet.orchestrator.queue.events import tail_events
from wavelet.orchestrator.queue.types import RolloutManifest


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
        if event.kind == kind and parsed is not None and parsed.timestamp() >= earliest_time:
            count += 1
    return count / window_seconds, parse_errors


def publish_rate(events_dir: Path, *, window_seconds: float = 60.0) -> tuple[float, int]:
    return event_rate(
        events_dir,
        kind="rollout_published",
        window_seconds=window_seconds,
    )


def consume_rate(events_dir: Path, *, window_seconds: float = 60.0) -> tuple[float, int]:
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

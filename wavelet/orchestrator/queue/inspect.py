from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from wavelet.orchestrator.queue.events import tail_events
from wavelet.orchestrator.queue.filesystem import parse_step
from wavelet.orchestrator.queue.lifecycle import (
    read_claim,
    read_consumed,
    read_manifest,
)
from wavelet.orchestrator.queue.metrics import consume_rate, policy_lag, publish_rate
from wavelet.orchestrator.queue.types import (
    STABLE_BATCH_MARKER,
    PolicyQueueSnapshot,
    QueueItemSnapshot,
    QueueSnapshot,
)


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
    known_consumed = {
        item.queue_step for item in items if item.status == "consumed"
    }
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

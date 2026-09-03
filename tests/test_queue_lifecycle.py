from __future__ import annotations

import json
from pathlib import Path

from wavelet.orchestrator.queue import (
    ClaimRecord,
    ConsumedRecord,
    QueueEvent,
    RolloutBatch,
    RolloutManifest,
    append_event,
    read_claim,
    read_consumed,
    read_manifest,
    record_rollout_claim,
    record_rollout_consumed,
    tail_events,
    write_claim,
    write_consumed,
    write_manifest,
)


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = RolloutManifest(
        format_version=1,
        queue_step=3,
        optimizer_step=1,
        chunk_index=0,
        policy_step=1,
        rows=4,
        tokens=None,
        reward_mean=0.5,
        producer_id="producer",
        created_at="2026-05-10T00:00:00+00:00",
    )

    write_manifest(tmp_path, manifest)

    assert read_manifest(tmp_path) == manifest


def test_claim_and_consumed_round_trip(tmp_path: Path) -> None:
    claim = ClaimRecord(
        format_version=1,
        queue_step=3,
        consumer_id="trainer",
        trainer_step_before=1,
        claimed_at="2026-05-10T00:00:01+00:00",
    )
    consumed = ConsumedRecord(
        format_version=1,
        queue_step=3,
        consumer_id="trainer",
        trainer_step_before=1,
        trainer_step_after=2,
        optimizer_step_completed=True,
        consumed_at="2026-05-10T00:00:02+00:00",
    )

    write_claim(tmp_path, claim)
    write_consumed(tmp_path, consumed)

    assert read_claim(tmp_path) == claim
    assert read_consumed(tmp_path) == consumed


def test_missing_lifecycle_records_return_none(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) is None
    assert read_claim(tmp_path) is None
    assert read_consumed(tmp_path) is None


def test_event_append_and_tail(tmp_path: Path) -> None:
    append_event(
        tmp_path,
        QueueEvent(
            time="2026-05-10T00:00:00+00:00",
            kind="rollout_published",
            queue_step=1,
        ),
    )
    (tmp_path / "queue.jsonl").write_text(
        (tmp_path / "queue.jsonl").read_text(encoding="utf-8") + "{bad json\n",
        encoding="utf-8",
    )

    events, parse_errors = tail_events(tmp_path, limit=10)

    assert events[0].kind == "rollout_published"
    assert events[0].queue_step == 1
    assert parse_errors == 1


def test_lifecycle_json_is_compact_and_sorted(tmp_path: Path) -> None:
    manifest = RolloutManifest(
        format_version=1,
        queue_step=1,
        optimizer_step=None,
        chunk_index=None,
        policy_step=None,
        rows=None,
        tokens=None,
        reward_mean=None,
        producer_id=None,
        created_at="2026-05-10T00:00:00+00:00",
    )

    path = write_manifest(tmp_path, manifest)

    assert "\n" not in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["queue_step"] == 1


def test_rollout_claim_and_consume_write_traces(tmp_path: Path) -> None:
    step_dir = tmp_path / "rollouts" / "step-000003"
    step_dir.mkdir(parents=True)
    batch_path = step_dir / "rollouts.jsonl"
    batch_path.write_text("{}\n", encoding="utf-8")
    batch = RolloutBatch(step=3, path=batch_path, step_dir=step_dir)
    write_manifest(
        step_dir,
        RolloutManifest(
            format_version=1,
            queue_step=3,
            optimizer_step=1,
            chunk_index=0,
            policy_step=0,
            rows=1,
            tokens=None,
            reward_mean=None,
            producer_id="inference",
            created_at="2026-05-10T00:00:00+00:00",
        ),
    )

    record_rollout_claim(
        batch,
        trainer_step_before=1,
        consumer_id="trainer",
        events_dir=tmp_path / "events",
    )
    record_rollout_consumed(
        batch,
        trainer_step_before=1,
        trainer_step_after=2,
        optimizer_step_completed=True,
        consumer_id="trainer",
        events_dir=tmp_path / "events",
    )

    step_one_trace = json.loads(
        (tmp_path / "traces" / "step-000001.jsonl").read_text(encoding="utf-8")
    )
    step_two_trace = json.loads(
        (tmp_path / "traces" / "step-000002.jsonl").read_text(encoding="utf-8")
    )
    assert step_one_trace["event"] == "rollout_claimed"
    assert step_one_trace["queue_step"] == 3
    assert step_one_trace["optimizer_step"] == 1
    assert step_one_trace["policy_step"] == 0
    assert step_two_trace["event"] == "rollout_consumed"
    assert step_two_trace["optimizer_step"] == 1
    assert step_two_trace["policy_step"] == 0

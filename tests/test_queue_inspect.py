from __future__ import annotations

from pathlib import Path

from wavelet.orchestrator.queue import (
    ClaimRecord,
    ConsumedRecord,
    RolloutManifest,
    get_step_dir,
    scan_policy_dir,
    scan_queue_dir,
    write_claim,
    write_consumed,
    write_manifest,
)


def _stable_step(queue_dir: Path, step: int) -> Path:
    step_dir = get_step_dir(queue_dir, step)
    step_dir.mkdir(parents=True)
    (step_dir / "rollouts.jsonl").write_text("{}\n", encoding="utf-8")
    (step_dir / "STABLE").touch()
    return step_dir


def test_scan_queue_counts_ready_claimed_consumed_and_incomplete(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "rollouts"
    ready = _stable_step(queue_dir, 0)
    claimed = _stable_step(queue_dir, 1)
    consumed = _stable_step(queue_dir, 2)
    incomplete = get_step_dir(queue_dir, 3)
    incomplete.mkdir(parents=True)
    (incomplete / "rollouts.jsonl").write_text("{}\n", encoding="utf-8")
    write_manifest(
        ready,
        RolloutManifest(
            format_version=1,
            queue_step=0,
            optimizer_step=0,
            chunk_index=None,
            policy_step=0,
            rows=1,
            tokens=None,
            reward_mean=None,
            producer_id="producer",
            created_at="2026-05-10T00:00:00+00:00",
        ),
    )
    write_claim(
        claimed,
        ClaimRecord(
            format_version=1,
            queue_step=1,
            consumer_id="trainer",
            trainer_step_before=0,
            claimed_at="2026-05-10T00:00:00+00:00",
        ),
    )
    write_consumed(
        consumed,
        ConsumedRecord(
            format_version=1,
            queue_step=2,
            consumer_id="trainer",
            trainer_step_before=1,
            trainer_step_after=2,
            optimizer_step_completed=True,
            consumed_at="2026-05-10T00:00:00+00:00",
        ),
    )

    snapshot = scan_queue_dir(queue_dir, abandoned_claim_age_seconds=10**9)

    assert snapshot.ready_count == 1
    assert snapshot.claimed_count == 1
    assert snapshot.consumed_count == 1
    assert snapshot.incomplete_count == 1
    assert snapshot.latest_queue_step == 3
    assert snapshot.next_expected_trainer_queue_step == 0


def test_scan_queue_tolerates_old_format_without_manifest(tmp_path: Path) -> None:
    queue_dir = tmp_path / "rollouts"
    _stable_step(queue_dir, 0)

    snapshot = scan_queue_dir(queue_dir)

    assert snapshot.ready_count == 1
    assert snapshot.unknown_count == 0


def test_scan_queue_detects_stale_ready_and_abandoned_claim(tmp_path: Path) -> None:
    queue_dir = tmp_path / "rollouts"
    stale = _stable_step(queue_dir, 0)
    abandoned = _stable_step(queue_dir, 1)
    write_manifest(
        stale,
        RolloutManifest(
            format_version=1,
            queue_step=0,
            optimizer_step=0,
            chunk_index=None,
            policy_step=1,
            rows=1,
            tokens=None,
            reward_mean=None,
            producer_id="producer",
            created_at="2026-05-10T00:00:00+00:00",
        ),
    )
    write_claim(
        abandoned,
        ClaimRecord(
            format_version=1,
            queue_step=1,
            consumer_id="trainer",
            trainer_step_before=0,
            claimed_at="2026-05-10T00:00:00+00:00",
        ),
    )

    snapshot = scan_queue_dir(
        queue_dir,
        latest_policy_step=5,
        stale_policy_lag=2,
        abandoned_claim_age_seconds=0,
    )

    assert snapshot.stale_ready_count == 1
    assert snapshot.abandoned_claim_count == 1


def test_scan_policy_dir_lists_stable_and_incomplete_steps(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies"
    stable = policy_dir / "step-000001"
    stable.mkdir(parents=True)
    (stable / "STABLE").touch()
    incomplete = policy_dir / "step-000002"
    incomplete.mkdir()

    snapshot = scan_policy_dir(policy_dir)

    assert snapshot.latest_exported_step == 1
    assert snapshot.steps == [1]
    assert snapshot.incomplete_steps == [2]

from pathlib import Path

from wavelet.configs.rl_config import RLTransportConfig
from wavelet.orchestrator.envs import _prune_eval_rollout_sets
from wavelet.transport.policy import prune_policy_snapshots
from wavelet.transport.queue import (
    CONSUMED_FILENAME,
    STABLE_BATCH_MARKER,
    get_policy_step_dir,
    prune_consumed_rollout_batches,
)


def _stable_policy_dir(root: Path, step: int) -> Path:
    path = get_policy_step_dir(root, step)
    path.mkdir(parents=True)
    (path / STABLE_BATCH_MARKER).touch()
    return path


def test_prune_policy_snapshots_keeps_latest_stable_steps(tmp_path) -> None:
    stable = [_stable_policy_dir(tmp_path, step) for step in (1, 3, 2)]
    incomplete = get_policy_step_dir(tmp_path, 0)
    incomplete.mkdir()

    removed = prune_policy_snapshots(tmp_path, keep_last=2)

    assert removed == [stable[0]]
    assert not stable[0].exists()
    assert stable[1].exists()
    assert stable[2].exists()
    assert incomplete.exists()


def test_prune_eval_rollout_sets_keeps_latest_steps(tmp_path) -> None:
    eval_dir = tmp_path / "evals"
    paths = []
    for step in (100, 300, 200):
        path = eval_dir / f"step-{step:06d}"
        path.mkdir(parents=True)
        paths.append(path)
    unrelated = eval_dir / "notes"
    unrelated.mkdir()

    removed = _prune_eval_rollout_sets(eval_dir, keep_last=2)

    assert removed == [paths[0]]
    assert not paths[0].exists()
    assert paths[1].exists()
    assert paths[2].exists()
    assert unrelated.exists()


def test_prune_consumed_rollout_batches_keeps_recent_audit_batches(tmp_path) -> None:
    queue_dir = tmp_path / "rollouts"
    paths = []
    for step in (0, 2, 1):
        path = queue_dir / f"step-{step:06d}"
        path.mkdir(parents=True)
        (path / CONSUMED_FILENAME).write_text("{}", encoding="utf-8")
        (queue_dir / f"materialized-step-{step:06d}.jsonl").touch()
        paths.append(path)

    removed = prune_consumed_rollout_batches(
        tmp_path,
        RLTransportConfig(),
        keep_last=2,
    )

    assert removed == [paths[0]]
    assert not paths[0].exists()
    assert not (queue_dir / "materialized-step-000000.jsonl").exists()
    assert paths[1].exists()
    assert paths[2].exists()

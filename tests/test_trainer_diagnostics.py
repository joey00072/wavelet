from __future__ import annotations

import json
from pathlib import Path

from wavelet.configs.rl_config import RLConfig
from wavelet.entrypoints.rl_trainer_debug import main as trainer_debug_main
from wavelet.trainer.diagnostics import inspect_rollout_batch


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_inspect_rollout_batch_reports_alignment_summary(tmp_path: Path) -> None:
    rollout_path = _write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {
                "input_ids": [1, 2, 3],
                "target_ids": [2, 3, 4],
                "loss_mask": [False, True, True],
                "inference_logprobs": [-0.2, -0.3],
                "teacher_logprobs": [-0.1, -0.2],
                "reward": 1.0,
                "metadata": {"example_id": "ex-1", "trajectory_id": "tr-1"},
            }
        ],
    )

    report = inspect_rollout_batch(RLConfig(), rollout_path=rollout_path)

    assert report["ok"] is True
    assert report["rows_scanned"] == 1
    assert report["summary"]["total_tokens"] == 3
    assert report["summary"]["trainable_tokens"] == 2
    assert report["summary"]["rows_with_inference_logprobs"] == 1
    assert report["samples"][0]["example_id"] == "ex-1"


def test_inspect_rollout_batch_reports_logprob_mismatch(tmp_path: Path) -> None:
    rollout_path = _write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {
                "input_ids": [1, 2, 3],
                "target_ids": [2, 3, 4],
                "loss_mask": [False, True, True],
                "inference_logprobs": [-0.2],
            }
        ],
    )

    report = inspect_rollout_batch(RLConfig(), rollout_path=rollout_path)

    assert report["ok"] is False
    assert report["errors"][0]["field"] == "inference_logprobs"


def test_trainer_debug_inspect_outputs_json(tmp_path: Path, capsys) -> None:
    rollout_path = _write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {
                "input_ids": [1, 2],
                "target_ids": [2, 3],
                "loss_mask": [False, True],
                "inference_logprobs": [-0.2],
            }
        ],
    )

    assert trainer_debug_main(["inspect", "--rollout-path", str(rollout_path), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["summary"]["trainable_tokens"] == 1

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.entrypoints.rl_debug import main as debug_main
from wavelet.trainer.diagnostics import (
    build_runtime_parity_report,
    export_rollout_token_debug,
    inspect_rollout_batch,
)


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

    assert (
        debug_main(["trainer", "inspect", "--rollout-path", str(rollout_path), "--json"])
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["summary"]["trainable_tokens"] == 1


def test_export_rollout_token_debug_writes_compact_jsonl(tmp_path: Path) -> None:
    rollout_path = _write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {
                "input_ids": [1, 2, 3],
                "target_ids": [2, 3, 4],
                "loss_mask": [False, True, True],
                "inference_logprobs": [-0.2, -0.3],
                "teacher_logprobs": [-0.1, -0.2],
                "temperatures": [0.7, 0.7],
                "advantage": 0.5,
                "reward": 1.0,
                "metadata": {
                    "example_id": "ex-1",
                    "trajectory_id": "tr-1",
                    "rollout_key": "rk-1",
                },
            }
        ],
    )
    write_path = tmp_path / "debug" / "tokens.jsonl"

    report = export_rollout_token_debug(
        RLConfig(),
        rollout_path=rollout_path,
        write_path=write_path,
    )

    rows = [json.loads(line) for line in write_path.read_text(encoding="utf-8").splitlines()]
    assert report["ok"] is True
    assert report["rows_exported"] == 1
    assert rows[0]["example_id"] == "ex-1"
    assert rows[0]["trajectory_id"] == "tr-1"
    assert rows[0]["rollout_key"] == "rk-1"
    assert rows[0]["trainable_indexes"] == [1, 2]
    assert rows[0]["trainable_target_ids"] == [3, 4]
    assert rows[0]["inference_logprobs"] == [-0.2, -0.3]


def test_trainer_debug_tokens_writes_export(tmp_path: Path, capsys) -> None:
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
    write_path = tmp_path / "tokens.jsonl"

    assert (
        debug_main(
            [
                "trainer",
                "tokens",
                "--rollout-path",
                str(rollout_path),
                "--write-tokens",
                str(write_path),
                "--json",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in write_path.read_text(encoding="utf-8").splitlines()]
    assert report["rows_exported"] == 1
    assert rows[0]["trainable_target_ids"] == [3]


def test_runtime_parity_report_passes_within_threshold(tmp_path: Path) -> None:
    rollout_path = _write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {
                "input_ids": [1, 2, 3],
                "target_ids": [2, 3, 4],
                "loss_mask": [False, True, True],
                "inference_logprobs": [-0.2, -0.3],
                "trainer_logprobs": [-0.2005, -0.2995],
            }
        ],
    )

    report = build_runtime_parity_report(
        RLConfig(),
        rollout_path=rollout_path,
        threshold=0.001,
    )

    assert report["passed"] is True
    assert report["skipped"] is False
    assert report["token_count"] == 2
    assert report["max_abs_diff"] == pytest.approx(0.0005)


def test_runtime_parity_report_fails_outside_threshold(tmp_path: Path) -> None:
    rollout_path = _write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {
                "input_ids": [1, 2],
                "target_ids": [2, 3],
                "loss_mask": [False, True],
                "inference_logprobs": [-0.2],
                "trainer_logprobs": [-0.4],
            }
        ],
    )

    report = build_runtime_parity_report(
        RLConfig(),
        rollout_path=rollout_path,
        threshold=0.01,
    )

    assert report["passed"] is False
    assert report["skipped"] is False
    assert report["max_abs_diff"] == 0.2


def test_runtime_parity_report_skips_without_trainer_logprobs(tmp_path: Path) -> None:
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

    report = build_runtime_parity_report(RLConfig(), rollout_path=rollout_path)

    assert report["passed"] is False
    assert report["skipped"] is True
    assert report["skip_reason"] == "all checked rows are missing trainer logprobs"


def test_trainer_debug_parity_writes_report(tmp_path: Path, capsys) -> None:
    rollout_path = _write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {
                "input_ids": [1, 2],
                "target_ids": [2, 3],
                "loss_mask": [False, True],
                "inference_logprobs": [-0.2],
                "trainer_logprobs": [-0.2],
            }
        ],
    )
    report_path = tmp_path / "reports" / "parity.json"

    assert (
        debug_main(
            [
                "trainer",
                "parity",
                "--rollout-path",
                str(rollout_path),
                "--write-report",
                str(report_path),
                "--json",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert printed["passed"] is True
    assert written["passed"] is True

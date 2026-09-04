from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from datasets import Dataset

from wavelet.tools.convert_traces import convert_traces, load_trace_rows


def _event(*, event: str, details: object = None) -> dict[str, object]:
    return {
        "format_version": 1,
        "timestamp": "2026-01-02T03:04:05+00:00",
        "subsystem": "queue",
        "event": event,
        "step": 3,
        "details": details,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_convert_traces_writes_loadable_local_dataset(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "traces" / "step-000003.jsonl",
        [_event(event="published", details={"bytes": 12})],
    )
    # A run directory can contain unrelated JSONL; directory discovery should
    # stay under its canonical traces directory.
    _write_jsonl(run_dir / "rollouts" / "rows.jsonl", [{"prompt": "ignore"}])
    output_dir = tmp_path / "dataset"

    output_path = convert_traces(
        [run_dir],
        output_dir=output_dir,
        subset="events",
        split="train",
    )

    assert output_path == output_dir / "events" / "train.parquet"
    dataset = Dataset.from_parquet(output_path.as_posix())
    assert dataset.num_rows == 1
    assert dataset[0]["event"] == "published"
    assert json.loads(dataset[0]["details"]) == {"bytes": 12}
    metadata = yaml.safe_load(
        (output_dir / "README.md").read_text(encoding="utf-8").split("---")[1]
    )
    assert metadata["configs"] == [
        {
            "config_name": "events",
            "data_files": [{"split": "train", "path": "events/train.parquet"}],
        }
    ]


def test_convert_traces_registers_multiple_splits_without_duplicates(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(trace_path, [_event(event="consumed")])
    output_dir = tmp_path / "dataset"

    convert_traces(
        [trace_path],
        output_dir=output_dir,
        subset="events",
        split="train",
    )
    convert_traces(
        [trace_path],
        output_dir=output_dir,
        subset="events",
        split="validation",
    )

    card = (output_dir / "README.md").read_text(encoding="utf-8")
    metadata = yaml.safe_load(card.split("---")[1])
    assert metadata["configs"][0]["data_files"] == [
        {"split": "train", "path": "events/train.parquet"},
        {"split": "validation", "path": "events/validation.parquet"},
    ]


def test_load_trace_rows_reports_source_line_for_invalid_rows(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("\n{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2 is missing"):
        load_trace_rows([trace_path])


def test_convert_traces_requires_one_destination(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(trace_path, [_event(event="published")])

    with pytest.raises(ValueError, match="exactly one"):
        convert_traces([trace_path])

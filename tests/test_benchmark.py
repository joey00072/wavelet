from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wavelet.tools.benchmark import (
    aggregate_metrics,
    compare_results,
    run_benchmark,
)


def test_aggregate_metrics_merges_rows_and_excludes_warmup(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 1, "perf/throughput": 10.0},
        {"step": 1, "perf/mfu": 20.0},
        {"step": 2, "perf/throughput": 30.0},
        {"step": 2, "perf/mfu": 40.0, "perf/step_seconds": 2.0},
        {"step": 3, "perf/throughput": 50.0},
        {"step": 3, "perf/mfu": 60.0, "perf/step_seconds": 1.0},
        {"step": None, "perf/throughput": 1_000.0},
        {"step": 4, "perf/throughput": float("nan")},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    metrics = aggregate_metrics(path, warmup_steps=1)

    assert metrics["perf/throughput"] == {
        "mean": 40.0,
        "std": 10.0,
        "min": 30.0,
        "max": 50.0,
        "samples": 2.0,
    }
    assert metrics["perf/mfu"]["mean"] == 50.0
    assert metrics["perf/step_seconds"]["mean"] == 1.5


def _result(metrics: dict[str, dict[str, float]]) -> dict[str, object]:
    return {
        "format_version": 1,
        "benchmark_key": "same-workload",
        "metrics": metrics,
    }


def test_compare_results_checks_both_metric_directions() -> None:
    baseline = _result(
        {
            "perf/throughput": {"mean": 100.0},
            "perf/step_seconds": {"mean": 2.0},
        }
    )
    current = _result(
        {
            "perf/throughput": {"mean": 89.0},
            "perf/step_seconds": {"mean": 2.3},
        }
    )

    regressions = compare_results(current, baseline, threshold=0.05)

    assert [regression.metric for regression in regressions] == [
        "perf/throughput",
        "perf/step_seconds",
    ]
    assert regressions[0].change_ratio == pytest.approx(-0.11)
    assert regressions[1].change_ratio == pytest.approx(0.15)


def test_compare_results_accepts_changes_within_threshold() -> None:
    baseline = _result(
        {
            "perf/throughput": {"mean": 100.0},
            "perf/step_seconds": {"mean": 2.0},
        }
    )
    current = _result(
        {
            "perf/throughput": {"mean": 96.0},
            "perf/step_seconds": {"mean": 2.09},
        }
    )

    assert compare_results(current, baseline, threshold=0.05) == []


def test_compare_results_rejects_mismatched_identity() -> None:
    current = _result({"perf/throughput": {"mean": 100.0}})
    baseline = {**current, "benchmark_key": "different-workload"}

    with pytest.raises(ValueError, match="identity differs"):
        compare_results(current, baseline, threshold=0.05)


def test_compare_results_treats_missing_current_metric_as_regression() -> None:
    baseline = _result({"perf/mfu": {"mean": 50.0}})
    current = _result({})

    regressions = compare_results(current, baseline, threshold=0.05)

    assert len(regressions) == 1
    assert regressions[0].current is None


def test_run_benchmark_executes_job_and_writes_result(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "sft.yaml"
    config_path.write_text("max_steps: 2\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    output_path = tmp_path / "result.json"

    def fake_run(invocation, **kwargs):
        output_index = invocation.index("--output-dir") + 1
        actual_run_dir = Path(invocation[output_index])
        (actual_run_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 1, "perf/throughput": 10.0})
            + "\n"
            + json.dumps({"step": 2, "perf/throughput": 20.0})
            + "\n",
            encoding="utf-8",
        )
        kwargs["stdout"].write("training complete\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("wavelet.tools.benchmark.subprocess.run", fake_run)
    monkeypatch.setattr(
        "wavelet.tools.benchmark.torch.cuda.is_available", lambda: False
    )

    result = run_benchmark("sft", config_path, run_dir, output_path)

    assert result["metrics"] == {
        "perf/throughput": {
            "mean": 20.0,
            "std": 0.0,
            "min": 20.0,
            "max": 20.0,
            "samples": 1.0,
        }
    }
    assert json.loads(output_path.read_text(encoding="utf-8"))["benchmark_key"]
    assert "training complete" in (run_dir / "benchmark.log").read_text(
        encoding="utf-8"
    )

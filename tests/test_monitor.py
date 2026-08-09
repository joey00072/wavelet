import json
from datetime import UTC, datetime

import pytest

import wavelet.monitor as canonical_monitor
import wavelet.orchestrator.metrics as legacy_metrics
import wavelet.utils.monitoring as legacy_monitoring
from wavelet.monitor import (
    append_metric_row,
    prometheus_exposition,
    read_jsonl,
    redact,
    series_stats,
    summary_stats,
    tail_jsonl,
    tail_metric_rows,
)


def test_legacy_observability_modules_alias_canonical_module() -> None:
    assert legacy_metrics is canonical_monitor
    assert legacy_monitoring is canonical_monitor
    assert legacy_metrics.RolloutMetricInputs is canonical_monitor.RolloutMetricInputs
    assert legacy_monitoring.RunMonitor is canonical_monitor.RunMonitor


def test_jsonl_readers_preserve_strict_and_tail_contracts(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"value": 1}\ninvalid\n[2]\n{"value": 3}\n')

    rows, errors = tail_jsonl(path, limit=4)

    assert rows == [{"value": 1}, {"value": 3}]
    assert errors == 2
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_redact_and_series_stats():
    assert redact({"nested": [{"api_token": "secret"}], "safe": 1}) == {
        "nested": [{"api_token": "<redacted>"}],
        "safe": 1,
    }
    assert series_stats("reward/all", [1.0, 3.0]) == {
        "reward/all/mean": 2.0,
        "reward/all/max": 3.0,
        "reward/all/std": 1.0,
        "reward/all/min": 1.0,
    }
    assert summary_stats([]) == {
        "count": 0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    assert summary_stats([1.0, 3.0]) == {
        "count": 2,
        "min": 1.0,
        "max": 3.0,
        "mean": 2.0,
        "std": 1.0,
    }


def test_canonical_metric_journal_combines_and_filters_subsystems(tmp_path) -> None:
    append_metric_row(
        tmp_path,
        {"loss": 1.5},
        step=1,
        subsystem="trainer",
    )
    append_metric_row(
        tmp_path,
        {"reward/all/mean": 0.75},
        step=2,
        subsystem="orchestrator",
    )
    append_metric_row(
        tmp_path,
        {"eval/reward/mean": 0.8},
        step=2,
        subsystem="eval",
    )

    rows = tail_metric_rows(tmp_path / "metrics.jsonl", limit=10)

    assert [row["subsystem"] for row in rows] == [
        "trainer",
        "orchestrator",
        "eval",
    ]
    assert (
        tail_metric_rows(
            tmp_path / "metrics.jsonl",
            limit=1,
            subsystem="orchestrator",
        )[0]["reward/all/mean"]
        == 0.75
    )
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "metrics.lock").is_file()


def test_metric_reader_treats_rows_without_subsystem_as_trainer(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "loss": 2.0}\n', encoding="utf-8")

    assert tail_metric_rows(path, limit=1, subsystem="trainer") == [
        {"step": 1, "loss": 2.0}
    ]
    assert tail_metric_rows(path, limit=1, subsystem="eval") == []


def test_prometheus_exposition_uses_latest_finite_numeric_values(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC).isoformat()
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "step": 1,
                        "subsystem": "trainer",
                        "loss": 2.0,
                        "healthy": True,
                        "note": "ignored",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "step": 2,
                        "subsystem": "trainer",
                        "loss": 1.25,
                        "not_finite": float("nan"),
                    }
                ),
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "step": 2,
                        "subsystem": "eval",
                        "eval/reward/mean": 0.75,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = prometheus_exposition(path, subsystem="trainer")

    assert 'wavelet_metric{subsystem="trainer",name="loss"} 1.25' in output
    assert 'wavelet_metric{subsystem="trainer",name="step"} 2' in output
    assert "healthy" not in output
    assert "not_finite" not in output
    assert 'subsystem="eval"' not in output
    assert "wavelet_metric_timestamp_seconds" in output

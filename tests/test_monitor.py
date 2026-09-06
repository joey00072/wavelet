import json
import logging

import pytest

import wavelet.monitor as canonical_monitor
import wavelet.monitor as legacy_metrics
import wavelet.monitor as legacy_monitoring
from wavelet.configs.sft import SFTConfig
from wavelet.monitor import (
    read_jsonl,
    redact,
    series_stats,
    setup_config_logger,
    summary_stats,
    tail_jsonl,
)


def test_legacy_observability_modules_alias_canonical_module() -> None:
    assert legacy_metrics is canonical_monitor
    assert legacy_monitoring is canonical_monitor
    assert legacy_metrics.RolloutMetricInputs is canonical_monitor.RolloutMetricInputs
    assert legacy_monitoring.RunMonitor is canonical_monitor.RunMonitor


def test_config_logger_writes_structured_file_and_console(
    capsys, monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    config = SFTConfig(
        output_dir=tmp_path,
        log={"level": "info", "json_console": True, "json_file": True},
    )
    try:
        logger = setup_config_logger("sft-test", config)

        logger.info("policy loaded", extra={"policy_step": 4})
        for handler in logging.getLogger().handlers:
            handler.flush()

        console_row = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        file_rows = read_jsonl(tmp_path / "logs" / "sft-test.jsonl")
        assert console_row["message"] == "policy loaded"
        assert console_row["policy_step"] == 4
        assert file_rows[-1]["level"] == "info"
        assert file_rows[-1]["logger"] == "sft-test"
        assert file_rows[-1]["policy_step"] == 4
    finally:
        canonical_monitor._remove_managed_log_handlers(logging.getLogger())


def test_config_logger_can_disable_json_file(capsys, tmp_path) -> None:
    config = SFTConfig(
        output_dir=tmp_path,
        log={"level": "warning", "json_console": False, "json_file": False},
    )
    try:
        logger = setup_config_logger("console-only", config)

        logger.info("hidden")
        logger.warning("visible")
        output = capsys.readouterr().err

        assert "visible" in output
        assert "hidden" not in output
        assert not (tmp_path / "logs").exists()
    finally:
        canonical_monitor._remove_managed_log_handlers(logging.getLogger())


def test_jsonl_readers_preserve_strict_and_tail_contracts(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"value": 1}\ninvalid\n[2]\n{"value": 3}\n')

    rows, errors = tail_jsonl(path, limit=4)

    assert rows == [{"value": 1}, {"value": 3}]
    assert errors == 2
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_tail_jsonl_reads_long_utf8_rows_across_blocks(tmp_path):
    path = tmp_path / "long-records.jsonl"
    long_value = "λ" * 40_000
    path.write_text(
        f"{json.dumps({'value': 1})}\n\n"
        f"{json.dumps({'value': long_value})}\n"
        "invalid\n"
        f"{json.dumps({'value': 3})}\n",
        encoding="utf-8",
    )

    rows, errors = tail_jsonl(path, limit=3)

    assert rows == [{"value": long_value}, {"value": 3}]
    assert errors == 1


def test_redact_and_series_stats():
    assert redact(
        {
            "nested": [{"api_token": "secret"}],
            "max_completion_tokens": 8192,
            "tokenizer": "Qwen",
            "verifier_api_key_var": "PRIME_API_KEY",
        }
    ) == {
        "nested": [{"api_token": "<redacted>"}],
        "max_completion_tokens": 8192,
        "tokenizer": "Qwen",
        "verifier_api_key_var": "PRIME_API_KEY",
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

from __future__ import annotations

from wavelet.utils.monitoring import emit_perf


def test_emit_perf_preserves_line_shape(monkeypatch, capsys) -> None:
    monkeypatch.setenv("WAVELET_PERF_LOG", "1")

    emit_perf("example", step=2, seconds=0.125, mode="async")

    assert capsys.readouterr().out == (
        "WAVELET_PERF example step=2 seconds=0.125 mode=async\n"
    )

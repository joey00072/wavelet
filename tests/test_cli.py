from __future__ import annotations

import sys

import wavelet.cli


def test_cli_usage_does_not_eagerly_import_entrypoints(monkeypatch, capsys) -> None:
    for module_name in list(sys.modules):
        if module_name.startswith("wavelet.entrypoints."):
            del sys.modules[module_name]
    monkeypatch.setattr(sys, "argv", ["wavelet"])

    assert wavelet.cli.main() == 1

    assert "Usage: wavelet <command> [args]" in capsys.readouterr().out
    assert not any(
        module_name.startswith("wavelet.entrypoints.") for module_name in sys.modules
    )

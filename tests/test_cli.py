from __future__ import annotations

import sys

import wavelet.cli


def _clear_lazy_import_targets() -> None:
    for module_name in list(sys.modules):
        if module_name.startswith("wavelet.entrypoints.") or module_name in {
            "wavelet.orchestrator.preflight",
            "wavelet.inference.diagnostics",
            "wavelet.orchestrator.diagnostics",
            "wavelet.trainer.diagnostics",
        }:
            del sys.modules[module_name]


def test_cli_usage_does_not_eagerly_import_entrypoints(monkeypatch, capsys) -> None:
    _clear_lazy_import_targets()
    monkeypatch.setattr(sys, "argv", ["wavelet"])

    assert wavelet.cli.main() == 1

    output = capsys.readouterr().out
    assert "Usage: wavelet <command> [args]" in output
    assert "debug" in output
    assert "inference-server" in output
    assert "inference-debug" not in output
    assert "orchestrator-debug" not in output
    assert "rl-vllm-server" not in output
    assert not any(
        module_name.startswith("wavelet.entrypoints.") for module_name in sys.modules
    )


def test_debug_usage_does_not_eagerly_import_debug_entrypoints(
    monkeypatch,
    capsys,
) -> None:
    _clear_lazy_import_targets()
    monkeypatch.setattr(sys, "argv", ["wavelet", "debug"])

    assert wavelet.cli.main() == 1

    output = capsys.readouterr().out
    assert "Usage: wavelet debug <subcommand> [args]" in output
    assert "preflight" in output
    assert "inference" in output
    assert "orchestrator" in output
    assert "trainer" in output
    assert "wavelet.entrypoints.rl_debug" in sys.modules
    assert "wavelet.entrypoints.rl_preflight_debug" not in sys.modules
    assert "wavelet.entrypoints.rl_inference_debug" not in sys.modules
    assert "wavelet.entrypoints.rl_orchestrator_debug" not in sys.modules
    assert "wavelet.entrypoints.rl_trainer_debug" not in sys.modules
    assert "wavelet.orchestrator.preflight" not in sys.modules
    assert "wavelet.inference.diagnostics" not in sys.modules
    assert "wavelet.orchestrator.diagnostics" not in sys.modules
    assert "wavelet.trainer.diagnostics" not in sys.modules

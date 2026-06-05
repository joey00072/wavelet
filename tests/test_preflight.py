from __future__ import annotations

import json

from wavelet.configs.rl_config import RLConfig
from wavelet.entrypoints.rl_debug import main as debug_main
from wavelet.orchestrator.preflight import build_preflight_report


def test_preflight_reports_unavailable_cuda_device(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight._available_gpu_indices",
        lambda: {"0"},
    )
    config = RLConfig(
        data={"source": "local", "path": data_path},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
        launcher={
            "mode": "process",
            "inference_cuda_visible_devices": "0",
            "trainer_cuda_visible_devices": "1",
        },
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert any(
        check["name"] == "trainer_devices"
        and check["status"] == "error"
        and "requests CUDA device(s) 1" in check["message"]
        for check in report["checks"]
    )


def test_preflight_reports_missing_local_data(tmp_path) -> None:
    config = RLConfig(
        data={"source": "local", "path": tmp_path / "missing.jsonl"},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert any(
        check["name"] == "data_path_0" and check["status"] == "error"
        for check in report["checks"]
    )


def test_preflight_resolves_process_commands(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight._available_gpu_indices",
        lambda: {"0", "1"},
    )
    config = RLConfig(
        data={"source": "local", "path": data_path},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
        launcher={
            "mode": "process",
            "inference_cuda_visible_devices": "0",
            "trainer_cuda_visible_devices": "1",
        },
    )

    report = build_preflight_report(config)

    assert report["ok"] is True
    assert [command["role"] for command in report["commands"]] == [
        "inference_server_0",
        "trainer",
        "inference",
    ]
    assert report["paths"]["queue_dir"] == str(tmp_path / "run" / "rollouts")
    assert report["paths"]["policy_dir"] == str(tmp_path / "run" / "policies")


def test_preflight_cli_returns_nonzero_for_errors(tmp_path, capsys) -> None:
    config_path = tmp_path / "rl.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  source: local",
                "  path: missing.jsonl",
                "reward:",
                "  mode: math_format",
                f"output_dir: {tmp_path / 'run'}",
            ]
        ),
        encoding="utf-8",
    )

    assert debug_main(["preflight", "--json", "@", str(config_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False


def test_preflight_reports_qlora_without_lora(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight.importlib.util.find_spec",
        lambda name: object() if name == "bitsandbytes" else None,
    )
    config = RLConfig(
        data={"source": "local", "path": data_path},
        model={"load_in_4bit": True},
        lora=None,
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert report["summary"]["low_precision"]["trainer_load_in_4bit"] is True
    assert any(
        check["name"] == "qlora_adapter"
        and check["status"] == "error"
        and "does not support full-model 4-bit training" in check["message"]
        for check in report["checks"]
    )


def test_preflight_reports_missing_bitsandbytes_for_qlora(
    tmp_path,
    monkeypatch,
) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight.importlib.util.find_spec",
        lambda name: None if name == "bitsandbytes" else object(),
    )
    config = RLConfig(
        data={"source": "local", "path": data_path},
        model={"load_in_4bit": True},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert any(
        check["name"] == "bitsandbytes_available"
        and check["status"] == "error"
        for check in report["checks"]
    )


def test_preflight_warns_for_quantized_inference_without_qlora(
    tmp_path,
    monkeypatch,
) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight._available_gpu_indices",
        lambda: {"0"},
    )
    config = RLConfig(
        data={"source": "local", "path": data_path},
        inference={"vllm": {"quantization": "bitsandbytes", "load_format": "bitsandbytes"}},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is True
    assert any(
        check["name"] == "low_precision_inference_mismatch"
        and check["status"] == "warning"
        for check in report["checks"]
    )


def test_preflight_reports_qlora_topology_errors(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight.importlib.util.find_spec",
        lambda name: object() if name == "bitsandbytes" else None,
    )
    config = RLConfig(
        data={"source": "local", "path": data_path},
        model={"load_in_4bit": True},
        fsdp={"enabled": True, "tp": 2},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    error_names = {
        check["name"]
        for check in report["checks"]
        if check["status"] == "error"
    }
    assert {"qlora_fsdp", "qlora_tensor_parallel"} <= error_names

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.entrypoints.rl_debug import main as debug_main
from wavelet.orchestrator.preflight import build_preflight_report


CUSTOM_ALGORITHM_FILE = Path(__file__).parent / "fixtures" / "custom_algorithm.py"


def _write_local_data(tmp_path: Path) -> Path:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    return data_path


def test_preflight_reports_unavailable_cuda_device(tmp_path, monkeypatch) -> None:
    data_path = _write_local_data(tmp_path)
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


def test_preflight_reports_missing_source_local_data(tmp_path) -> None:
    data_path = _write_local_data(tmp_path)
    config = RLConfig(
        data={"source": "local", "path": data_path},
        output_dir=tmp_path / "run",
        orchestrator={
            "train_sources": [
                {
                    "name": "opd-data",
                    "data": {
                        "source": "local",
                        "path": tmp_path / "missing-opd.jsonl",
                    },
                }
            ]
        },
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert report["summary"]["train_sources"][0]["name"] == "opd-data"
    assert any(
        check["name"] == "data_source_opd-data_path_0" and check["status"] == "error"
        for check in report["checks"]
    )


def test_preflight_reports_unimportable_custom_algorithm(tmp_path) -> None:
    data_path = _write_local_data(tmp_path)
    config = RLConfig(
        algo={
            "file": tmp_path / "algorithm_that_does_not_exist.py",
            "algorithm": "Algorithm",
            "scope": "group",
        },
        data={"source": "local", "path": data_path},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert report["summary"]["algo"]["type"] == "custom"
    assert any(
        check["name"] == "algorithm"
        and check["status"] == "error"
        and "algorithm_that_does_not_exist.py" in check["message"]
        for check in report["checks"]
    )


def test_preflight_reports_custom_algorithm_syntax_error(tmp_path) -> None:
    data_path = _write_local_data(tmp_path)
    algorithm_path = tmp_path / "broken_algorithm.py"
    algorithm_path.write_text("def broken(:\n", encoding="utf-8")
    config = RLConfig(
        algo={
            "file": algorithm_path,
            "algorithm": "broken",
            "scope": "rollout",
        },
        data={"source": "local", "path": data_path},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert any(
        check["name"] == "algorithm"
        and check["status"] == "error"
        and "SyntaxError" in check["message"]
        for check in report["checks"]
    )


def test_preflight_reports_custom_algorithm_constructor_error(tmp_path) -> None:
    data_path = _write_local_data(tmp_path)
    config = RLConfig(
        algo={
            "file": CUSTOM_ALGORITHM_FILE,
            "algorithm": "multiplier",
            "scope": "group",
        },
        data={"source": "local", "path": data_path},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert any(
        check["name"] == "algorithm"
        and check["status"] == "error"
        and "TypeError" in check["message"]
        and "multiplier" in check["message"]
        for check in report["checks"]
    )


def test_preflight_reports_each_source_algorithm_and_opd_teacher(tmp_path) -> None:
    data_path = _write_local_data(tmp_path)
    config = RLConfig(
        data={"source": "local", "path": data_path},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
        orchestrator={
            "train_sources": [
                {
                    "name": "math-opd",
                    "algo": {
                        "type": "opd",
                        "teacher": {
                            "name": "math-teacher",
                            "base_url": "http://teacher:8001/v1",
                        },
                    },
                }
            ]
        },
    )

    report = build_preflight_report(config)

    source_check = next(
        check
        for check in report["checks"]
        if check["name"] == "algorithm_source_math-opd"
    )
    assert source_check["status"] == "ok"
    assert source_check["details"]["teacher"] == "math-teacher"


def test_config_rejects_invalid_opd_teacher_url(tmp_path) -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        RLConfig(
            data={"source": "local", "path": _write_local_data(tmp_path)},
            output_dir=tmp_path / "run",
            reward={"mode": "math_format"},
            algo={
                "type": "opd",
                "teacher": {"name": "teacher", "base_url": "teacher:8001"},
            },
        )


def test_preflight_resolves_process_commands(tmp_path, monkeypatch) -> None:
    data_path = _write_local_data(tmp_path)
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
    data_path = _write_local_data(tmp_path)
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
    data_path = _write_local_data(tmp_path)
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
        check["name"] == "bitsandbytes_available" and check["status"] == "error"
        for check in report["checks"]
    )


def test_preflight_warns_for_quantized_inference_without_qlora(
    tmp_path,
    monkeypatch,
) -> None:
    data_path = _write_local_data(tmp_path)
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight._available_gpu_indices",
        lambda: {"0"},
    )
    config = RLConfig(
        data={"source": "local", "path": data_path},
        inference={
            "vllm": {"quantization": "bitsandbytes", "load_format": "bitsandbytes"}
        },
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
    data_path = _write_local_data(tmp_path)
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
        check["name"] for check in report["checks"] if check["status"] == "error"
    }
    assert {"qlora_fsdp", "qlora_tensor_parallel"} <= error_names


def test_qlora_check_json_contract_is_stable(tmp_path, monkeypatch) -> None:
    data_path = _write_local_data(tmp_path)
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight.importlib.util.find_spec",
        lambda name: object() if name == "bitsandbytes" else None,
    )
    report = build_preflight_report(
        RLConfig(
            data={"source": "local", "path": data_path},
            model={"load_in_4bit": True},
            output_dir=tmp_path / "run",
            reward={"mode": "math_format"},
        )
    )
    checks = {
        item["name"]: item
        for item in report["checks"]
        if item["name"].startswith("qlora_") or item["name"] == "bitsandbytes_available"
    }

    assert {name: item["status"] for name, item in checks.items()} == {
        "bitsandbytes_available": "ok",
        "qlora_adapter": "ok",
        "qlora_colocate_sleep": "ok",
        "qlora_fsdp": "ok",
    }
    expected_keys = {"name", "status", "message", "details"}
    assert all(set(item) == expected_keys for item in checks.values())

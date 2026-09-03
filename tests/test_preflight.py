from __future__ import annotations

import json
from pathlib import Path

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


def test_preflight_reports_effective_rollout_batch_shape(tmp_path) -> None:
    config = RLConfig(
        data={"source": "local", "path": _write_local_data(tmp_path)},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
        orchestrator={
            "examples_per_step": 17,
            "rollouts_per_example": 8,
            "rollout_chunk_examples": 3,
        },
    )

    report = build_preflight_report(config)
    check = next(item for item in report["checks"] if item["name"] == "rollout_chunks")

    assert check["details"] == {
        "groups": 17,
        "rollouts_per_group": 8,
        "rollouts": 136,
        "chunks": 6,
    }


def test_preflight_reports_missing_model_adapter(tmp_path) -> None:
    data_path = _write_local_data(tmp_path)
    adapter_path = tmp_path / "missing-adapter"
    config = RLConfig(
        data={"source": "local", "path": data_path},
        model={"adapter_path": adapter_path},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert any(
        check["name"] == "model_adapter_path"
        and check["status"] == "error"
        and check["details"]["missing_files"]
        == ["adapter_config.json", "adapter_model.safetensors"]
        for check in report["checks"]
    )


def test_preflight_accepts_loadable_model_adapter(tmp_path) -> None:
    data_path = _write_local_data(tmp_path)
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
    config = RLConfig(
        data={"source": "local", "path": data_path},
        model={"adapter_path": adapter_path},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert any(
        check["name"] == "model_adapter_path" and check["status"] == "ok"
        for check in report["checks"]
    )


def test_preflight_rejects_adapter_removed_by_clean_output_dir(tmp_path) -> None:
    data_path = _write_local_data(tmp_path)
    output_dir = tmp_path / "run"
    adapter_path = output_dir / "policies" / "step-000001" / "adapter"
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
    config = RLConfig(
        data={"source": "local", "path": data_path},
        model={"adapter_path": adapter_path},
        output_dir=output_dir,
        clean_output_dir=True,
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert any(
        check["name"] == "model_adapter_path"
        and check["status"] == "error"
        and check["details"]["removed_by_clean_output_dir"] is True
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


def test_checkpoint_output_dir_does_not_replace_run_output_dir(
    tmp_path, monkeypatch
) -> None:
    data_path = _write_local_data(tmp_path)
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight._available_gpu_indices",
        lambda: {"0"},
    )
    run_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "large-volume"
    config = RLConfig(
        data={"source": "local", "path": data_path},
        output_dir=run_dir,
        ckpt={"mode": "async", "interval": 10, "output_dir": checkpoint_dir},
    )

    report = build_preflight_report(config)

    assert config.output_dir == run_dir
    assert config.checkpoint_output_dir == checkpoint_dir
    assert report["paths"]["output_dir"] == str(run_dir)
    assert report["paths"]["checkpoint_dir"] == str(checkpoint_dir)
    assert any(
        check["name"] == "checkpoint_parent_writable"
        and check["status"] == "ok"
        for check in report["checks"]
    )


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


def test_preflight_requires_flash_attention_when_explicit(
    tmp_path,
    monkeypatch,
) -> None:
    data_path = _write_local_data(tmp_path)
    monkeypatch.setattr(
        "wavelet.orchestrator.preflight.importlib.util.find_spec",
        lambda name: None if name == "flash_attn" else object(),
    )
    config = RLConfig(
        data={"source": "local", "path": data_path},
        model={"attn_implementation": "flash_attention_2"},
        output_dir=tmp_path / "run",
        reward={"mode": "math_format"},
    )

    report = build_preflight_report(config)

    assert report["ok"] is False
    assert report["summary"]["trainer_attention"] == "flash_attention_2"
    assert any(
        check["name"] == "flash_attention_available"
        and check["status"] == "error"
        and "uv sync --extra flash-attn" in check["message"]
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

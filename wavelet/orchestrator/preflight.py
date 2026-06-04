from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.queue import resolve_policy_dir, resolve_queue_dir
from wavelet.orchestrator.schedule import chunks_per_step, target_steps


CheckStatus = Literal["ok", "warning", "error"]


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] | None = None


def build_preflight_report(config: RLConfig) -> dict[str, Any]:
    """Build cheap launch diagnostics without starting trainer or inference."""
    checks = [
        *_path_checks(config),
        *_launcher_checks(config),
        *_port_checks(config),
        *_schedule_checks(config),
        *_low_precision_checks(config),
    ]
    commands: list[dict[str, Any]] = []
    try:
        commands = _resolved_commands(config)
    except ValueError as exc:
        checks.append(
            PreflightCheck(
                name="resolved_commands",
                status="error",
                message=str(exc),
            )
        )
    return {
        "ok": not any(check.status == "error" for check in checks),
        "summary": _summary(config),
        "paths": _paths(config),
        "commands": commands,
        "checks": [asdict(check) for check in checks],
    }


def _summary(config: RLConfig) -> dict[str, Any]:
    return {
        "model": config.model.name,
        "output_dir": str(config.output_dir),
        "launcher_mode": config.launcher.mode,
        "orchestrator_enabled": config.orchestrator.enabled,
        "inference_mode": config.inference.mode,
        "inference_backend": config.inference.vllm.server_backend,
        "policy_transfer": config.policy_transfer.type,
        "target_steps": target_steps(config),
        "low_precision": _low_precision_summary(config),
    }


def _low_precision_summary(config: RLConfig) -> dict[str, Any]:
    return {
        "trainer_load_in_4bit": config.model.load_in_4bit,
        "lora_enabled": config.lora is not None,
        "fsdp_enabled": config.fsdp.enabled,
        "launcher_mode": config.launcher.mode,
        "inference_quantization": config.inference.vllm.quantization,
        "inference_load_format": config.inference.vllm.load_format,
    }


def _paths(config: RLConfig) -> dict[str, str]:
    return {
        "output_dir": str(config.output_dir),
        "queue_dir": str(resolve_queue_dir(config.output_dir, config.transport)),
        "policy_dir": str(resolve_policy_dir(config.output_dir, config.policy_transfer)),
        "events_dir": str(config.output_dir / "events"),
    }


def _path_checks(config: RLConfig) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    checks.extend(_data_path_checks(config))
    checks.append(_output_dir_check(config.output_dir, clean=config.clean_output_dir))
    checks.append(
        _parent_writable_check(
            resolve_queue_dir(config.output_dir, config.transport),
            name="queue_parent_writable",
        )
    )
    checks.append(
        _parent_writable_check(
            resolve_policy_dir(config.output_dir, config.policy_transfer),
            name="policy_parent_writable",
        )
    )
    return checks


def _data_path_checks(config: RLConfig) -> list[PreflightCheck]:
    if config.data.source != "local":
        return [
            PreflightCheck(
                name="data_source",
                status="ok",
                message=f"data.source={config.data.source!r} does not require a local path preflight.",
            )
        ]
    paths = config.data.path if isinstance(config.data.path, list) else [config.data.path]
    checks: list[PreflightCheck] = []
    for index, path in enumerate(paths):
        exists = Path(path).exists()
        checks.append(
            PreflightCheck(
                name=f"data_path_{index}",
                status="ok" if exists else "error",
                message=(
                    f"Local data path exists: {path}"
                    if exists
                    else f"Local data path does not exist: {path}"
                ),
                details={"path": str(path)},
            )
        )
    return checks


def _output_dir_check(output_dir: Path, *, clean: bool) -> PreflightCheck:
    if output_dir.exists() and not clean:
        return PreflightCheck(
            name="output_dir",
            status="warning",
            message=(
                "Output directory already exists; use a clean run directory unless "
                "you are intentionally resuming or inspecting existing state."
            ),
            details={"path": str(output_dir)},
        )
    return PreflightCheck(
        name="output_dir",
        status="ok",
        message=f"Output directory is ready to create: {output_dir}",
        details={"path": str(output_dir)},
    )


def _parent_writable_check(path: Path, *, name: str) -> PreflightCheck:
    parent = _existing_parent(path)
    writable = os.access(parent, os.W_OK)
    return PreflightCheck(
        name=name,
        status="ok" if writable else "error",
        message=(
            f"Parent directory is writable: {parent}"
            if writable
            else f"Parent directory is not writable: {parent}"
        ),
        details={"path": str(path), "parent": str(parent)},
    )


def _existing_parent(path: Path) -> Path:
    cursor = path if path.exists() else path.parent
    while not cursor.exists() and cursor != cursor.parent:
        cursor = cursor.parent
    return cursor


def _launcher_checks(config: RLConfig) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if config.launcher.mode == "process" and world_size > 1:
        checks.append(
            PreflightCheck(
                name="torchrun_launcher",
                status="error",
                message="Do not run 'wavelet rl' process launcher under torchrun.",
                details={"WORLD_SIZE": world_size},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="torchrun_launcher",
                status="ok",
                message="Launcher mode is compatible with the current WORLD_SIZE.",
                details={"WORLD_SIZE": world_size},
            )
        )

    checks.extend(_device_group_checks(config))
    return checks


def _device_group_checks(config: RLConfig) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    gpu_indices = _available_gpu_indices()
    replicas = config.launcher.inference_num_replicas
    try:
        inference_groups = _as_device_groups(
            config.launcher.inference_cuda_visible_devices,
            replicas,
        )
    except ValueError as exc:
        return [
            PreflightCheck(
                name="inference_cuda_visible_devices",
                status="error",
                message=str(exc),
            )
        ]

    for index, group in enumerate(inference_groups):
        device_count = _device_group_size(config, group)
        required = _required_inference_devices(config)
        status: CheckStatus = "ok" if device_count >= required else "warning"
        checks.append(
            PreflightCheck(
                name=f"inference_devices_{index}",
                status=_device_status(status, group, gpu_indices),
                message=(
                    _device_message(
                        f"Inference replica {index}",
                        group,
                        gpu_indices,
                        fallback=(
                            f"Inference replica {index} has {device_count} visible "
                            f"device(s); configured vLLM needs {required}."
                        ),
                    )
                ),
                details={"cuda_visible_devices": group, "required_devices": required},
            )
        )

    try:
        trainer_group = _trainer_device_group(config)
    except ValueError as exc:
        checks.append(
            PreflightCheck(
                name="trainer_devices",
                status="error",
                message=str(exc),
            )
        )
        return checks
    if (
        config.launcher.mode in {"colocate", "colocate_sleep"}
        and config.launcher.inference_num_replicas != 1
        and config.launcher.trainer_cuda_visible_devices is None
    ):
        checks.append(
            PreflightCheck(
                name="trainer_devices",
                status="error",
                message=(
                    f"launcher.mode={config.launcher.mode!r} requires "
                    "launcher.trainer_cuda_visible_devices when using multiple "
                    "inference replicas."
                ),
            )
        )
        return checks
    checks.append(
        PreflightCheck(
            name="trainer_devices",
            status=_device_status("ok", trainer_group, gpu_indices),
            message=_device_message(
                "Trainer",
                trainer_group,
                gpu_indices,
                fallback=(
                    "Trainer CUDA_VISIBLE_DEVICES is resolved."
                    if trainer_group is not None
                    else "Trainer CUDA_VISIBLE_DEVICES is not pinned; current environment will be used."
                ),
            ),
            details={"cuda_visible_devices": trainer_group},
        )
    )
    return checks


def _port_checks(config: RLConfig) -> list[PreflightCheck]:
    if config.inference.mode != "vllm_http":
        return [
            PreflightCheck(
                name="inference_ports",
                status="ok",
                message="Inference mode does not start HTTP vLLM servers.",
            )
        ]
    try:
        ports = _http_ports(config, config.launcher.inference_num_replicas)
    except ValueError as exc:
        return [
            PreflightCheck(
                name="inference_ports",
                status="error",
                message=str(exc),
            )
        ]
    checks = []
    for port in ports:
        available = _port_available(config.inference.http.host, port)
        checks.append(
            PreflightCheck(
                name=f"inference_port_{port}",
                status="ok" if available else "warning",
                message=(
                    f"HTTP inference port appears available: {port}"
                    if available
                    else f"HTTP inference port appears in use: {port}"
                ),
                details={"host": config.inference.http.host, "port": port},
            )
        )
    return checks


def _schedule_checks(config: RLConfig) -> list[PreflightCheck]:
    checks = [
        PreflightCheck(
            name="target_steps",
            status="ok",
            message=f"Resolved target steps: {target_steps(config)}",
        )
    ]
    if config.orchestrator.examples_per_step is not None:
        checks.append(
            PreflightCheck(
                name="rollout_chunks",
                status="ok",
                message=f"Resolved chunks per optimizer step: {chunks_per_step(config)}",
            )
        )
    return checks


def _low_precision_checks(config: RLConfig) -> list[PreflightCheck]:
    trainer_4bit = config.model.load_in_4bit
    inference_quantized = bool(
        config.inference.vllm.quantization or config.inference.vllm.load_format
    )
    if not trainer_4bit and not inference_quantized:
        return [
            PreflightCheck(
                name="low_precision",
                status="ok",
                message="No low-precision trainer or inference settings enabled.",
                details=_low_precision_summary(config),
            )
        ]

    checks: list[PreflightCheck] = [
        PreflightCheck(
            name="low_precision",
            status="ok",
            message="Resolved low-precision launch settings.",
            details=_low_precision_summary(config),
        )
    ]
    if trainer_4bit:
        checks.extend(_trainer_4bit_checks(config))
    if inference_quantized and not trainer_4bit:
        checks.append(
            PreflightCheck(
                name="low_precision_inference_mismatch",
                status="warning",
                message=(
                    "Inference is configured with vLLM low-precision loading, but "
                    "trainer model.load_in_4bit is false. Verify the train/serve "
                    "precision mismatch is intentional."
                ),
                details={
                    "quantization": config.inference.vllm.quantization,
                    "load_format": config.inference.vllm.load_format,
                },
            )
        )
    return checks


def _trainer_4bit_checks(config: RLConfig) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
    checks.append(
        PreflightCheck(
            name="bitsandbytes_available",
            status="ok" if bitsandbytes_available else "error",
            message=(
                "bitsandbytes is importable for QLoRA training."
                if bitsandbytes_available
                else "model.load_in_4bit=true requires bitsandbytes to be installed."
            ),
        )
    )
    checks.append(
        PreflightCheck(
            name="qlora_adapter",
            status="ok" if config.lora is not None else "error",
            message=(
                "QLoRA adapter training is enabled."
                if config.lora is not None
                else (
                    "model.load_in_4bit=true requires a LoRA config; Wavelet does "
                    "not support full-model 4-bit training."
                )
            ),
        )
    )
    if config.fsdp.enabled:
        checks.append(
            PreflightCheck(
                name="qlora_fsdp",
                status="error",
                message=(
                    "QLoRA training uses replicated DDP in Wavelet. Disable "
                    "fsdp.enabled for model.load_in_4bit=true."
                ),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="qlora_fsdp",
                status="ok",
                message="FSDP is disabled for QLoRA training.",
            )
        )
    if config.launcher.mode == "colocate_sleep":
        checks.append(
            PreflightCheck(
                name="qlora_colocate_sleep",
                status="error",
                message=(
                    "QLoRA does not support colocate_sleep yet because bitsandbytes "
                    "4-bit modules cannot be moved between CPU and GPU."
                ),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="qlora_colocate_sleep",
                status="ok",
                message="Launcher mode is compatible with QLoRA.",
            )
        )
    if config.fsdp.tp > 1:
        checks.append(
            PreflightCheck(
                name="qlora_tensor_parallel",
                status="error",
                message=(
                    "QLoRA with trainer tensor parallelism is not supported. Set "
                    "fsdp.tp=1 for model.load_in_4bit=true."
                ),
                details={"fsdp_tp": config.fsdp.tp},
            )
        )
    return checks


def _resolved_commands(config: RLConfig) -> list[dict[str, Any]]:
    if not config.orchestrator.enabled:
        return [
            {
                "role": "trainer",
                "command": "uv run python -m wavelet rl",
                "config": "<provided config>",
                "cuda_visible_devices": _trainer_device_group(config),
            }
        ]

    if config.launcher.mode in {"process", "colocate", "colocate_sleep"}:
        commands: list[dict[str, Any]] = []
        ports = _http_ports(config, config.launcher.inference_num_replicas)
        inference_groups = _as_device_groups(
            config.launcher.inference_cuda_visible_devices,
            len(ports),
        )
        if config.inference.mode == "vllm_http":
            server_command = (
                "inference-server"
                if config.inference.vllm.server_backend == "openai"
                else "native-inference-server"
            )
            for index, (port, devices) in enumerate(
                zip(ports, inference_groups, strict=True)
            ):
                commands.append(
                    {
                        "role": f"inference_server_{index}",
                        "command": f"uv run python -m wavelet {server_command}",
                        "port": port,
                        "cuda_visible_devices": devices,
                    }
                )
        commands.extend(
            [
                {
                    "role": "trainer",
                    "command": "uv run python -m wavelet rl-trainer",
                    "torchrun_nproc_per_node": config.launcher.trainer_num_processes,
                    "cuda_visible_devices": _trainer_device_group(config),
                },
                {
                    "role": "inference",
                    "command": "uv run python -m wavelet rl-inference",
                    "cuda_visible_devices": None,
                },
            ]
        )
        return commands

    return [
        {
            "role": "integrated",
            "command": "uv run python -m wavelet rl",
            "cuda_visible_devices": _trainer_device_group(config),
        }
    ]


def _as_device_groups(value: str | list[str] | None, count: int) -> list[str | None]:
    if isinstance(value, list):
        groups = [str(item) for item in value]
    elif value is None:
        groups = [None]
    else:
        separator = ";" if ";" in value else "|"
        groups = [item.strip() for item in value.split(separator) if item.strip()]
        if not groups:
            groups = [value]
    if len(groups) == 1:
        return groups * count
    if len(groups) != count:
        raise ValueError(
            "Expected one CUDA device group or exactly "
            f"{count} groups, got {len(groups)}."
        )
    return groups


def _trainer_device_group(config: RLConfig) -> str | None:
    if config.launcher.mode in {"colocate", "colocate_sleep"}:
        if config.launcher.trainer_cuda_visible_devices is not None:
            return _as_device_groups(
                config.launcher.trainer_cuda_visible_devices,
                1,
            )[0]
        if config.launcher.inference_num_replicas != 1:
            return None
        return _as_device_groups(
            config.launcher.inference_cuda_visible_devices,
            1,
        )[0]
    return _as_device_groups(
        config.launcher.trainer_cuda_visible_devices,
        1,
    )[0]


def _http_ports(config: RLConfig, count: int) -> list[int]:
    configured = config.inference.http.ports
    if configured is not None:
        if len(configured) != count:
            raise ValueError(
                "inference.http.ports must have exactly "
                f"{count} entries when launcher.inference_num_replicas={count}."
            )
        return configured
    return [config.inference.http.port + offset for offset in range(count)]


def _device_group_size(config: RLConfig, cuda_visible_devices: str | None) -> int:
    if cuda_visible_devices is None:
        dp_size = (
            config.inference.vllm.data_parallel_size_local
            or config.inference.vllm.data_parallel_size
        )
        return config.inference.vllm.tensor_parallel_size * dp_size
    return len([device for device in cuda_visible_devices.split(",") if device.strip()])


def _required_inference_devices(config: RLConfig) -> int:
    dp_size = (
        config.inference.vllm.data_parallel_size_local
        or config.inference.vllm.data_parallel_size
    )
    return config.inference.vllm.tensor_parallel_size * dp_size


def _available_gpu_indices() -> set[str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    indices = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return indices or None


def _missing_gpu_indices(
    cuda_visible_devices: str | None,
    available_indices: set[str] | None,
) -> list[str]:
    if cuda_visible_devices is None or available_indices is None:
        return []
    requested = [device.strip() for device in cuda_visible_devices.split(",")]
    numeric = [device for device in requested if device.isdigit()]
    return [device for device in numeric if device not in available_indices]


def _device_status(
    fallback: CheckStatus,
    cuda_visible_devices: str | None,
    available_indices: set[str] | None,
) -> CheckStatus:
    if _missing_gpu_indices(cuda_visible_devices, available_indices):
        return "error"
    return fallback


def _device_message(
    label: str,
    cuda_visible_devices: str | None,
    available_indices: set[str] | None,
    *,
    fallback: str,
) -> str:
    missing = _missing_gpu_indices(cuda_visible_devices, available_indices)
    if missing:
        available = ", ".join(sorted(available_indices or [])) or "unknown"
        return (
            f"{label} requests CUDA device(s) {', '.join(missing)}, but "
            f"nvidia-smi reports available device index(es): {available}."
        )
    return fallback


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError:
        return False
    return True

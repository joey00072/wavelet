from __future__ import annotations

from wavelet.configs.rl_config import RLConfig


def device_groups(
    value: str | list[str] | None,
    count: int,
) -> list[str | None]:
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


def trainer_device_group(config: RLConfig, *, strict: bool = True) -> str | None:
    if config.launcher.mode in {"colocate", "colocate_sleep"}:
        if config.launcher.trainer_cuda_visible_devices is not None:
            return device_groups(config.launcher.trainer_cuda_visible_devices, 1)[0]
        if config.launcher.inference_num_replicas != 1:
            if not strict:
                return None
            raise ValueError(
                f"launcher.mode='{config.launcher.mode}' requires "
                "launcher.trainer_cuda_visible_devices when using multiple "
                "inference replicas."
            )
        return device_groups(config.launcher.inference_cuda_visible_devices, 1)[0]
    return device_groups(config.launcher.trainer_cuda_visible_devices, 1)[0]


def http_ports(config: RLConfig, count: int) -> list[int]:
    configured = config.inference.http.ports
    if configured is not None:
        if len(configured) != count:
            raise ValueError(
                "inference.http.ports must have exactly "
                f"{count} entries when launcher.inference_num_replicas={count}."
            )
        return configured
    return [config.inference.http.port + offset for offset in range(count)]


def device_group_size(config: RLConfig, cuda_visible_devices: str | None) -> int:
    if cuda_visible_devices is None:
        dp_size = (
            config.inference.vllm.data_parallel_size_local
            or config.inference.vllm.data_parallel_size
        )
        return config.inference.vllm.tensor_parallel_size * dp_size
    return len([device for device in cuda_visible_devices.split(",") if device.strip()])

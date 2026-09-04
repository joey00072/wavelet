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


def required_inference_devices(config: RLConfig) -> int:
    dp_size = (
        config.inference.vllm.data_parallel_size_local
        or config.inference.vllm.data_parallel_size
    )
    return config.inference.vllm.tensor_parallel_size * dp_size


def device_group_size(config: RLConfig, cuda_visible_devices: str | None) -> int:
    if cuda_visible_devices is None:
        return required_inference_devices(config)
    return len([device for device in cuda_visible_devices.split(",") if device.strip()])


def nccl_inference_ranks(config: RLConfig, cuda_visible_devices: str | None) -> int:
    """Return the NCCL ranks one inference replica joins with.

    Only vLLM tensor/data-parallel workers call init_broadcaster, so a device
    group that does not match them exactly would stall the weight broadcast.
    """
    required = required_inference_devices(config)
    visible = device_group_size(config, cuda_visible_devices)
    if visible != required:
        raise ValueError(
            "NCCL policy transfer requires each inference replica to expose "
            f"exactly {required} CUDA device(s) (tensor_parallel_size x "
            "data_parallel_size), but inference_cuda_visible_devices="
            f"'{cuda_visible_devices}' lists {visible}."
        )
    return required


def _device_set(group: str | None) -> set[str] | None:
    if group is None:
        return None
    return {device.strip() for device in group.split(",") if device.strip()}


def device_group_conflict_error(config: RLConfig) -> str | None:
    """Explain why separate processes would compete for the same GPUs.

    Colocated modes share devices by design; in ``process`` mode the trainer
    and every inference replica run as independent processes that each expect
    their whole device group, so overlapping groups only surface later as
    out-of-memory failures.
    """
    if config.launcher.mode != "process":
        return None
    replicas = config.launcher.inference_num_replicas
    inference_groups = device_groups(
        config.launcher.inference_cuda_visible_devices, replicas
    )
    inference_sets = [_device_set(group) for group in inference_groups]
    trainer_set = _device_set(trainer_device_group(config, strict=False))
    if trainer_set is not None:
        for index, devices in enumerate(inference_sets):
            if devices is not None and trainer_set & devices:
                shared = ",".join(sorted(trainer_set & devices))
                return (
                    "launcher.mode='process' runs the trainer and inference "
                    f"replica {index} as separate processes, but both are pinned "
                    f"to CUDA device(s) {shared}. Give each process disjoint "
                    "trainer_cuda_visible_devices/inference_cuda_visible_devices "
                    "or use a colocate mode."
                )
    for left in range(replicas):
        for right in range(left + 1, replicas):
            left_set, right_set = inference_sets[left], inference_sets[right]
            if left_set is not None and right_set is not None and left_set & right_set:
                shared = ",".join(sorted(left_set & right_set))
                return (
                    f"launcher.mode='process' inference replicas {left} and "
                    f"{right} share CUDA device(s) {shared}. List one disjoint "
                    "device group per replica in inference_cuda_visible_devices "
                    "(separated by ';')."
                )
    return None


def validate_device_groups(config: RLConfig) -> None:
    error = device_group_conflict_error(config)
    if error is not None:
        raise ValueError(error)


def rollout_reward_mode_error(config: RLConfig) -> str | None:
    """Explain why ``reward.mode`` cannot score the configured rollout source.

    ``passthrough`` copies rewards from the dataset, so it cannot score
    completions that vLLM generates unless a custom rollout function provides
    them. Checked once by the launcher (and by preflight) rather than in config
    validation so process-mode roles can re-validate the parent's full dump.
    """
    generates_rollouts = (
        config.orchestrator.enabled
        and config.inference.enabled
        and config.inference.mode == "vllm_http"
        and config.orchestrator.custom_rollout_function is None
    )
    if not generates_rollouts or config.reward.mode != "passthrough":
        return None
    return (
        "reward.mode='passthrough' does not score completions generated by "
        "inference.mode='vllm_http'; choose a scoring reward.mode or set "
        "orchestrator.custom_rollout_function."
    )


def validate_rollout_reward_mode(config: RLConfig) -> None:
    error = rollout_reward_mode_error(config)
    if error is not None:
        raise ValueError(error)

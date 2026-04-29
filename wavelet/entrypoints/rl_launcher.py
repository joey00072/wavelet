from __future__ import annotations

import os
import sys
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.policy import create_policy_inference_engine
from wavelet.orchestrator.launcher import (
    RoleSpec,
    close_handles,
    create_role_launcher,
    terminate_remaining,
    wait_for_roles,
)
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.trainer.rl_trainer import RLTrainer
from wavelet.orchestrator.queue import (
    FileSystemPolicyReceiver,
    FileSystemRolloutReceiver,
    RolloutBatch,
)
from wavelet.utils.config import load_config
from wavelet.utils.pathing import (
    get_config_dir,
    resolve_resume_checkpoint,
    validate_output_dir,
)
from wavelet.utils.serialization import dump_yaml


@dataclass(slots=True)
class StepTimes:
    started_at: float
    update_weights: float = 0.0
    generate_completions: float = 0.0
    wait_for_batch: float = 0.0
    load_data: float = 0.0
    train_until: float = 0.0
    export_policy: float = 0.0


def _write_subconfigs(config: RLConfig, trainer_config: RLConfig | None = None) -> None:
    config_dir = get_config_dir(config.output_dir)
    full_trainer_config = trainer_config or config
    dump_yaml(
        config_dir / "rl_trainer.yaml",
        full_trainer_config.model_dump(mode="json", exclude_none=True),
    )
    dump_yaml(
        config_dir / "rl_orchestrator.yaml",
        config.model_dump(mode="json", exclude_none=True),
    )
    dump_yaml(
        config_dir / "rl_inference.yaml",
        config.model_dump(mode="json", exclude_none=True),
    )


def _config_path_for_role(config: RLConfig, name: str, role_config: RLConfig) -> Path:
    config_path = get_config_dir(config.output_dir) / f"{name}.yaml"
    dump_yaml(config_path, role_config.model_dump(mode="json", exclude_none=True))
    return config_path


def _trainer_config_for_rollouts(config: RLConfig, rollout_path) -> RLConfig:
    return config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "source": "local",
                    "path": rollout_path,
                }
            )
        }
    )


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
            raise ValueError(
                f"launcher.mode='{config.launcher.mode}' requires "
                "launcher.trainer_cuda_visible_devices when using multiple "
                "inference replicas."
            )
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


def _vllm_base_urls(config: RLConfig, ports: list[int]) -> list[str]:
    return [
        f"http://{config.inference.http.host}:{port}/v1"
        for port in ports
    ]


def _inference_replica_config(config: RLConfig, *, port: int) -> RLConfig:
    return config.model_copy(
        update={
            "inference": config.inference.model_copy(
                update={
                    "http": config.inference.http.model_copy(update={"port": port}),
                }
            ),
            "launcher": config.launcher.model_copy(
                update={"inference_num_replicas": 1}
            ),
        }
    )


def _rollout_client_config(config: RLConfig, *, ports: list[int]) -> RLConfig:
    inference = config.inference.model_copy(
        update={
            "http": config.inference.http.model_copy(update={"ports": ports}),
        }
    )
    orchestrator = config.orchestrator
    if config.inference.vllm.server_backend == "openai" and config.lora is not None:
        orchestrator = orchestrator.model_copy(
            update={"verifier_model": config.policy_transfer.adapter_name}
        )
    if config.orchestrator.verifier_base_url is not None:
        return config.model_copy(
            update={"inference": inference, "orchestrator": orchestrator}
        )
    return config.model_copy(
        update={
            "inference": inference,
            "orchestrator": orchestrator.model_copy(
                update={"verifier_base_url": _vllm_base_urls(config, ports)}
            ),
        }
    )


def _load_policy_for_step(
    config: RLConfig,
    policy_receiver: FileSystemPolicyReceiver,
    inference_engine,
    *,
    step: int,
) -> float:
    if step <= 0 and not config.policy_transfer.export_initial:
        return 0.0
    started_at = perf_counter()
    policy = policy_receiver.wait_for_step(step)
    inference_engine.load_policy(policy.step_dir, step=policy.step)
    return perf_counter() - started_at


def _publish_rollout_timed(
    orchestrator: RLOrchestrator,
    *,
    step: int,
    inference_engine,
) -> tuple[RolloutBatch, float]:
    started_at = perf_counter()
    batch = orchestrator.publish(
        step=step,
        inference_engine=inference_engine,
    )
    return batch, perf_counter() - started_at


def _run_sync_rollout_loop(
    config: RLConfig,
    *,
    trainer: RLTrainer,
    receiver: FileSystemRolloutReceiver,
    policy_receiver: FileSystemPolicyReceiver,
    inference_engine,
    orchestrator: RLOrchestrator,
) -> None:
    target_step = config.max_steps or 1
    while trainer.step < target_step:
        timings = StepTimes(started_at=perf_counter())
        timings.update_weights = _load_policy_for_step(
            config,
            policy_receiver,
            inference_engine,
            step=trainer.step,
        )
        timings.generate_completions = _publish_rollout_timed(
            orchestrator,
            step=trainer.step,
            inference_engine=inference_engine,
        )[1]
        _consume_and_train_step(trainer, receiver, timings)
        _log_step_times(trainer, timings)


def _run_async_rollout_loop(
    config: RLConfig,
    *,
    trainer: RLTrainer,
    receiver: FileSystemRolloutReceiver,
    policy_receiver: FileSystemPolicyReceiver,
    inference_engine,
    orchestrator: RLOrchestrator,
) -> None:
    target_step = config.max_steps or 1
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="wavelet-rollout"
    ) as pool:
        _load_policy_for_step(
            config,
            policy_receiver,
            inference_engine,
            step=trainer.step,
        )
        pending: Future[tuple[RolloutBatch, float]] | None = pool.submit(
            _publish_rollout_timed,
            orchestrator,
            step=trainer.step,
            inference_engine=inference_engine,
        )
        while trainer.step < target_step:
            if pending is None:
                raise RuntimeError(
                    "Async rollout loop lost its pending rollout future."
                )
            timings = StepTimes(started_at=perf_counter())
            wait_started_at = perf_counter()
            _published, timings.generate_completions = pending.result()
            received = receiver.wait()
            timings.wait_for_batch = perf_counter() - wait_started_at

            next_step = trainer.step + 1
            if next_step < target_step:
                timings.update_weights = _load_policy_for_step(
                    config,
                    policy_receiver,
                    inference_engine,
                    step=trainer.step,
                )
                pending = pool.submit(
                    _publish_rollout_timed,
                    orchestrator,
                    step=next_step,
                    inference_engine=inference_engine,
                )
            else:
                pending = None

            _load_and_train_received_batch(trainer, received, timings)
            _log_step_times(trainer, timings)


def _consume_and_train_step(
    trainer: RLTrainer,
    receiver: FileSystemRolloutReceiver,
    timings: StepTimes,
) -> None:
    wait_started_at = perf_counter()
    received = receiver.wait()
    timings.wait_for_batch = perf_counter() - wait_started_at
    _load_and_train_received_batch(trainer, received, timings)


def _load_and_train_received_batch(
    trainer: RLTrainer,
    received: RolloutBatch,
    timings: StepTimes,
) -> None:
    load_started_at = perf_counter()
    trainer.load_rollout_path(received.path)
    timings.load_data = perf_counter() - load_started_at

    train_started_at = perf_counter()
    trainer.train_until(trainer.step + 1)
    timings.train_until = perf_counter() - train_started_at

    export_started_at = perf_counter()
    trainer.export_policy(step=trainer.step)
    timings.export_policy = perf_counter() - export_started_at


def _log_step_times(
    trainer: RLTrainer,
    timings: StepTimes,
) -> None:
    if trainer.monitor is None:
        return
    trainer.monitor.log(
        {
            "time/step": perf_counter() - timings.started_at,
            "time/update_weights": timings.update_weights,
            "time/generate_completions": timings.generate_completions,
            "time/wait_for_batch": timings.wait_for_batch,
            "time/load_data": timings.load_data,
            "time/train_until": timings.train_until,
            "time/export_policy": timings.export_policy,
        },
        trainer.step,
    )


def _run_process_launcher(config: RLConfig) -> int:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Do not run 'wavelet rl' under torchrun. For distributed RL, run "
            "'wavelet rl-inference' once and 'torchrun -m wavelet rl-trainer' for "
            "the trainer ranks."
        )

    inference_replicas = config.launcher.inference_num_replicas
    inference_ports = _http_ports(config, inference_replicas)
    rollout_config = _rollout_client_config(config, ports=inference_ports)

    _write_subconfigs(rollout_config, config)
    config_dir = get_config_dir(config.output_dir)
    trainer_config_path = config_dir / "rl_trainer.yaml"
    inference_config_path = config_dir / "rl_inference.yaml"
    dump_yaml(
        inference_config_path,
        rollout_config.model_dump(mode="json", exclude_none=True),
    )

    roles: list[RoleSpec] = []
    if config.inference.mode == "vllm_http":
        inference_devices = _as_device_groups(
            config.launcher.inference_cuda_visible_devices,
            inference_replicas,
        )
        for replica, (port, cuda_visible_devices) in enumerate(
            zip(inference_ports, inference_devices, strict=True)
        ):
            replica_config = _inference_replica_config(config, port=port)
            replica_config_path = _config_path_for_role(
                config,
                f"rl_vllm_server_{replica}",
                replica_config,
            )
            roles.append(
                RoleSpec(
                    name=f"vllm_server_{replica}",
                    command=(
                        "rl-vllm-openai-server"
                        if config.inference.vllm.server_backend == "openai"
                        else "rl-vllm-server"
                    ),
                    config_path=replica_config_path,
                    log_name=f"rl_vllm_server_{replica}",
                    cuda_visible_devices=cuda_visible_devices,
                    service=True,
                )
            )
    roles.extend(
        [
            RoleSpec(
                name="trainer",
                command="rl-trainer",
                config_path=trainer_config_path,
                log_name="rl_trainer",
                cuda_visible_devices=_trainer_device_group(config),
                torchrun_nproc_per_node=config.launcher.trainer_num_processes,
            ),
            RoleSpec(
                name="inference",
                command="rl-inference",
                config_path=inference_config_path,
                log_name="rl_inference",
                cuda_visible_devices=None,
            ),
        ]
    )

    launcher = create_role_launcher(config)
    handles = []
    try:
        service_roles = [role for role in roles if role.service]
        job_roles = [role for role in roles if not role.service]
        handles = [launcher.start(role) for role in service_roles]
        if config.inference.mode == "vllm_http":
            for port in inference_ports:
                _wait_for_vllm_http_server(config, port=port)
            if config.launcher.mode == "colocate_sleep":
                _sleep_vllm_http_server(config, port=inference_ports[0])
        handles.extend(launcher.start(role) for role in job_roles)
        wait_for_roles(
            handles,
            poll_interval_seconds=config.launcher.poll_interval_seconds,
        )
    finally:
        terminate_remaining(handles)
        close_handles(handles)
    print(f"Published rollout batches under {config.output_dir / 'rollouts'}")
    return 0


def _wait_for_vllm_http_server(config: RLConfig, *, port: int | None = None) -> None:
    port = config.inference.http.port if port is None else port
    url = f"http://{config.inference.http.host}:{port}/health"
    deadline = time.monotonic() + config.inference.http.startup_timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(config.launcher.poll_interval_seconds)
    raise TimeoutError(
        f"Timed out waiting for vLLM HTTP server at {url}"
    ) from last_error


def _sleep_vllm_http_server(config: RLConfig, *, port: int | None = None) -> None:
    port = config.inference.http.port if port is None else port
    url = f"http://{config.inference.http.host}:{port}/sleep"
    data = b'{"level": 1}'
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(
        request,
        timeout=config.inference.http.request_timeout_seconds,
    ):
        return None


def _run_integrated_launcher(config: RLConfig) -> int:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and config.orchestrator.enabled:
        raise RuntimeError(
            "The integrated RL launcher is single-process only. For distributed RL, "
            "run 'wavelet rl-inference' once and 'torchrun -m wavelet rl-trainer' "
            "for trainer ranks, or use launcher.mode='process' outside torchrun."
        )

    if config.orchestrator.enabled:
        queue_dir = config.transport.queue_dir or (config.output_dir / "rollouts")
        trainer_config = config
        _write_subconfigs(config, trainer_config)
        trainer = RLTrainer(trainer_config)
        trainer.setup()
        if not all(
            hasattr(trainer, attr)
            for attr in (
                "load_rollout_path",
                "train_until",
                "finalize",
                "export_policy",
            )
        ):
            rollout_path = RLOrchestrator(config).materialize()
            trainer_config = _trainer_config_for_rollouts(config, rollout_path)
            _write_subconfigs(config, trainer_config)
            trainer = RLTrainer(trainer_config)
            trainer.setup()
            trainer.train()
            return 0
        receiver = FileSystemRolloutReceiver(
            config.output_dir,
            config.transport,
            start_step=getattr(trainer, "step", 0),
        )
        policy_receiver = FileSystemPolicyReceiver(
            config.output_dir,
            config.policy_transfer,
            start_step=getattr(trainer, "step", 0),
        )
        inference_engine = create_policy_inference_engine(config)
        inference_engine.setup()
        orchestrator = RLOrchestrator(config)
        try:
            trainer.export_policy(step=trainer.step)
            if (
                config.orchestrator.max_async_level > 0
                and config.orchestrator.max_off_policy_steps > 0
            ):
                _run_async_rollout_loop(
                    config,
                    trainer=trainer,
                    receiver=receiver,
                    policy_receiver=policy_receiver,
                    inference_engine=inference_engine,
                    orchestrator=orchestrator,
                )
            else:
                _run_sync_rollout_loop(
                    config,
                    trainer=trainer,
                    receiver=receiver,
                    policy_receiver=policy_receiver,
                    inference_engine=inference_engine,
                    orchestrator=orchestrator,
                )
        except Exception:
            trainer.finalize(status="failed")
            raise
        trainer.finalize(status="completed")
        print(f"Published rollout batches under {queue_dir}")
    else:
        rollout_path = config.data.path
        trainer_config = _trainer_config_for_rollouts(config, rollout_path)
        _write_subconfigs(config, trainer_config)
        trainer = RLTrainer(trainer_config)
        trainer.setup()
        trainer.train()
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    config = load_config(RLConfig, argv)
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    validate_output_dir(
        config.output_dir,
        resuming=resuming,
        clean=config.clean_output_dir,
    )
    if resuming:
        assert config.ckpt is not None
        resolve_resume_checkpoint(config.output_dir, config.ckpt.resume_step)
    dump_yaml(
        get_config_dir(config.output_dir) / "rl.yaml",
        config.model_dump(mode="json", exclude_none=True),
    )

    if config.dry_run:
        if config.orchestrator.enabled and config.inference.mode == "passthrough":
            rollout_path = RLOrchestrator(config).materialize()
        elif config.orchestrator.enabled:
            rollout_path = config.output_dir / "rollouts"
        else:
            rollout_path = config.data.path
        _write_subconfigs(config, _trainer_config_for_rollouts(config, rollout_path))
        print("Dry run - configuration loaded successfully")
        print(f"Materialized rollouts: {rollout_path}")
        print(config.model_dump_json(indent=2))
        return 0

    if (
        config.launcher.mode in {"process", "colocate", "colocate_sleep"}
        and config.orchestrator.enabled
    ):
        return _run_process_launcher(config)
    return _run_integrated_launcher(config)


if __name__ == "__main__":
    sys.exit(main())

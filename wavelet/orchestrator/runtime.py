from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl import count_nonempty_jsonl_rows
from wavelet.inference.policy import (
    create_policy_inference_engine,
    expected_served_model_names,
    require_expected_served_model,
)
from wavelet.monitor import setup_config_logger
from wavelet.orchestrator.launcher import (
    RoleHandle,
    RoleSpec,
    close_handles,
    create_role_launcher,
    terminate_remaining,
    wait_for_roles,
)
from wavelet.orchestrator.placement import (
    device_groups as _as_device_groups,
)
from wavelet.orchestrator.placement import (
    http_ports as _http_ports,
)
from wavelet.orchestrator.placement import (
    nccl_inference_ranks as _nccl_inference_ranks,
)
from wavelet.orchestrator.placement import (
    trainer_device_group as _trainer_device_group,
)
from wavelet.orchestrator.placement import (
    validate_device_groups as _validate_device_groups,
)
from wavelet.orchestrator.placement import (
    validate_rollout_reward_mode as _validate_rollout_reward_mode,
)
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.schedule import (
    latest_exported_policy_step_at_or_before as _latest_exported_policy_step,
)
from wavelet.orchestrator.schedule import (
    required_policy_step as _required_policy_step,
)
from wavelet.orchestrator.schedule import target_steps as _target_steps
from wavelet.orchestrator.scheduler import (
    IntegratedRolloutScheduler,
    discard_rollout_batches_after_resume,
)
from wavelet.trainer.rl_trainer import RLTrainer
from wavelet.transport.queue import (
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


def _data_paths(path: Path | list[Path] | None) -> list[Path]:
    if path is None:
        return []
    return list(path) if isinstance(path, list) else [path]


def _write_subconfigs(config: RLConfig, trainer_config: RLConfig | None = None) -> None:
    config_dir = get_config_dir(config.output_dir)
    full_trainer_config = trainer_config or config
    dump_yaml(config_dir / "rl_trainer.yaml", _role_config_payload(full_trainer_config))
    dump_yaml(config_dir / "rl_orchestrator.yaml", _role_config_payload(config))
    dump_yaml(config_dir / "rl_inference.yaml", _role_config_payload(config))


def _role_config_payload(config: RLConfig) -> dict:
    return config.model_dump(mode="json", exclude_none=False)


def _config_path_for_role(config: RLConfig, name: str, role_config: RLConfig) -> Path:
    config_path = get_config_dir(config.output_dir) / f"{name}.yaml"
    dump_yaml(config_path, _role_config_payload(role_config))
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


def _vllm_base_urls(config: RLConfig, ports: list[int]) -> list[str]:
    return [f"http://{config.inference.http.host}:{port}/v1" for port in ports]


def _config_with_nccl_inference_world_size(
    config: RLConfig,
    *,
    inference_replicas: int,
) -> RLConfig:
    if config.policy_transfer.type != "nccl" or config.inference.mode != "vllm_http":
        return config
    inference_devices = _as_device_groups(
        config.launcher.inference_cuda_visible_devices,
        inference_replicas,
    )
    inference_world_size = sum(
        _nccl_inference_ranks(config, cuda_visible_devices)
        for cuda_visible_devices in inference_devices
    )
    return config.model_copy(
        update={
            "policy_transfer": config.policy_transfer.model_copy(
                update={
                    "nccl_inference_world_size": inference_world_size,
                    "nccl_rank_offset": 1,
                }
            )
        }
    )


def _inference_replica_config(
    config: RLConfig,
    *,
    port: int,
    nccl_rank_offset: int | None = None,
) -> RLConfig:
    policy_transfer = config.policy_transfer
    if nccl_rank_offset is not None:
        policy_transfer = policy_transfer.model_copy(
            update={"nccl_rank_offset": nccl_rank_offset}
        )
    return config.model_copy(
        update={
            "inference": config.inference.model_copy(
                update={
                    "http": config.inference.http.model_copy(update={"port": port}),
                }
            ),
            "policy_transfer": policy_transfer,
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
    serves_adapter = config.lora is not None and (
        _target_steps(config) > 0 or config.model.adapter_path is not None
    )
    if config.inference.vllm.server_backend == "openai" and serves_adapter:
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


def _policy_step_for_trainer_step(
    config: RLConfig,
    policy_receiver: FileSystemPolicyReceiver,
    *,
    step: int,
) -> int | None:
    """Resolve the newest policy the trainer has exported at or before ``step``.

    The trainer only exports every ``export_every_steps`` (plus forced exports
    on resume), so waiting for the exact trainer step would block forever.
    """
    if step in policy_receiver.available_steps():
        return step
    return _latest_exported_policy_step(config, step)


def _load_policy_for_step(
    config: RLConfig,
    policy_receiver: FileSystemPolicyReceiver,
    inference_engine,
    *,
    step: int,
) -> float:
    policy_step = _policy_step_for_trainer_step(config, policy_receiver, step=step)
    if policy_step is None:
        return 0.0
    if getattr(inference_engine, "policy_step", None) == policy_step:
        return 0.0
    started_at = perf_counter()
    policy = policy_receiver.wait_for_step(policy_step)
    inference_engine.load_policy(policy.step_dir, step=policy.step)
    return perf_counter() - started_at


def _pipelined_rollouts(config: RLConfig) -> bool:
    """Whether rollouts for step S+1 may be generated on policy S.

    The pipelined loop always produces lag-1 rollouts, so it is only valid when
    the trainer's freshness contract admits a one-step-old policy.
    """
    return _required_policy_step(config, 1) < 1


def _publish_rollout_timed(
    orchestrator: RLOrchestrator,
    *,
    step: int,
    inference_engine,
) -> tuple[RolloutBatch, float]:
    started_at = perf_counter()
    policy_step = getattr(inference_engine, "policy_step", None)
    if policy_step is None:
        # No policy has been loaded yet, so the engine serves the base weights,
        # which are policy step 0.
        policy_step = 0
    batch = orchestrator.publish(
        step=step,
        inference_engine=inference_engine,
        policy_step=policy_step,
    )
    return batch, perf_counter() - started_at


def _run_rollout_loop(
    config: RLConfig,
    *,
    trainer: RLTrainer,
    receiver: FileSystemRolloutReceiver,
    policy_receiver: FileSystemPolicyReceiver,
    inference_engine,
    orchestrator: RLOrchestrator,
    pipelined: bool,
) -> None:
    target_step = _target_steps(config)
    if not pipelined:
        timings: StepTimes | None = None

        def prepare_policy(step: int) -> None:
            nonlocal timings
            timings = StepTimes(started_at=perf_counter())
            timings.update_weights = _load_policy_for_step(
                config,
                policy_receiver,
                inference_engine,
                step=step,
            )

        def publish(step: int) -> None:
            assert timings is not None
            timings.generate_completions = _publish_rollout_timed(
                orchestrator,
                step=step,
                inference_engine=inference_engine,
            )[1]

        def consume_and_train() -> None:
            assert timings is not None
            _consume_and_train_step(trainer, receiver, timings)

        def after_step() -> None:
            assert timings is not None
            _log_step_times(trainer, timings)

        IntegratedRolloutScheduler(
            target_step=target_step,
            current_step=lambda: trainer.step,
            prepare_policy=prepare_policy,
            publish=publish,
            consume_and_train=consume_and_train,
            after_step=after_step,
        ).run()
        return

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
                raise RuntimeError("Rollout scheduler lost its pending future.")
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
    trainer_step_before = trainer.step
    row_count = count_nonempty_jsonl_rows(
        received.path,
        description="Rollout batch",
    )
    trainer.validate_rollout_batch(received, row_count=row_count)
    trainer.record_rollout_claim(
        received,
        trainer_step_before=trainer_step_before,
    )
    load_started_at = perf_counter()
    trainer.load_rollout_path(received.path)
    timings.load_data = perf_counter() - load_started_at

    train_started_at = perf_counter()
    trainer.train_until(trainer.step + 1)
    timings.train_until = perf_counter() - train_started_at
    trainer.record_rollout_consumed(
        received,
        trainer_step_before=trainer_step_before,
        optimizer_step_completed=True,
    )

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


def _role_specs(
    config: RLConfig,
    *,
    trainer_config_path: Path,
    inference_config_path: Path,
    inference_ports: list[int],
) -> list[RoleSpec]:
    inference_replicas = len(inference_ports)
    eval_only = _target_steps(config) == 0
    roles: list[RoleSpec] = []
    if config.inference.mode == "vllm_http":
        inference_devices = _as_device_groups(
            config.launcher.inference_cuda_visible_devices,
            inference_replicas,
        )
        nccl_rank_offset = 1
        for replica, (port, cuda_visible_devices) in enumerate(
            zip(inference_ports, inference_devices, strict=True)
        ):
            replica_rank_offset = None
            if config.policy_transfer.type == "nccl":
                replica_rank_offset = nccl_rank_offset
                nccl_rank_offset += _nccl_inference_ranks(
                    config,
                    cuda_visible_devices,
                )
            replica_config = _inference_replica_config(
                config,
                port=port,
                nccl_rank_offset=replica_rank_offset,
            )
            replica_config_path = _config_path_for_role(
                config,
                f"inference_server_{replica}",
                replica_config,
            )
            roles.append(
                RoleSpec(
                    name=f"inference_server_{replica}",
                    command=(
                        "inference-server"
                        if config.inference.vllm.server_backend == "openai"
                        else "native-inference-server"
                    ),
                    config_path=replica_config_path,
                    log_name=f"inference_server_{replica}",
                    cuda_visible_devices=cuda_visible_devices,
                    service=True,
                )
            )
    if not eval_only:
        roles.append(
            RoleSpec(
                name="trainer",
                command="rl-trainer",
                config_path=trainer_config_path,
                log_name="rl_trainer",
                cuda_visible_devices=_trainer_device_group(config),
                torchrun_nproc_per_node=config.launcher.trainer_num_processes,
            )
        )
    roles.append(
        RoleSpec(
            name="inference",
            command="rl-inference",
            config_path=inference_config_path,
            log_name="rl_inference",
            cuda_visible_devices=None,
        )
    )
    return roles


def _run_process_launcher(config: RLConfig) -> int:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Do not run 'wavelet rl' under torchrun. For distributed RL, run "
            "'wavelet rl-inference' once and 'torchrun -m wavelet rl-trainer' for "
            "the trainer ranks."
        )

    inference_replicas = config.launcher.inference_num_replicas
    config = _config_with_nccl_inference_world_size(
        config,
        inference_replicas=inference_replicas,
    )
    inference_ports = _http_ports(config, inference_replicas)
    rollout_config = _rollout_client_config(config, ports=inference_ports)

    _write_subconfigs(rollout_config, config)
    config_dir = get_config_dir(config.output_dir)
    trainer_config_path = config_dir / "rl_trainer.yaml"
    inference_config_path = config_dir / "rl_inference.yaml"
    dump_yaml(inference_config_path, _role_config_payload(rollout_config))
    roles = _role_specs(
        config,
        trainer_config_path=trainer_config_path,
        inference_config_path=inference_config_path,
        inference_ports=inference_ports,
    )

    launcher = create_role_launcher(config)
    handles = []
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        service_roles = [role for role in roles if role.service]
        job_roles = [role for role in roles if not role.service]
        handles = [launcher.start(role) for role in service_roles]
        if config.inference.mode == "vllm_http":
            for port, handle in zip(inference_ports, handles, strict=True):
                _wait_for_vllm_http_server(config, port=port, handle=handle)
            if config.launcher.mode == "colocate_sleep":
                _sleep_vllm_http_servers(config, ports=inference_ports)
        handles.extend(launcher.start(role) for role in job_roles)
        wait_for_roles(
            handles,
            poll_interval_seconds=config.launcher.poll_interval_seconds,
        )
    finally:
        try:
            terminate_remaining(handles)
            close_handles(handles)
        finally:
            launcher.close()
            signal.signal(signal.SIGTERM, previous_sigterm)
    print(f"Published rollout batches under {config.output_dir / 'rollouts'}")
    return 0


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    # Role processes run in their own sessions, so a SIGTERM to the launcher
    # (systemd, SLURM, ``timeout``) would otherwise orphan every GPU process.
    raise KeyboardInterrupt(f"received signal {signum}")


def _wait_for_vllm_http_server(
    config: RLConfig,
    *,
    port: int | None = None,
    handle: RoleHandle | None = None,
) -> None:
    port = config.inference.http.port if port is None else port
    base_url = f"http://{config.inference.http.host}:{port}"
    health_url = f"{base_url}/health"
    models_url = f"{base_url}/v1/models"
    deadline = time.monotonic() + config.inference.http.startup_timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if handle is not None:
            code = handle.poll()
            if code is not None:
                raise RuntimeError(
                    f"vLLM HTTP server exited with code {code} before {health_url} "
                    f"became healthy. Check '{handle.log_path}'."
                )
        try:
            with urllib.request.urlopen(health_url, timeout=5.0):
                pass
            with urllib.request.urlopen(models_url, timeout=5.0) as response:
                models = json.loads(response.read().decode("utf-8"))
            require_expected_served_model(
                models,
                expected_names=expected_served_model_names(config),
                server=base_url,
            )
            return
        except OSError as exc:
            last_error = exc
            time.sleep(config.launcher.poll_interval_seconds)
    raise TimeoutError(
        f"Timed out waiting for vLLM HTTP server at {health_url}"
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
        return


def _sleep_vllm_http_servers(config: RLConfig, *, ports: list[int]) -> None:
    if len(ports) == 1:
        _sleep_vllm_http_server(config, port=ports[0])
        return
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        futures = [
            executor.submit(_sleep_vllm_http_server, config, port=port)
            for port in ports
        ]
        for future in futures:
            future.result()


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
        if trainer.resume_checkpoint_dir is not None:
            discard_rollout_batches_after_resume(config, start_step=trainer.step)
        receiver = FileSystemRolloutReceiver(
            config.output_dir,
            config.transport,
            start_step=getattr(trainer, "step", 0),
            events_dir=config.output_dir / "events",
        )
        policy_receiver = FileSystemPolicyReceiver(
            config.output_dir,
            config.policy_transfer,
            start_step=getattr(trainer, "step", 0),
            events_dir=config.output_dir / "events",
        )
        inference_engine = create_policy_inference_engine(config)
        orchestrator = RLOrchestrator(config)
        status = "failed"
        try:
            inference_engine.setup()
            trainer.export_policy(
                step=trainer.step,
                force=trainer.resume_checkpoint_dir is not None,
            )
            pipelined = _pipelined_rollouts(config)
            _run_rollout_loop(
                config,
                trainer=trainer,
                receiver=receiver,
                policy_receiver=policy_receiver,
                inference_engine=inference_engine,
                orchestrator=orchestrator,
                pipelined=pipelined,
            )
            status = "completed"
        finally:
            from wavelet.orchestrator.envs import _teardown_cached_verifier_envs

            try:
                asyncio.run(_teardown_cached_verifier_envs())
            finally:
                try:
                    inference_engine.close()
                finally:
                    trainer.finalize(status=status)
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
    _validate_rollout_reward_mode(config)
    _validate_device_groups(config)
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    validate_output_dir(
        config.output_dir,
        resuming=resuming,
        clean=config.clean_output_dir,
        protected_paths=(config.model.adapter_path, *_data_paths(config.data.path)),
    )
    if resuming:
        assert config.ckpt is not None
        resolve_resume_checkpoint(
            config.checkpoint_output_dir,
            config.ckpt.resume_step,
        )
    dump_yaml(
        get_config_dir(config.output_dir) / "rl.yaml",
        _role_config_payload(config),
    )

    if config.dry_run:
        if config.orchestrator.enabled:
            rollout_path = config.output_dir / "rollouts"
        else:
            rollout_path = config.data.path
        _write_subconfigs(config, _trainer_config_for_rollouts(config, rollout_path))
        print("Dry run - configuration loaded successfully")
        print(f"Materialized rollouts: {rollout_path}")
        print(config.model_dump_json(indent=2))
        return 0

    setup_config_logger("rl", config)
    if (
        config.launcher.mode in {"process", "colocate", "colocate_sleep"}
        and config.orchestrator.enabled
    ):
        return _run_process_launcher(config)
    return _run_integrated_launcher(config)


if __name__ == "__main__":
    sys.exit(main())

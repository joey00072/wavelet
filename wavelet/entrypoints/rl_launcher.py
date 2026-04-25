from __future__ import annotations

import os
import sys
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
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

    _write_subconfigs(config, config)
    config_dir = get_config_dir(config.output_dir)
    trainer_config_path = config_dir / "rl_trainer.yaml"
    inference_config_path = config_dir / "rl_inference.yaml"

    roles: list[RoleSpec] = []
    if config.inference.mode == "vllm_http":
        roles.append(
            RoleSpec(
                name="vllm_server",
                command="rl-vllm-server",
                config_path=inference_config_path,
                log_name="rl_vllm_server",
                cuda_visible_devices=config.launcher.inference_cuda_visible_devices,
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
                cuda_visible_devices=config.launcher.trainer_cuda_visible_devices,
            ),
            RoleSpec(
                name="inference",
                command="rl-inference",
                config_path=inference_config_path,
                log_name="rl_inference",
                cuda_visible_devices=config.launcher.inference_cuda_visible_devices,
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
            _wait_for_vllm_http_server(config)
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


def _wait_for_vllm_http_server(config: RLConfig) -> None:
    url = f"http://{config.inference.http.host}:{config.inference.http.port}/health"
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

    if config.launcher.mode == "process" and config.orchestrator.enabled:
        return _run_process_launcher(config)
    return _run_integrated_launcher(config)


if __name__ == "__main__":
    sys.exit(main())

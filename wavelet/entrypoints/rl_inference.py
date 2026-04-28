from __future__ import annotations

import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from time import perf_counter

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.policy import create_policy_inference_engine
from wavelet.orchestrator.eval_utils import compute_eval_policy_step
from wavelet.orchestrator.queue import FileSystemPolicyReceiver, FileSystemRolloutSender
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.utils.config import load_config


def _perf_enabled() -> bool:
    return os.environ.get("WAVELET_PERF_LOG", "").lower() in {"1", "true", "yes", "on"}


def _preload_rollout_resources(config: RLConfig) -> None:
    if (
        config.orchestrator.custom_rollout_function
        != "wavelet.orchestrator.verifiers:generate_rollouts"
    ):
        return
    try:
        import verifiers as vf
    except ImportError:
        return

    from wavelet.orchestrator.verifiers import _load_cached_env

    env_id = config.orchestrator.verifier_env_id
    if env_id is None:
        return
    _load_cached_env(vf, env_id, config.orchestrator.verifier_env_args)


def _required_policy_step(config: RLConfig, rollout_step: int) -> int:
    """Oldest policy step allowed for a rollout under the async window."""
    async_level = config.orchestrator.max_async_level
    off_policy_steps = config.orchestrator.max_off_policy_steps
    if off_policy_steps > 0:
        async_level = min(async_level, off_policy_steps)
    return max(rollout_step - async_level, 0)


def _next_exported_policy_step(config: RLConfig, required_step: int) -> int:
    if required_step <= 0 and config.policy_transfer.export_initial:
        return 0
    interval = config.policy_transfer.export_every_steps
    return ((max(required_step, 1) + interval - 1) // interval) * interval


def _policy_step_to_load(
    config: RLConfig,
    policy_receiver: FileSystemPolicyReceiver,
    *,
    rollout_step: int,
    loaded_policy_step: int | None,
) -> int | None:
    required_step = _required_policy_step(config, rollout_step)
    available_steps = policy_receiver.available_steps()
    available_newer = [
        step
        for step in available_steps
        if step >= required_step
        and (loaded_policy_step is None or step > loaded_policy_step)
    ]
    if available_newer:
        return max(available_newer)
    if loaded_policy_step is None or loaded_policy_step < required_step:
        return _next_exported_policy_step(config, required_step)
    return None


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    policy_receiver = FileSystemPolicyReceiver(
        config.output_dir,
        config.policy_transfer,
    )
    inference_engine = create_policy_inference_engine(config)
    inference_engine.setup()
    orchestrator = RLOrchestrator(config)
    target_step = config.max_steps or 1
    _preload_rollout_resources(config)
    loaded_policy_step: int | None = None
    last_eval_steps = {
        env.resolved_name: -1
        for env in config.eval.env
    } if config.eval is not None else {}
    prefetch_steps = max(1, min(4, config.orchestrator.max_async_level, target_step))
    next_step_to_submit = 0
    next_step_to_publish = 0
    rollout_sender = FileSystemRolloutSender(config.output_dir, config.transport)
    pending: dict[Future[tuple[int, object, float]], int] = {}

    def submit_step(
        pool: ThreadPoolExecutor,
        step: int,
    ) -> None:
        future = pool.submit(_publish_step, orchestrator, step, inference_engine)
        pending[future] = step

    with ThreadPoolExecutor(
        max_workers=prefetch_steps,
        thread_name_prefix="wavelet-inference-step",
    ) as pool:
        while next_step_to_submit < target_step or pending:
            while (
                next_step_to_submit < target_step
                and len(pending) < prefetch_steps
            ):
                step = next_step_to_submit
                next_step_to_submit += 1
                step_started_at = perf_counter()
                policy_step = _policy_step_to_load(
                    config,
                    policy_receiver,
                    rollout_step=step,
                    loaded_policy_step=loaded_policy_step,
                )
                if policy_step is not None:
                    wait_started_at = perf_counter()
                    policy = policy_receiver.wait_for_step(policy_step)
                    wait_policy_seconds = perf_counter() - wait_started_at
                    load_started_at = perf_counter()
                    inference_engine.load_policy(policy.step_dir, step=policy.step)
                    load_policy_seconds = perf_counter() - load_started_at
                    loaded_policy_step = policy.step
                    _maybe_run_evals(
                        config,
                        orchestrator,
                        policy_step=policy.step,
                        rollout_step=step,
                        last_eval_steps=last_eval_steps,
                    )
                else:
                    wait_policy_seconds = 0.0
                    load_policy_seconds = 0.0
                submit_step(pool, step)
                if _perf_enabled():
                    print(
                        "WAVELET_PERF inference_submit "
                        f"step={step} wait_policy={wait_policy_seconds:.3f} "
                        f"load_policy={load_policy_seconds:.3f} "
                        f"submit={perf_counter() - step_started_at:.3f}",
                        flush=True,
                    )

            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                materialize_step = pending.pop(future)
                _, materialized_path, materialize_seconds = future.result()
                step = next_step_to_publish
                next_step_to_publish += 1
                publish_started_at = perf_counter()
                batch = rollout_sender.publish(materialized_path, step=step)
                publish_seconds = perf_counter() - publish_started_at
                if _perf_enabled():
                    print(
                        "WAVELET_PERF inference_step "
                        f"step={step} materialize_step={materialize_step} "
                        f"wait_policy=0.000 "
                        f"load_policy=0.000 "
                        f"publish={publish_seconds:.3f} "
                        f"materialize={materialize_seconds:.3f} "
                        f"total={materialize_seconds + publish_seconds:.3f}",
                        flush=True,
                    )
                print(batch.path)
    if config.eval is not None and config.eval.final_eval:
        final_policy_step = _final_eval_policy_step(config, target_step)
        if final_policy_step is None:
            return 0
        if loaded_policy_step is None or loaded_policy_step < final_policy_step:
            policy = policy_receiver.wait_for_step(final_policy_step)
            inference_engine.load_policy(policy.step_dir, step=policy.step)
            loaded_policy_step = policy.step
        _run_evals(
            config,
            orchestrator,
            policy_step=loaded_policy_step,
            rollout_step=target_step,
            envs=config.eval.env,
        )
    return 0


def _publish_step(
    orchestrator: RLOrchestrator,
    step: int,
    inference_engine,
):
    materialize_started_at = perf_counter()
    materialized_path = orchestrator.materialize(
        step=step,
        inference_engine=inference_engine,
    )
    materialize_seconds = perf_counter() - materialize_started_at
    return step, materialized_path, materialize_seconds


def _final_eval_policy_step(config: RLConfig, target_step: int) -> int | None:
    if target_step <= 0:
        return 0 if config.policy_transfer.export_initial else None
    interval = config.policy_transfer.export_every_steps
    final_step = (target_step // interval) * interval
    if final_step > 0:
        return final_step
    return 0 if config.policy_transfer.export_initial else None


def _maybe_run_evals(
    config: RLConfig,
    orchestrator: RLOrchestrator,
    *,
    policy_step: int,
    rollout_step: int,
    last_eval_steps: dict[str, int],
) -> None:
    if config.eval is None:
        return

    envs = []
    for env in config.eval.env:
        eval_step = compute_eval_policy_step(
            policy_step=policy_step,
            last_eval_step=last_eval_steps[env.resolved_name],
            interval=env.interval,
            eval_base_model=config.eval.eval_base_model,
        )
        if eval_step is None:
            continue
        last_eval_steps[env.resolved_name] = eval_step
        envs.append(env)
    _run_evals(
        config,
        orchestrator,
        policy_step=policy_step,
        rollout_step=rollout_step,
        envs=envs,
    )


def _run_evals(
    config: RLConfig,
    orchestrator: RLOrchestrator,
    *,
    policy_step: int,
    rollout_step: int,
    envs,
) -> None:
    if not envs:
        return
    if (
        config.orchestrator.custom_rollout_function
        != "wavelet.orchestrator.verifiers:generate_rollouts"
    ):
        raise ValueError("RL eval is currently supported for verifier rollouts only.")

    from wavelet.orchestrator.verifiers import evaluate_env

    for env in envs:
        metrics = evaluate_env(
            orchestrator,
            env,
            step=rollout_step,
            policy_step=policy_step,
        )
        print(json_dumps_compact(metrics), flush=True)


def json_dumps_compact(payload) -> str:
    import json

    return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())

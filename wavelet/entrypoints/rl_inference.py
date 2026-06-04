from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from time import perf_counter, sleep

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.inference.policy import create_policy_inference_engine
from wavelet.orchestrator.queue import (
    FileSystemPolicyReceiver,
    FileSystemRolloutSender,
    QueueEvent,
    append_event_best_effort,
    publish_adapter_policy_snapshot,
    utc_now,
)
from wavelet.orchestrator.policy_metadata import policy_metadata
from wavelet.orchestrator.metrics import log_rollout_metrics
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.schedule import (
    chunks_per_step as _chunks_per_step,
    policy_step_to_load as _policy_step_to_load,
    required_policy_step as _required_policy_step,
    rollout_chunk_examples as _rollout_chunk_examples,
    select_due_eval_envs,
    target_steps as _target_steps,
)
from wavelet.orchestrator.state_server import OrchestratorRunState, maybe_state_server
from wavelet.utils.config import load_config
from wavelet.utils.perf import emit_perf


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

    from wavelet.orchestrator.verifiers import (
        _load_cached_env,
        _verifier_extra_env_kwargs,
    )

    env_id = config.orchestrator.verifier_env_id
    if env_id is None:
        return
    _load_cached_env(
        vf,
        env_id,
        config.orchestrator.verifier_env_args,
        _verifier_extra_env_kwargs(config),
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    policy_receiver = FileSystemPolicyReceiver(
        config.output_dir,
        config.policy_transfer,
        events_dir=config.output_dir / "events",
    )
    if _target_steps(config) == 0 and config.model.adapter_path is not None:
        publish_adapter_policy_snapshot(
            config.output_dir,
            config.policy_transfer,
            config.model.adapter_path,
            step=0,
            metadata=policy_metadata(
                config=config,
                format_version=1,
                step=0,
                kind="adapter",
            ),
        )
    inference_engine = create_policy_inference_engine(config)
    inference_engine.setup()
    orchestrator = RLOrchestrator(config)
    target_step = _target_steps(config)
    _preload_rollout_resources(config)
    with maybe_state_server(config, target_step=target_step) as state:
        if state is not None:
            state.set_status("running", phase="inference")
        if _use_rolling_verifier_scheduler(config):
            return asyncio.run(
                _run_rolling_verifier_inference(
                    config=config,
                    orchestrator=orchestrator,
                    inference_engine=inference_engine,
                    policy_receiver=policy_receiver,
                    target_step=target_step,
                    state=state,
                )
            )
        if _use_streaming_native_scheduler(config):
            return _run_streaming_native_inference(
                config=config,
                orchestrator=orchestrator,
                inference_engine=inference_engine,
                policy_receiver=policy_receiver,
                target_step=target_step,
                state=state,
            )
        result = _run_prefetch_inference(
            config=config,
            orchestrator=orchestrator,
            inference_engine=inference_engine,
            policy_receiver=policy_receiver,
            target_step=target_step,
            state=state,
        )
        if state is not None:
            state.set_status("completed", phase="completed")
        return result


def _run_prefetch_inference(
    *,
    config: RLConfig,
    orchestrator: RLOrchestrator,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    target_step: int,
    state: OrchestratorRunState | None = None,
) -> int:
    loaded_policy_step: int | None = None
    last_eval_steps = (
        {env.resolved_name: -1 for env in config.eval.env}
        if config.eval is not None
        else {}
    )
    prefetch_steps = max(1, min(config.orchestrator.max_async_level, target_step))
    next_step_to_submit = 0
    next_step_to_publish = 0
    rollout_sender = FileSystemRolloutSender(config.output_dir, config.transport)
    pending: dict[Future[tuple[int, object, float, float]], int] = {}
    completed: dict[int, tuple[object, float, float]] = {}
    pending_policy_load: Future[tuple[int, float, float]] | None = None

    def submit_step(
        pool: ThreadPoolExecutor,
        step: int,
    ) -> None:
        future = pool.submit(
            _publish_step,
            orchestrator,
            rollout_sender,
            step,
            inference_engine,
            loaded_policy_step,
        )
        pending[future] = step

    def finish_policy_load(*, block: bool = False) -> bool:
        nonlocal loaded_policy_step, pending_policy_load
        if pending_policy_load is None:
            return False
        if not block and not pending_policy_load.done():
            return False
        policy_step, wait_policy_seconds, load_policy_seconds = (
            pending_policy_load.result()
        )
        loaded_policy_step = policy_step
        pending_policy_load = None
        if state is not None:
            state.update_policy(
                loaded_step=policy_step,
                pending_load=False,
                requested_step=None,
                available_tail=policy_receiver.available_steps()[-20:],
            )
        _maybe_run_evals(
            config,
            orchestrator,
            policy_step=policy_step,
            rollout_step=next_step_to_submit,
            last_eval_steps=last_eval_steps,
        )
        emit_perf(
            "policy_load",
            step=policy_step,
            wait_policy=wait_policy_seconds,
            load_policy=load_policy_seconds,
        )
        return True

    def collect_done(done) -> None:
        for future in done:
            materialize_step = pending.pop(future)
            _, batch, materialize_seconds, publish_seconds = future.result()
            completed[materialize_step] = (
                batch,
                materialize_seconds,
                publish_seconds,
            )
            if state is not None:
                state.mark_completed(
                    queue_step=materialize_step,
                    optimizer_step=materialize_step,
                    pending_count=len(pending),
                    completed_count=len(completed),
                )

    def publish_ready() -> bool:
        nonlocal next_step_to_publish
        published = False
        while next_step_to_publish in completed:
            batch, materialize_seconds, publish_seconds = completed.pop(
                next_step_to_publish
            )
            step = next_step_to_publish
            next_step_to_publish += 1
            if state is not None:
                state.mark_published(
                    queue_step=step,
                    optimizer_step=step,
                    path=str(batch.path),
                    next_queue_step_to_publish=next_step_to_publish,
                    completed_count=len(completed),
                )
            _sleep_for_colocated_sleep(config, inference_engine)
            emit_perf(
                "inference_step",
                step=step,
                wait_policy=0.0,
                load_policy=0.0,
                publish=publish_seconds,
                materialize=materialize_seconds,
                total=materialize_seconds + publish_seconds,
            )
            print(batch.path)
            published = True
        return published

    def collect_finished_rollouts() -> bool:
        done = [future for future in pending if future.done()]
        if not done:
            return False
        collect_done(done)
        return publish_ready()

    with (
        ThreadPoolExecutor(
            max_workers=prefetch_steps,
            thread_name_prefix="wavelet-inference-step",
        ) as pool,
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="wavelet-policy-load",
        ) as policy_pool,
    ):
        while next_step_to_submit < target_step or pending or pending_policy_load:
            finish_policy_load()
            if collect_finished_rollouts():
                continue

            submitted = False
            while next_step_to_submit < target_step and len(pending) < prefetch_steps:
                step = next_step_to_submit
                step_started_at = perf_counter()
                policy_step = _policy_step_to_load(
                    config,
                    policy_receiver,
                    rollout_step=step,
                    loaded_policy_step=loaded_policy_step,
                )
                if policy_step is not None:
                    required_policy_step = _required_policy_step(config, step)
                    must_load_before_rollout = (
                        loaded_policy_step is None
                        or loaded_policy_step < required_policy_step
                    )
                    if must_load_before_rollout:
                        if publish_ready():
                            continue
                        if pending_policy_load is not None:
                            if finish_policy_load():
                                continue
                            if pending:
                                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                                collect_done(done)
                                publish_ready()
                                continue
                            finish_policy_load(block=True)
                            continue
                        if pending:
                            pending_policy_load = policy_pool.submit(
                                _load_policy_step,
                                config,
                                inference_engine,
                                policy_receiver,
                                policy_step,
                            )
                            if state is not None:
                                state.update_policy(
                                    pending_load=True,
                                    requested_step=policy_step,
                                    available_tail=policy_receiver.available_steps()[
                                        -20:
                                    ],
                                )
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            collect_done(done)
                            publish_ready()
                            continue
                        (
                            loaded_policy_step,
                            wait_policy_seconds,
                            load_policy_seconds,
                        ) = _load_policy_step(
                            config,
                            inference_engine,
                            policy_receiver,
                            policy_step,
                        )
                        if state is not None:
                            state.update_policy(
                                loaded_step=loaded_policy_step,
                                pending_load=False,
                                requested_step=None,
                                available_tail=policy_receiver.available_steps()[-20:],
                            )
                        _maybe_run_evals(
                            config,
                            orchestrator,
                            policy_step=loaded_policy_step,
                            rollout_step=step,
                            last_eval_steps=last_eval_steps,
                        )
                    else:
                        if pending_policy_load is None:
                            pending_policy_load = policy_pool.submit(
                                _load_policy_step,
                                config,
                                inference_engine,
                                policy_receiver,
                                policy_step,
                            )
                            if state is not None:
                                state.update_policy(
                                    pending_load=True,
                                    requested_step=policy_step,
                                    available_tail=policy_receiver.available_steps()[
                                        -20:
                                    ],
                                )
                        wait_policy_seconds = 0.0
                        load_policy_seconds = 0.0
                else:
                    wait_policy_seconds = 0.0
                    load_policy_seconds = 0.0
                    _wake_for_colocated_sleep(config, inference_engine)
                next_step_to_submit += 1
                submit_step(pool, step)
                if state is not None:
                    state.update_rollouts(
                        next_queue_step_to_submit=next_step_to_submit,
                    )
                    state.mark_submitted(
                        queue_step=step,
                        optimizer_step=step,
                        pending_count=len(pending),
                    )
                submitted = True
                emit_perf(
                    "inference_submit",
                    step=step,
                    wait_policy=wait_policy_seconds,
                    load_policy=load_policy_seconds,
                    submit=perf_counter() - step_started_at,
                )

            if submitted or finish_policy_load():
                continue
            if pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                collect_done(done)
                publish_ready()
            elif pending_policy_load is not None:
                finish_policy_load(block=True)
    if config.eval is not None and config.eval.final_eval:
        final_policy_step = _final_eval_policy_step(config, target_step)
        if final_policy_step is None:
            return 0
        if loaded_policy_step is None or loaded_policy_step < final_policy_step:
            policy = policy_receiver.wait_for_step(final_policy_step)
            _wake_for_colocated_sleep(
                config,
                inference_engine,
                tags=["weights"],
            )
            inference_engine.load_policy(policy.step_dir, step=policy.step)
            _wake_for_colocated_sleep(
                config,
                inference_engine,
                tags=["kv_cache"],
            )
            loaded_policy_step = policy.step
        else:
            _wake_for_colocated_sleep(config, inference_engine)
        _run_evals(
            config,
            orchestrator,
            policy_step=loaded_policy_step,
            rollout_step=target_step,
            envs=config.eval.env,
        )
        _sleep_for_colocated_sleep(config, inference_engine)
    if state is not None:
        state.set_status("completed", phase="completed")
    return 0


def _use_rolling_verifier_scheduler(config: RLConfig) -> bool:
    return (
        config.launcher.mode == "process"
        and config.orchestrator.custom_rollout_function
        == "wavelet.orchestrator.verifiers:generate_rollouts"
        and config.orchestrator.max_async_level > 0
    )


def _use_streaming_native_scheduler(config: RLConfig) -> bool:
    return (
        config.launcher.mode == "process"
        and config.orchestrator.custom_rollout_function is None
        and config.orchestrator.max_async_level > 0
        and config.orchestrator.examples_per_step is not None
    )


def _run_streaming_native_inference(
    *,
    config: RLConfig,
    orchestrator: RLOrchestrator,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    target_step: int,
    state: OrchestratorRunState | None = None,
) -> int:
    rollout_sender = FileSystemRolloutSender(config.output_dir, config.transport)
    loaded_policy_step: int | None = None
    last_eval_steps = (
        {env.resolved_name: -1 for env in config.eval.env}
        if config.eval is not None
        else {}
    )
    chunk_examples = _rollout_chunk_examples(config)
    chunks_per_step = _chunks_per_step(config)
    target_chunks = target_step * chunks_per_step
    pending_chunk_limit = (
        config.orchestrator.max_pending_rollout_chunks
        or config.orchestrator.max_async_level * chunks_per_step
    )
    max_pending_chunks = max(1, min(pending_chunk_limit, target_chunks))
    next_queue_step_to_submit = 0
    next_queue_step_to_publish = 0
    published_queue_steps: set[int] = set()
    pending: dict[Future[tuple[int, object, float, float]], int] = {}
    completed: dict[int, tuple[object, float, float]] = {}
    pending_policy_load: Future[tuple[int, float, float]] | None = None

    def submit_chunk(pool: ThreadPoolExecutor, queue_step: int) -> None:
        optimizer_step = queue_step // chunks_per_step
        chunk_index = queue_step % chunks_per_step
        future = pool.submit(
            _publish_native_chunk,
            orchestrator,
            rollout_sender,
            optimizer_step,
            chunk_index,
            queue_step,
            chunk_examples,
            inference_engine,
            loaded_policy_step,
        )
        pending[future] = queue_step

    def finish_policy_load(*, block: bool = False) -> bool:
        nonlocal loaded_policy_step, pending_policy_load
        if pending_policy_load is None:
            return False
        if not block and not pending_policy_load.done():
            return False
        policy_step, wait_policy_seconds, load_policy_seconds = (
            pending_policy_load.result()
        )
        loaded_policy_step = policy_step
        pending_policy_load = None
        if state is not None:
            state.update_policy(
                loaded_step=policy_step,
                pending_load=False,
                requested_step=None,
                available_tail=policy_receiver.available_steps()[-20:],
            )
        _maybe_run_evals(
            config,
            orchestrator,
            policy_step=policy_step,
            rollout_step=next_queue_step_to_submit // chunks_per_step,
            last_eval_steps=last_eval_steps,
        )
        emit_perf(
            "policy_load",
            step=policy_step,
            wait_policy=wait_policy_seconds,
            load_policy=load_policy_seconds,
        )
        return True

    def collect_done(done) -> None:
        for future in done:
            queue_step = pending.pop(future)
            _, batch, materialize_seconds, publish_seconds = future.result()
            completed[queue_step] = (batch, materialize_seconds, publish_seconds)
            if state is not None:
                state.mark_completed(
                    queue_step=queue_step,
                    optimizer_step=queue_step // chunks_per_step,
                    chunk_index=queue_step % chunks_per_step,
                    pending_count=len(pending),
                    completed_count=len(completed),
                )

    def publish_ready() -> bool:
        nonlocal next_queue_step_to_publish
        published = False
        for queue_step in sorted(completed):
            batch, materialize_seconds, publish_seconds = completed.pop(queue_step)
            optimizer_step = queue_step // chunks_per_step
            chunk_index = queue_step % chunks_per_step
            published_queue_steps.add(queue_step)
            while next_queue_step_to_publish in published_queue_steps:
                published_queue_steps.remove(next_queue_step_to_publish)
                next_queue_step_to_publish += 1
            if state is not None:
                state.mark_published(
                    queue_step=queue_step,
                    optimizer_step=optimizer_step,
                    chunk_index=chunk_index,
                    path=str(batch.path),
                    next_queue_step_to_publish=next_queue_step_to_publish,
                    completed_count=len(completed),
                )
            _sleep_for_colocated_sleep(config, inference_engine)
            emit_perf(
                "inference_native_chunk",
                queue_step=queue_step,
                optimizer_step=optimizer_step,
                chunk_index=chunk_index,
                wait_policy=0.0,
                load_policy=0.0,
                publish=publish_seconds,
                materialize=materialize_seconds,
                total=materialize_seconds + publish_seconds,
            )
            print(batch.path)
            published = True
        return published

    def collect_finished_rollouts() -> bool:
        done = [future for future in pending if future.done()]
        if not done:
            return False
        collect_done(done)
        return publish_ready()

    def start_policy_load_if_available(optimizer_step: int) -> bool:
        nonlocal pending_policy_load
        if pending_policy_load is not None:
            return False
        if pending:
            return False
        policy_step = _policy_step_to_load(
            config,
            policy_receiver,
            rollout_step=optimizer_step,
            loaded_policy_step=loaded_policy_step,
        )
        if policy_step is None:
            return False
        pending_policy_load = policy_pool.submit(
            _load_policy_step,
            config,
            inference_engine,
            policy_receiver,
            policy_step,
        )
        if state is not None:
            state.update_policy(
                pending_load=True,
                requested_step=policy_step,
                available_tail=policy_receiver.available_steps()[-20:],
            )
        return True

    with (
        ThreadPoolExecutor(
            max_workers=max_pending_chunks,
            thread_name_prefix="wavelet-native-chunk",
        ) as pool,
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="wavelet-policy-load",
        ) as policy_pool,
    ):
        while (
            next_queue_step_to_submit < target_chunks or pending or pending_policy_load
        ):
            finish_policy_load()
            collect_finished_rollouts()
            if next_queue_step_to_submit < target_chunks and pending:
                start_policy_load_if_available(
                    next_queue_step_to_submit // chunks_per_step
                )

            submitted = False
            while (
                next_queue_step_to_submit < target_chunks
                and len(pending) < max_pending_chunks
            ):
                queue_step = next_queue_step_to_submit
                optimizer_step = queue_step // chunks_per_step
                step_started_at = perf_counter()
                policy_step = _policy_step_to_load(
                    config,
                    policy_receiver,
                    rollout_step=optimizer_step,
                    loaded_policy_step=loaded_policy_step,
                )
                if policy_step is not None:
                    required_policy_step = _required_policy_step(
                        config,
                        optimizer_step,
                    )
                    must_load_before_rollout = (
                        loaded_policy_step is None
                        or loaded_policy_step < required_policy_step
                    )
                    if must_load_before_rollout:
                        if publish_ready():
                            continue
                        if pending_policy_load is not None:
                            if finish_policy_load():
                                continue
                            if pending:
                                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                                collect_done(done)
                                publish_ready()
                                continue
                            finish_policy_load(block=True)
                            continue
                        if pending:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            collect_done(done)
                            publish_ready()
                            continue
                        (
                            loaded_policy_step,
                            wait_policy_seconds,
                            load_policy_seconds,
                        ) = _load_policy_step(
                            config,
                            inference_engine,
                            policy_receiver,
                            policy_step,
                        )
                        if state is not None:
                            state.update_policy(
                                loaded_step=loaded_policy_step,
                                pending_load=False,
                                requested_step=None,
                                available_tail=policy_receiver.available_steps()[-20:],
                            )
                        _maybe_run_evals(
                            config,
                            orchestrator,
                            policy_step=loaded_policy_step,
                            rollout_step=optimizer_step,
                            last_eval_steps=last_eval_steps,
                        )
                    else:
                        if pending_policy_load is None and not pending:
                            (
                                loaded_policy_step,
                                wait_policy_seconds,
                                load_policy_seconds,
                            ) = _load_policy_step(
                                config,
                                inference_engine,
                                policy_receiver,
                                policy_step,
                            )
                            if state is not None:
                                state.update_policy(
                                    loaded_step=loaded_policy_step,
                                    pending_load=False,
                                    requested_step=None,
                                    available_tail=policy_receiver.available_steps()[
                                        -20:
                                    ],
                                )
                            _maybe_run_evals(
                                config,
                                orchestrator,
                                policy_step=loaded_policy_step,
                                rollout_step=optimizer_step,
                                last_eval_steps=last_eval_steps,
                            )
                        else:
                            wait_policy_seconds = 0.0
                            load_policy_seconds = 0.0
                else:
                    wait_policy_seconds = 0.0
                    load_policy_seconds = 0.0
                    _wake_for_colocated_sleep(config, inference_engine)
                next_queue_step_to_submit += 1
                submit_chunk(pool, queue_step)
                if state is not None:
                    state.update_rollouts(
                        next_queue_step_to_submit=next_queue_step_to_submit,
                    )
                    state.mark_submitted(
                        queue_step=queue_step,
                        optimizer_step=optimizer_step,
                        chunk_index=queue_step % chunks_per_step,
                        pending_count=len(pending),
                    )
                submitted = True
                emit_perf(
                    "inference_native_submit",
                    queue_step=queue_step,
                    optimizer_step=optimizer_step,
                    chunk_index=queue_step % chunks_per_step,
                    wait_policy=wait_policy_seconds,
                    load_policy=load_policy_seconds,
                    submit=perf_counter() - step_started_at,
                )

            if submitted or finish_policy_load():
                continue
            if pending:
                if next_queue_step_to_submit < target_chunks:
                    start_policy_load_if_available(
                        next_queue_step_to_submit // chunks_per_step
                    )
                done, _ = wait(
                    pending,
                    timeout=config.transport.poll_interval_seconds,
                    return_when=FIRST_COMPLETED,
                )
                if done:
                    collect_done(done)
                    publish_ready()
            elif pending_policy_load is not None:
                finish_policy_load(block=True)

    if config.eval is not None and config.eval.final_eval:
        final_policy_step = _final_eval_policy_step(config, target_step)
        if final_policy_step is None:
            return 0
        if loaded_policy_step is None or loaded_policy_step < final_policy_step:
            policy = policy_receiver.wait_for_step(final_policy_step)
            _wake_for_colocated_sleep(
                config,
                inference_engine,
                tags=["weights"],
            )
            inference_engine.load_policy(policy.step_dir, step=policy.step)
            _wake_for_colocated_sleep(
                config,
                inference_engine,
                tags=["kv_cache"],
            )
            loaded_policy_step = policy.step
        else:
            _wake_for_colocated_sleep(config, inference_engine)
        _run_evals(
            config,
            orchestrator,
            policy_step=loaded_policy_step,
            rollout_step=target_step,
            envs=config.eval.env,
        )
        _sleep_for_colocated_sleep(config, inference_engine)
    if state is not None:
        state.set_status("completed", phase="completed")
    return 0


async def _run_rolling_verifier_inference(
    *,
    config: RLConfig,
    orchestrator: RLOrchestrator,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    target_step: int,
    state: OrchestratorRunState | None = None,
) -> int:
    from wavelet.orchestrator.verifiers import VerifierRolloutScheduler

    scheduler = VerifierRolloutScheduler(orchestrator)
    rollout_sender = FileSystemRolloutSender(config.output_dir, config.transport)
    loaded_policy_step: int | None = None
    pending_policy_update: asyncio.Task[int] | None = None
    last_eval_steps = (
        {env.resolved_name: -1 for env in config.eval.env}
        if config.eval is not None
        else {}
    )
    chunk_groups = _rollout_chunk_examples(config)
    chunks_per_step = _chunks_per_step(config)
    target_chunks = target_step * chunks_per_step
    try:
        for queue_step in range(target_chunks):
            step_started_at = perf_counter()
            optimizer_step = queue_step // chunks_per_step
            if pending_policy_update is not None and pending_policy_update.done():
                loaded_policy_step = pending_policy_update.result()
                pending_policy_update = None
                if state is not None:
                    state.update_policy(
                        loaded_step=loaded_policy_step,
                        pending_load=False,
                        requested_step=None,
                        available_tail=policy_receiver.available_steps()[-20:],
                    )
                await _maybe_run_evals_async(
                    config,
                    orchestrator,
                    policy_step=loaded_policy_step,
                    rollout_step=optimizer_step,
                    last_eval_steps=last_eval_steps,
                )
                emit_perf(
                    "policy_load",
                    step=loaded_policy_step,
                    wait_policy=0.0,
                    load_policy="async",
                )
            policy_step = _policy_step_to_load(
                config,
                policy_receiver,
                rollout_step=optimizer_step,
                loaded_policy_step=loaded_policy_step,
            )
            if policy_step is not None and pending_policy_update is None:
                wait_started_at = perf_counter()
                required_policy_step = _required_policy_step(config, optimizer_step)
                must_load_before_rollout = (
                    loaded_policy_step is None
                    or loaded_policy_step < required_policy_step
                )
                if must_load_before_rollout:
                    loaded_policy_step = await _load_policy_async(
                        config,
                        inference_engine,
                        policy_receiver,
                        policy_step,
                    )
                    scheduler.set_policy_step(
                        loaded_policy_step,
                        model_name=_current_policy_model_name(inference_engine),
                    )
                    if state is not None:
                        state.update_policy(
                            loaded_step=loaded_policy_step,
                            pending_load=False,
                            requested_step=None,
                            available_tail=policy_receiver.available_steps()[-20:],
                        )
                    wait_policy_seconds = perf_counter() - wait_started_at
                    load_policy_seconds = wait_policy_seconds
                    await _maybe_run_evals_async(
                        config,
                        orchestrator,
                        policy_step=loaded_policy_step,
                        rollout_step=optimizer_step,
                        last_eval_steps=last_eval_steps,
                    )
                else:
                    pending_policy_update = asyncio.create_task(
                        _load_policy_and_update_scheduler(
                            config,
                            inference_engine,
                            policy_receiver,
                            policy_step,
                            scheduler,
                        )
                    )
                    if state is not None:
                        state.update_policy(
                            pending_load=True,
                            requested_step=policy_step,
                            available_tail=policy_receiver.available_steps()[-20:],
                        )
                    wait_policy_seconds = 0.0
                    load_policy_seconds = 0.0
            elif policy_step is not None and pending_policy_update is not None:
                required_policy_step = _required_policy_step(config, optimizer_step)
                must_load_before_rollout = (
                    loaded_policy_step is None
                    or loaded_policy_step < required_policy_step
                )
                if must_load_before_rollout:
                    wait_started_at = perf_counter()
                    loaded_policy_step = await pending_policy_update
                    pending_policy_update = None
                    if state is not None:
                        state.update_policy(
                            loaded_step=loaded_policy_step,
                            pending_load=False,
                            requested_step=None,
                            available_tail=policy_receiver.available_steps()[-20:],
                        )
                    await _maybe_run_evals_async(
                        config,
                        orchestrator,
                        policy_step=loaded_policy_step,
                        rollout_step=optimizer_step,
                        last_eval_steps=last_eval_steps,
                    )
                    wait_policy_seconds = perf_counter() - wait_started_at
                    load_policy_seconds = wait_policy_seconds
                    emit_perf(
                        "policy_load",
                        step=loaded_policy_step,
                        wait_policy=wait_policy_seconds,
                        load_policy="async",
                    )
                    if loaded_policy_step < required_policy_step:
                        wait_started_at = perf_counter()
                        loaded_policy_step = await _load_policy_async(
                            config,
                            inference_engine,
                            policy_receiver,
                            policy_step,
                        )
                        scheduler.set_policy_step(
                            loaded_policy_step,
                            model_name=_current_policy_model_name(inference_engine),
                        )
                        if state is not None:
                            state.update_policy(
                                loaded_step=loaded_policy_step,
                                pending_load=False,
                                requested_step=None,
                                available_tail=policy_receiver.available_steps()[-20:],
                            )
                        wait_policy_seconds += perf_counter() - wait_started_at
                        load_policy_seconds = wait_policy_seconds
                        await _maybe_run_evals_async(
                            config,
                            orchestrator,
                            policy_step=loaded_policy_step,
                            rollout_step=optimizer_step,
                            last_eval_steps=last_eval_steps,
                        )
                else:
                    wait_policy_seconds = 0.0
                    load_policy_seconds = 0.0
            else:
                wait_policy_seconds = 0.0
                load_policy_seconds = 0.0
                _wake_for_colocated_sleep(config, inference_engine)

            generate_started_at = perf_counter()
            records = await scheduler.generate_batch(target_groups=chunk_groups)
            generate_seconds = perf_counter() - generate_started_at
            materialize_started_at = perf_counter()
            materialized_path = _write_materialized_records(
                orchestrator,
                records,
                step=queue_step,
            )
            materialize_seconds = perf_counter() - materialize_started_at
            publish_started_at = perf_counter()
            batch = rollout_sender.publish(
                materialized_path,
                step=queue_step,
                optimizer_step=optimizer_step,
                chunk_index=queue_step % chunks_per_step,
                policy_step=loaded_policy_step,
                rows=_count_nonempty_lines(materialized_path),
            )
            publish_seconds = perf_counter() - publish_started_at
            log_rollout_metrics(
                config,
                materialized_path,
                step=optimizer_step,
                policy_step=loaded_policy_step,
                queue_step=queue_step,
                optimizer_step=optimizer_step,
                chunk_index=queue_step % chunks_per_step,
                timings={
                    "generate_completions": generate_seconds,
                    "parallel_preprocess": materialize_seconds,
                    "publish": publish_seconds,
                    "step": perf_counter() - step_started_at,
                },
            )
            if state is not None:
                state.update_rollouts(next_queue_step_to_submit=queue_step + 1)
                state.mark_submitted(
                    queue_step=queue_step,
                    optimizer_step=optimizer_step,
                    chunk_index=queue_step % chunks_per_step,
                    pending_count=0,
                )
                state.mark_completed(
                    queue_step=queue_step,
                    optimizer_step=optimizer_step,
                    chunk_index=queue_step % chunks_per_step,
                    pending_count=0,
                    completed_count=1,
                )
                state.mark_published(
                    queue_step=queue_step,
                    optimizer_step=optimizer_step,
                    chunk_index=queue_step % chunks_per_step,
                    path=str(batch.path),
                    next_queue_step_to_publish=queue_step + 1,
                    completed_count=0,
                )
            emit_perf(
                "inference_chunk",
                queue_step=queue_step,
                optimizer_step=optimizer_step,
                groups=chunk_groups,
                wait_policy=wait_policy_seconds,
                load_policy=load_policy_seconds,
                generate=generate_seconds,
                materialize=materialize_seconds,
                publish=publish_seconds,
                pending_policy_update=int(pending_policy_update is not None),
                total=perf_counter() - step_started_at,
            )
            print(batch.path)
        if pending_policy_update is not None:
            loaded_policy_step = await pending_policy_update
            pending_policy_update = None
            if state is not None:
                state.update_policy(
                    loaded_step=loaded_policy_step,
                    pending_load=False,
                    requested_step=None,
                    available_tail=policy_receiver.available_steps()[-20:],
                )
        if config.eval is not None and config.eval.final_eval:
            final_policy_step = _final_eval_policy_step(config, target_step)
            if final_policy_step is not None:
                if loaded_policy_step is None or loaded_policy_step < final_policy_step:
                    policy = await asyncio.to_thread(
                        policy_receiver.wait_for_step,
                        final_policy_step,
                    )
                    _wake_for_colocated_sleep(
                        config,
                        inference_engine,
                        tags=["weights"],
                    )
                    inference_engine.load_policy(policy.step_dir, step=policy.step)
                    _wake_for_colocated_sleep(
                        config,
                        inference_engine,
                        tags=["kv_cache"],
                    )
                    loaded_policy_step = policy.step
                    scheduler.set_policy_step(
                        loaded_policy_step,
                        model_name=_current_policy_model_name(inference_engine),
                    )
                else:
                    _wake_for_colocated_sleep(config, inference_engine)
                await _run_evals_async(
                    config,
                    orchestrator,
                    policy_step=loaded_policy_step,
                    rollout_step=target_step,
                    envs=config.eval.env,
                )
                _sleep_for_colocated_sleep(config, inference_engine)
        if state is not None:
            state.set_status("completed", phase="completed")
        return 0
    finally:
        if pending_policy_update is not None:
            pending_policy_update.cancel()
            await asyncio.gather(pending_policy_update, return_exceptions=True)
        await scheduler.aclose()


async def _load_policy_async(
    config: RLConfig,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    policy_step: int,
) -> int:
    step, _, _ = await asyncio.to_thread(
        _load_policy_step,
        config,
        inference_engine,
        policy_receiver,
        policy_step,
    )
    return step


async def _load_policy_and_update_scheduler(
    config: RLConfig,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    policy_step: int,
    scheduler,
) -> int:
    loaded_step = await _load_policy_async(
        config,
        inference_engine,
        policy_receiver,
        policy_step,
    )
    scheduler.set_policy_step(
        loaded_step,
        model_name=_current_policy_model_name(inference_engine),
    )
    await scheduler.mark_policy_update()
    return loaded_step


def _current_policy_model_name(inference_engine) -> str | None:
    model_name = getattr(inference_engine, "policy_model_name", None)
    return model_name if isinstance(model_name, str) and model_name else None


def _load_policy_step(
    config: RLConfig,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    policy_step: int,
) -> tuple[int, float, float]:
    wait_started_at = perf_counter()
    policy = policy_receiver.wait_for_step(policy_step)
    wait_policy_seconds = perf_counter() - wait_started_at
    load_started_at = perf_counter()
    _wake_for_colocated_sleep(config, inference_engine, tags=["weights"])
    inference_engine.load_policy(policy.step_dir, step=policy.step)
    _wake_for_colocated_sleep(config, inference_engine, tags=["kv_cache"])
    load_policy_seconds = perf_counter() - load_started_at
    append_event_best_effort(
        config.output_dir / "events",
        QueueEvent(
            time=utc_now(),
            kind="policy_load_completed",
            policy_step=policy.step,
            details={
                "load_seconds": load_policy_seconds,
                "wait_seconds": wait_policy_seconds,
            },
        ),
    )
    return policy.step, wait_policy_seconds, load_policy_seconds


def _write_materialized_records(
    orchestrator: RLOrchestrator,
    records: list[RLExample],
    *,
    step: int,
) -> Path:
    if not records:
        raise RuntimeError(
            "Rolling verifier scheduler produced no trainable rollout records."
        )
    output_path = orchestrator._resolve_output_path(step=step)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not orchestrator.config.orchestrator.overwrite:
        raise FileExistsError(
            f"Rollout file '{output_path}' already exists and overwrite is disabled."
        )
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.temperatures is None:
                raise ValueError("Rollout record is missing temperatures.")
            handle.write(json.dumps(orchestrator._serialize_record(record)) + "\n")
    return output_path


def _publish_step(
    orchestrator: RLOrchestrator,
    rollout_sender: FileSystemRolloutSender,
    step: int,
    inference_engine,
    policy_step: int | None,
):
    return _time_materialize_and_publish(
        orchestrator,
        lambda: orchestrator.materialize(
            step=step,
            inference_engine=inference_engine,
        ),
        rollout_sender,
        step=step,
        optimizer_step=step,
        policy_step=policy_step,
    )


def _publish_native_chunk(
    orchestrator: RLOrchestrator,
    rollout_sender: FileSystemRolloutSender,
    optimizer_step: int,
    chunk_index: int,
    queue_step: int,
    chunk_examples: int,
    inference_engine,
    policy_step: int | None,
):
    return _time_materialize_and_publish(
        orchestrator,
        lambda: orchestrator.materialize_native_chunk(
            optimizer_step=optimizer_step,
            chunk_index=chunk_index,
            queue_step=queue_step,
            chunk_examples=chunk_examples,
            inference_engine=inference_engine,
        ),
        rollout_sender,
        step=queue_step,
        optimizer_step=optimizer_step,
        chunk_index=chunk_index,
        policy_step=policy_step,
    )


def _time_materialize_and_publish(
    orchestrator: RLOrchestrator,
    materialize,
    rollout_sender: FileSystemRolloutSender,
    *,
    step: int,
    optimizer_step: int | None = None,
    chunk_index: int | None = None,
    policy_step: int | None = None,
):
    materialize_started_at = perf_counter()
    materialized_path = materialize()
    materialize_seconds = perf_counter() - materialize_started_at
    publish_started_at = perf_counter()
    batch = rollout_sender.publish(
        materialized_path,
        step=step,
        optimizer_step=optimizer_step,
        chunk_index=chunk_index,
        policy_step=policy_step,
        rows=_count_nonempty_lines(materialized_path),
    )
    publish_seconds = perf_counter() - publish_started_at
    log_step = optimizer_step if optimizer_step is not None else step
    log_rollout_metrics(
        orchestrator.config,
        materialized_path,
        step=log_step,
        policy_step=policy_step,
        queue_step=step,
        optimizer_step=optimizer_step,
        chunk_index=chunk_index,
        timings={
            "generate_completions": materialize_seconds,
            "parallel_preprocess": 0.0,
            "publish": publish_seconds,
            "step": materialize_seconds + publish_seconds,
        },
    )
    return step, batch, materialize_seconds, publish_seconds


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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
    envs = select_due_eval_envs(
        config,
        policy_step=policy_step,
        last_eval_steps=last_eval_steps,
    )
    _run_evals(
        config,
        orchestrator,
        policy_step=policy_step,
        rollout_step=rollout_step,
        envs=envs,
    )


async def _maybe_run_evals_async(
    config: RLConfig,
    orchestrator: RLOrchestrator,
    *,
    policy_step: int,
    rollout_step: int,
    last_eval_steps: dict[str, int],
) -> None:
    envs = select_due_eval_envs(
        config,
        policy_step=policy_step,
        last_eval_steps=last_eval_steps,
    )
    await _run_evals_async(
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
    _validate_eval_supported(config)

    from wavelet.orchestrator.verifiers import evaluate_env

    for env in envs:
        metrics = evaluate_env(
            orchestrator,
            env,
            step=rollout_step,
            policy_step=policy_step,
        )
        print(json_dumps_compact(metrics), flush=True)


async def _run_evals_async(
    config: RLConfig,
    orchestrator: RLOrchestrator,
    *,
    policy_step: int,
    rollout_step: int,
    envs,
) -> None:
    if not envs:
        return
    _validate_eval_supported(config)

    from wavelet.orchestrator.verifiers import evaluate_env_async

    for env in envs:
        metrics = await evaluate_env_async(
            orchestrator,
            env,
            step=rollout_step,
            policy_step=policy_step,
        )
        print(json_dumps_compact(metrics), flush=True)


def _validate_eval_supported(config: RLConfig) -> None:
    if (
        config.orchestrator.custom_rollout_function
        != "wavelet.orchestrator.verifiers:generate_rollouts"
    ):
        raise ValueError("RL eval is currently supported for verifier rollouts only.")


def _sleep_for_colocated_sleep(config: RLConfig, inference_engine) -> None:
    if config.launcher.mode == "colocate_sleep":
        inference_engine.sleep()


def _wake_for_colocated_sleep(
    config: RLConfig,
    inference_engine,
    *,
    tags: list[str] | None = None,
) -> None:
    if config.launcher.mode == "colocate_sleep":
        if tags is None or "weights" in tags:
            _wait_for_colocated_training_memory(config)
        inference_engine.wake(tags=tags)


def _wait_for_colocated_training_memory(config: RLConfig) -> None:
    timeout = config.launcher.colocate_memory_wait_timeout_seconds
    if timeout <= 0:
        return
    devices = _colocated_trainer_device_ids(config)
    if not devices:
        return

    required_free_fraction = max(
        config.inference.vllm.gpu_memory_utilization
        - config.launcher.colocate_memory_wait_margin,
        0.0,
    )
    deadline = perf_counter() + timeout
    last_stats: dict[str, tuple[int, int]] = {}
    while True:
        stats = _query_gpu_memory_mib(devices)
        if not stats:
            return
        last_stats = stats
        if all(
            (total - used) >= int(total * required_free_fraction)
            for total, used in stats.values()
        ):
            return
        if perf_counter() >= deadline:
            break
        sleep(config.launcher.colocate_memory_wait_poll_seconds)

    details = ", ".join(
        f"gpu{idx}: used={used}MiB free={total - used}MiB total={total}MiB"
        for idx, (total, used) in sorted(last_stats.items(), key=lambda item: item[0])
    )
    raise RuntimeError(
        "Timed out waiting for colocated trainer GPU memory to be released before "
        f"waking vLLM. Required free fraction is {required_free_fraction:.2f}; "
        f"last observed: {details}"
    )


def _colocated_trainer_device_ids(config: RLConfig) -> set[str]:
    value = (
        config.launcher.trainer_cuda_visible_devices
        or config.launcher.inference_cuda_visible_devices
    )
    if value is None:
        return set()
    if isinstance(value, str):
        raw_parts = value.replace(";", ",").replace("|", ",").split(",")
    else:
        raw_parts = []
        for item in value:
            raw_parts.extend(item.split(","))
    return {part.strip() for part in raw_parts if part.strip().isdigit()}


def _query_gpu_memory_mib(devices: set[str]) -> dict[str, tuple[int, int]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    stats: dict[str, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or parts[0] not in devices:
            continue
        try:
            stats[parts[0]] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    return stats


def json_dumps_compact(payload) -> str:
    import json

    return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())

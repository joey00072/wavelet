# ruff: noqa: E402, F811

from dataclasses import dataclass


from enum import StrEnum


from typing import Callable


from wavelet.configs.rl_config import RLConfig


from wavelet.orchestrator.schedule import rollout_chunk_examples


from wavelet.orchestrator.sources import RolloutSourceKind, source_kind
from wavelet.orchestrator.envs import (
    _ensure_verifier_openai_patches as _ensure_verifier_openai_patches,  # noqa: F401
    _load_verifiers as _load_verifiers,  # noqa: F401
    evaluate_env as evaluate_env,  # noqa: F401
    evaluate_env_async as evaluate_env_async,  # noqa: F401
    _evaluate_env_async as _evaluate_env_async,  # noqa: F401
    _run_eval_examples as _run_eval_examples,  # noqa: F401
    _eval_metrics as _eval_metrics,  # noqa: F401
    _completion_len as _completion_len,  # noqa: F401
    _write_eval_rollouts as _write_eval_rollouts,  # noqa: F401
    _append_eval_metrics as _append_eval_metrics,  # noqa: F401
    _verifier_client_routes as _verifier_client_routes,  # noqa: F401
    _verifier_clients as _verifier_clients,  # noqa: F401
    _verifier_base_urls as _verifier_base_urls,  # noqa: F401
    _verifier_model as _verifier_model,  # noqa: F401
    _verifier_extra_env_kwargs as _verifier_extra_env_kwargs,  # noqa: F401
    _load_cached_env as _load_cached_env,  # noqa: F401
    _run_all as _run_all,  # noqa: F401
    _run_complete_record_set as _run_complete_record_set,  # noqa: F401
    _run_until_target_groups as _run_until_target_groups,  # noqa: F401
    _run_group as _run_group,  # noqa: F401
    _env_name as _env_name,  # noqa: F401
    _stamp_env_name as _stamp_env_name,  # noqa: F401
    _run_single_rollout as _run_single_rollout,  # noqa: F401
    _successful_rollout_outputs as _successful_rollout_outputs,  # noqa: F401
    _assign_completed_group_advantages as _assign_completed_group_advantages,  # noqa: F401
    _completed_group_outputs as _completed_group_outputs,  # noqa: F401
    _raise_if_external_rate_limit as _raise_if_external_rate_limit,  # noqa: F401
    _truncate_error as _truncate_error,  # noqa: F401
    _assign_group_advantages as _assign_group_advantages,  # noqa: F401
    _algorithm_record_from_output as _algorithm_record_from_output,  # noqa: F401
    _has_trainable_advantage as _has_trainable_advantage,  # noqa: F401
    _is_usable_training_group as _is_usable_training_group,  # noqa: F401
    _is_complete_training_group as _is_complete_training_group,  # noqa: F401
    _has_trainable_rollout_record as _has_trainable_rollout_record,  # noqa: F401
    _mark_zero_advantage_records_metric_only as _mark_zero_advantage_records_metric_only,  # noqa: E501
    _has_trainable_trajectory as _has_trainable_trajectory,  # noqa: F401
    _patch_env_response_messages as _patch_env_response_messages,  # noqa: F401
    _coerce_vf_message as _coerce_vf_message,  # noqa: F401
    _verifier_example as _verifier_example,  # noqa: F401
    _sampling_args as _sampling_args,  # noqa: F401
    _sampling_args_with_cache_salt as _sampling_args_with_cache_salt,  # noqa: F401
    _assign_rollout_advantages as _assign_rollout_advantages,  # noqa: F401
    _records_from_output as _records_from_output,  # noqa: F401
    _output_group_key as _output_group_key,  # noqa: F401
    _interleave_output as _interleave_output,  # noqa: F401
    _step_token_segment as _step_token_segment,  # noqa: F401
    _messages as _messages,  # noqa: F401
    _mask_prompt_history as _mask_prompt_history,  # noqa: F401
)


class PublishMode(StrEnum):
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass(frozen=True)
class RolloutSchedule:
    """Explicit scheduler parameters derived from legacy-compatible config."""

    source: RolloutSourceKind
    max_async_level: int
    chunk_examples: int
    publish_mode: PublishMode
    max_pending_chunks: int | None

    @property
    def is_sync(self) -> bool:
        return self.max_async_level == 0


@dataclass(slots=True)
class IntegratedRolloutScheduler:
    """Shared sync stepping core for the in-process trainer/rollout runtime."""

    target_step: int
    current_step: Callable[[], int]
    prepare_policy: Callable[[int], None]
    publish: Callable[[int], None]
    consume_and_train: Callable[[], None]
    after_step: Callable[[], None] = lambda: None

    def run(self) -> None:
        while self.current_step() < self.target_step:
            step = self.current_step()
            self.prepare_policy(step)
            self.publish(step)
            self.consume_and_train()
            self.after_step()


def resolve_rollout_schedule(config: RLConfig) -> RolloutSchedule:
    source = source_kind(config.orchestrator.custom_rollout_function)
    streaming = (
        config.launcher.mode == "process"
        and config.orchestrator.max_async_level > 0
        and source in {RolloutSourceKind.NATIVE, RolloutSourceKind.VERIFIER}
        and (
            source is RolloutSourceKind.VERIFIER
            or config.orchestrator.examples_per_step is not None
        )
    )
    return RolloutSchedule(
        source=source,
        max_async_level=config.orchestrator.max_async_level,
        chunk_examples=rollout_chunk_examples(config),
        publish_mode=PublishMode.STREAMING if streaming else PublishMode.BATCH,
        max_pending_chunks=config.orchestrator.max_pending_rollout_chunks,
    )


import asyncio


import json


import math


import random


from dataclasses import dataclass, field


from pathlib import Path


from time import perf_counter


from typing import Any


from wavelet.data.rl import RLExample, load_rl_records


from wavelet.orchestrator.algorithms import (
    algorithm_epsilon,
)


from wavelet.orchestrator.rollouts import RLOrchestrator


from wavelet.orchestrator.schedule import (
    rollout_chunk_examples as _rollout_chunk_examples,
)


from wavelet.utils.monitoring import emit_perf


from wavelet.utils.pathing import resolve_resume_checkpoint


def _resume_optimizer_step(config: RLConfig) -> int:
    checkpoint = config.ckpt
    if checkpoint is None or checkpoint.resume_step is None:
        return 0
    checkpoint_dir = resolve_resume_checkpoint(
        config.checkpoint_output_dir,
        checkpoint.resume_step,
    )
    try:
        step = int(checkpoint_dir.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise ValueError(
            f"Could not resolve optimizer step from checkpoint '{checkpoint_dir}'."
        ) from exc
    if config.max_steps is not None and step > config.max_steps:
        raise ValueError(
            f"Checkpoint step {step} exceeds configured max_steps={config.max_steps}."
        )
    return step


from wavelet.orchestrator.envs import (
    _load_verifiers,
    _verifier_clients,
    _verifier_base_urls,
    _verifier_model,
    _verifier_extra_env_kwargs,
    _load_cached_env,
    _run_all,
    _run_group,
    _env_name,
    _stamp_env_name,
    _run_single_rollout,
    _assign_completed_group_advantages,
    _completed_group_outputs,
    _has_trainable_rollout_record,
    _mark_zero_advantage_records_metric_only,
    _verifier_example,
    _sampling_args,
    _records_from_output,
    _scale_verifier_executors,
    _teardown_cached_verifier_envs,
)


@dataclass(slots=True)
class _PendingVerifierRequest:
    group_id: int
    client_index: int
    rollout_count: int
    off_policy_steps: int = 0
    policy_step: int | None = None


@dataclass(slots=True)
class _VerifierGroupState:
    example: dict[str, Any]
    rollouts_to_schedule: int
    policy_step: int | None = None
    completed_outputs: list[dict[str, Any]] = field(default_factory=list)
    pinned_client_index: int | None = None


@dataclass(slots=True)
class _VerifierBatchStats:
    """Metrics for every verifier group completed while building one batch."""

    admitted_groups: int = 0
    rejected_groups: int = 0
    rollout_rewards: list[float] = field(default_factory=list)
    group_reward_sums: list[float] = field(default_factory=list)

    def observe(self, outputs: list[dict[str, Any]], *, admitted: bool) -> None:
        rewards = [float(output["reward"]) for output in outputs]
        self.rollout_rewards.extend(rewards)
        self.group_reward_sums.append(sum(rewards))
        if admitted:
            self.admitted_groups += 1
        else:
            self.rejected_groups += 1

    def metrics(self, *, rollouts_per_group: int) -> dict[str, float]:
        completed = len(self.group_reward_sums)
        metrics = {
            "generation/groups/completed": float(completed),
            "generation/groups/admitted": float(self.admitted_groups),
            "generation/groups/rejected": float(self.rejected_groups),
            "generation/rollouts/scored": float(len(self.rollout_rewards)),
        }
        if completed == 0:
            return metrics

        solve_none = sum(value == 0.0 for value in self.group_reward_sums)
        solve_all = sum(
            value >= rollouts_per_group for value in self.group_reward_sums
        )
        metrics.update(
            {
                "generation/groups/admission_rate": (
                    self.admitted_groups / completed
                ),
                "generation/reward/mean": (
                    sum(self.rollout_rewards) / len(self.rollout_rewards)
                    if self.rollout_rewards
                    else 0.0
                ),
                "generation/solve_none/rate": solve_none / completed,
                "generation/solve_all/rate": solve_all / completed,
                "generation/effective_groups/rate": (
                    completed - solve_none - solve_all
                )
                / completed,
            }
        )
        return metrics


def generate_rollouts(
    orchestrator: RLOrchestrator,
    records: list[RLExample],
    _inference_engine,
) -> list[RLExample]:
    vf = _load_verifiers("rollouts")

    config = orchestrator.config
    env_id = config.orchestrator.verifier_env_id
    if env_id is None:
        raise ValueError("orchestrator.verifier_env_id is required.")
    base_urls = _verifier_base_urls(config)
    model = _verifier_model(config, _inference_engine)
    env_started_at = perf_counter()
    env, env_cache_hit = _load_cached_env(
        vf,
        env_id,
        config.orchestrator.verifier_env_args,
        _verifier_extra_env_kwargs(config),
    )
    env_load_seconds = perf_counter() - env_started_at
    clients = _verifier_clients(vf, config, base_urls=base_urls)
    policy_step = getattr(_inference_engine, "policy_step", None)
    sampling_args = _sampling_args(
        config,
        cache_salt=None if policy_step is None else str(policy_step),
    )
    rollout_count = config.orchestrator.rollouts_per_example or 1

    rollout_started_at = perf_counter()
    outputs = asyncio.run(
        _run_all(
            vf,
            env,
            records,
            clients=clients,
            model=model,
            sampling_args=sampling_args,
            rollout_count=rollout_count,
            max_retries=config.orchestrator.verifier_max_retries,
            target_groups=config.orchestrator.examples_per_step,
            filter_zero_advantage=config.orchestrator.filter_zero_advantage,
            advantage_epsilon=algorithm_epsilon(config.algo),
            algorithm_config=config.algo,
            env_name=_env_name(env, fallback=env_id),
        )
    )
    if isinstance(policy_step, int) and not isinstance(policy_step, bool):
        for output in outputs:
            output["_wavelet_policy_step"] = policy_step
    rollout_seconds = perf_counter() - rollout_started_at
    convert_started_at = perf_counter()
    records = [record for output in outputs for record in _records_from_output(output)]
    convert_seconds = perf_counter() - convert_started_at
    emit_perf(
        "verifier_rollouts",
        env_load=env_load_seconds,
        env_cache_hit=int(env_cache_hit),
        rollout=rollout_seconds,
        convert=convert_seconds,
        outputs=len(outputs),
        records=len(records),
    )
    return records


class VerifierRolloutScheduler:
    """Persistent verifier rollout scheduler for process-mode async RL."""

    def __init__(
        self,
        orchestrator: RLOrchestrator,
        *,
        start_record_cursor: int = 0,
    ) -> None:
        vf = _load_verifiers("rollouts")

        config = orchestrator.config
        env_id = config.orchestrator.verifier_env_id
        if env_id is None:
            raise ValueError("orchestrator.verifier_env_id is required.")

        self.vf = vf
        self.orchestrator = orchestrator
        self.config = config
        self.env, _ = _load_cached_env(
            vf,
            env_id,
            config.orchestrator.verifier_env_args,
            _verifier_extra_env_kwargs(config),
        )
        self.env_name = _env_name(self.env, fallback=env_id)
        self.model = _verifier_model(config)
        self.policy_step: int | None = None
        self.rollout_count = config.orchestrator.rollouts_per_example or 1
        self.target_groups = config.orchestrator.examples_per_step
        if self.target_groups is None:
            raise ValueError(
                "orchestrator.examples_per_step is required for rolling verifier "
                "scheduling."
            )
        self.clients = _verifier_clients(vf, config)
        self.records = load_rl_records(config.data)
        if not self.records:
            raise ValueError("Verifier scheduler requires at least one train record.")
        self.record_cursor = start_record_cursor
        self._record_order_epoch: int | None = None
        self._record_order: list[int] = []
        self.next_group_id = 0
        self.groups: dict[int, _VerifierGroupState] = {}
        self.pending: dict[
            asyncio.Task[list[dict[str, Any]]],
            _PendingVerifierRequest,
        ] = {}
        self.pending_clients: dict[asyncio.Task[list[dict[str, Any]]], int] = {}
        self.ready_groups: list[list[dict[str, Any]]] = []
        self.ready_group_off_policy_steps: list[int] = []
        self.requires_group_scoring = bool(
            getattr(self.env, "requires_group_scoring", False)
        )
        self.cancelled_rollouts_count = 0
        self.rejected_groups_count = 0
        self.last_batch_metrics: dict[str, float] = {}
        self._policy_update_ready = asyncio.Event()
        self._policy_update_ready.set()
        self.policy_update_wait_seconds = 0.0

    def set_policy_step(
        self,
        policy_step: int | None,
        *,
        model_name: str | None = None,
    ) -> None:
        self.policy_step = policy_step
        if model_name is not None:
            self.model = model_name

    @property
    def max_inflight_groups(self) -> int:
        explicit_rollouts = self.config.orchestrator.max_inflight_rollouts
        if explicit_rollouts is not None:
            return max(1, explicit_rollouts // self.rollout_count)
        base_groups = self.target_groups
        pending_chunk_limit = self.config.orchestrator.max_pending_rollout_chunks
        if pending_chunk_limit is not None:
            bounded_groups = (
                _rollout_chunk_examples(self.config) * pending_chunk_limit
            )
            return bounded_groups
        oversampled_groups = math.ceil(
            base_groups * self.config.orchestrator.oversampling_factor
        )
        return max(
            len(self.clients),
            oversampled_groups,
        )

    @property
    def max_inflight_rollouts(self) -> int:
        explicit_rollouts = self.config.orchestrator.max_inflight_rollouts
        if explicit_rollouts is not None:
            return explicit_rollouts
        return self.max_inflight_groups * self.rollout_count

    @property
    def inflight_rollout_count(self) -> int:
        return sum(info.rollout_count for info in self.pending.values())

    async def generate_batch(
        self, *, target_groups: int | None = None
    ) -> list[RLExample]:
        started_at = perf_counter()
        self.executor_concurrency = _scale_verifier_executors(
            self.max_inflight_rollouts
        )
        target_groups = self.target_groups if target_groups is None else target_groups
        outputs: list[dict[str, Any]] = []
        accepted_groups = 0
        rejected_groups = 0
        completed_groups = 0
        attempt = 0
        batch_stats = _VerifierBatchStats()
        max_completed_groups = target_groups * (
            self.config.orchestrator.zero_advantage_max_retries + 1
        )
        self._sync_ready_group_ages()
        self.policy_update_wait_seconds = 0.0

        try:
            drained_completed, drained_rejected = self._drain_completed_groups_to_ready(
                target_groups=target_groups,
                outputs=outputs,
                accepted_groups=accepted_groups,
                batch_stats=batch_stats,
            )
            completed_groups += drained_completed
            rejected_groups += drained_rejected
            accepted_groups += len(outputs) // self.rollout_count
            while self.ready_groups and accepted_groups < target_groups:
                outputs.extend(self.ready_groups.pop(0))
                if self.ready_group_off_policy_steps:
                    self.ready_group_off_policy_steps.pop(0)
                accepted_groups += 1

            while True:
                if accepted_groups >= target_groups:
                    if self._has_trainable_batch(outputs):
                        break
                    attempt += 1
                    self._raise_if_retries_exhausted(
                        completed_groups=completed_groups,
                        max_completed_groups=max_completed_groups,
                        accepted_groups=accepted_groups,
                        target_groups=target_groups,
                        rejected_groups=rejected_groups,
                    )
                    outputs = []
                    accepted_groups = 0

                await self._wait_for_policy_update()
                self._fill_inflight()
                done, _ = await asyncio.wait(
                    self.pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    accepted, completed, rejected = self._consume_completed_task(
                        task,
                        target_groups=target_groups,
                        outputs=outputs,
                        accepted_groups=accepted_groups,
                        batch_stats=batch_stats,
                    )
                    accepted_groups += accepted
                    completed_groups += completed
                    rejected_groups += rejected
                    self._raise_if_retries_exhausted(
                        completed_groups=completed_groups,
                        max_completed_groups=max_completed_groups,
                        accepted_groups=accepted_groups,
                        target_groups=target_groups,
                        rejected_groups=rejected_groups,
                    )
        except Exception:
            await self.aclose()
            raise

        drained_completed, drained_rejected = self._drain_completed_groups_to_ready(
            target_groups=target_groups,
            outputs=outputs,
            accepted_groups=accepted_groups,
            batch_stats=batch_stats,
        )
        completed_groups += drained_completed
        rejected_groups += drained_rejected
        return self._finalize_batch(
            outputs,
            started_at=started_at,
            attempts=attempt + 1,
            accepted_groups=accepted_groups,
            rejected_groups=rejected_groups,
            completed_groups=completed_groups,
            batch_stats=batch_stats,
        )

    def _has_trainable_batch(self, outputs: list[dict[str, Any]]) -> bool:
        records = [
            record for output in outputs for record in _records_from_output(output)
        ]
        records = _mark_zero_advantage_records_metric_only(records, self.config)
        return _has_trainable_rollout_record(records)

    @staticmethod
    def _raise_if_retries_exhausted(
        *,
        completed_groups: int,
        max_completed_groups: int,
        accepted_groups: int,
        target_groups: int,
        rejected_groups: int,
    ) -> None:
        if (
            accepted_groups >= target_groups
            or completed_groups < max_completed_groups
        ):
            return
        raise RuntimeError(
            "Verifier scheduler could not produce enough trainable rollout groups "
            f"after {completed_groups} completed group(s): accepted "
            f"{accepted_groups}, rejected {rejected_groups}. Increase "
            "orchestrator.zero_advantage_max_retries, relax filtering, or check "
            "reward/model behavior."
        )

    def _finalize_batch(
        self,
        outputs: list[dict[str, Any]],
        *,
        started_at: float,
        attempts: int,
        accepted_groups: int,
        rejected_groups: int,
        completed_groups: int,
        batch_stats: _VerifierBatchStats,
    ) -> list[RLExample]:
        self._fill_inflight()
        convert_started_at = perf_counter()
        records = [
            record for output in outputs for record in _records_from_output(output)
        ]
        records = _mark_zero_advantage_records_metric_only(records, self.config)
        self.last_batch_metrics = batch_stats.metrics(
            rollouts_per_group=self.rollout_count
        )
        self.last_batch_metrics["generation/executor_concurrency"] = float(
            self.executor_concurrency
        )
        record_cursor = getattr(self, "record_cursor", 0)
        self.last_batch_metrics["generation/data/cursor"] = float(record_cursor)
        self.last_batch_metrics["generation/data/epoch"] = float(
            record_cursor // len(getattr(self, "records", [None]))
        )
        self.last_batch_metrics["generation/policy_update_wait_seconds"] = float(
            getattr(self, "policy_update_wait_seconds", 0.0)
        )
        emit_perf(
            "verifier_scheduler",
            attempts=attempts,
            accepted_groups=accepted_groups,
            rejected_groups=rejected_groups,
            completed_groups=completed_groups,
            inflight_rollouts=self.inflight_rollout_count,
            records=len(records),
            convert=perf_counter() - convert_started_at,
            total=perf_counter() - started_at,
        )
        return records

    def _drain_completed_groups_to_ready(
        self,
        *,
        target_groups: int,
        outputs: list[dict[str, Any]],
        accepted_groups: int,
        batch_stats: _VerifierBatchStats,
    ) -> tuple[int, int]:
        completed_groups = 0
        rejected_groups = 0
        for task in [task for task in self.pending if task.done()]:
            accepted, completed, rejected = self._consume_completed_task(
                task,
                target_groups=target_groups,
                outputs=outputs,
                accepted_groups=accepted_groups,
                batch_stats=batch_stats,
            )
            accepted_groups += accepted
            completed_groups += completed
            rejected_groups += rejected
        self.rejected_groups_count = (
            getattr(self, "rejected_groups_count", 0) + rejected_groups
        )
        return completed_groups, rejected_groups

    def _consume_completed_task(
        self,
        task: asyncio.Task[list[dict[str, Any]]],
        *,
        target_groups: int,
        outputs: list[dict[str, Any]],
        accepted_groups: int,
        batch_stats: _VerifierBatchStats | None = None,
    ) -> tuple[int, int, int]:
        """Consume one finished request and classify its completed group."""
        request = self.pending.pop(task, None)
        self.pending_clients.pop(task, None)
        if request is None:
            return 0, 0, 0
        group = self.groups.get(request.group_id)
        if group is None:
            return 0, 0, 0

        group_outputs = _completed_group_outputs(task)
        for output in group_outputs:
            output["_wavelet_policy_step"] = request.policy_step
            output["_wavelet_group_id"] = f"persistent:{request.group_id}"
        missing_rollouts = request.rollout_count - len(group_outputs)
        if missing_rollouts > 0:
            if request.policy_step != getattr(self, "policy_step", None):
                self.groups.pop(request.group_id, None)
                return 0, 1, 1
            if self.requires_group_scoring:
                group.completed_outputs.clear()
                group.rollouts_to_schedule = self.rollout_count
            else:
                group.rollouts_to_schedule += missing_rollouts
        group.completed_outputs.extend(group_outputs)
        if len(group.completed_outputs) < self.rollout_count:
            return 0, 0, 0

        completed_outputs = group.completed_outputs
        self.groups.pop(request.group_id, None)
        _stamp_env_name(completed_outputs, getattr(self, "env_name", "verifier"))
        _assign_completed_group_advantages(completed_outputs, self.config)
        is_usable = _is_usable_training_group(
            completed_outputs,
            expected_rollouts=self.rollout_count,
            filter_zero_advantage=self.config.orchestrator.filter_zero_advantage,
            advantage_epsilon=algorithm_epsilon(self.config.algo),
        )
        if batch_stats is not None:
            batch_stats.observe(completed_outputs, admitted=is_usable)
        if not is_usable:
            return 0, 1, 1
        if accepted_groups < target_groups:
            outputs.extend(completed_outputs)
            return 1, 1, 0

        self.ready_groups.append(completed_outputs)
        self.ready_group_off_policy_steps.append(0)
        return 0, 1, 0

    async def mark_policy_update(self) -> int:
        max_off_policy_steps = self.config.orchestrator.max_off_policy_steps
        cancelled_rollouts = self._age_ready_groups(max_off_policy_steps)
        if not self.pending:
            self.cancelled_rollouts_count += cancelled_rollouts
            return cancelled_rollouts

        stale_group_ids = {
            request.group_id
            for request in self.pending.values()
            if self._request_policy_lag(request) > max_off_policy_steps
        }
        tasks_to_cancel = [
            task
            for task, request in self.pending.items()
            if request.group_id in stale_group_ids
        ]
        for task in tasks_to_cancel:
            request = self.pending.pop(task, None)
            self.pending_clients.pop(task, None)
            if request is not None:
                cancelled_rollouts += request.rollout_count
            task.cancel()
        for group_id in stale_group_ids:
            self.groups.pop(group_id, None)

        for request in self.pending.values():
            request.off_policy_steps = self._request_policy_lag(request)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        self.cancelled_rollouts_count += cancelled_rollouts
        return cancelled_rollouts

    def _request_policy_lag(self, request: _PendingVerifierRequest) -> int:
        current_policy_step = getattr(self, "policy_step", None)
        if isinstance(current_policy_step, int) and isinstance(
            request.policy_step, int
        ):
            return max(current_policy_step - request.policy_step, 0)
        return request.off_policy_steps + 1

    def _age_ready_groups(self, max_off_policy_steps: int) -> int:
        if not hasattr(self, "ready_groups"):
            self.ready_groups = []
        self._sync_ready_group_ages()
        if not self.ready_groups:
            self.ready_group_off_policy_steps.clear()
            return 0

        kept_groups: list[list[dict[str, Any]]] = []
        kept_ages: list[int] = []
        dropped_rollouts = 0
        for group_outputs, off_policy_steps in zip(
            self.ready_groups,
            self.ready_group_off_policy_steps,
            strict=False,
        ):
            policy_steps = [
                output.get("_wavelet_policy_step")
                for output in group_outputs
                if isinstance(output.get("_wavelet_policy_step"), int)
                and not isinstance(output.get("_wavelet_policy_step"), bool)
            ]
            current_policy_step = getattr(self, "policy_step", None)
            if isinstance(current_policy_step, int) and policy_steps:
                next_age = max(current_policy_step - min(policy_steps), 0)
            else:
                next_age = off_policy_steps + 1
            if next_age > max_off_policy_steps:
                dropped_rollouts += len(group_outputs)
                continue
            kept_groups.append(group_outputs)
            kept_ages.append(next_age)
        self.ready_groups = kept_groups
        self.ready_group_off_policy_steps = kept_ages
        return dropped_rollouts

    def _sync_ready_group_ages(self) -> None:
        if not hasattr(self, "ready_groups"):
            self.ready_groups = []
        if not hasattr(self, "ready_group_off_policy_steps"):
            self.ready_group_off_policy_steps = [0] * len(self.ready_groups)
            return
        if len(self.ready_group_off_policy_steps) < len(self.ready_groups):
            self.ready_group_off_policy_steps.extend(
                [0] * (len(self.ready_groups) - len(self.ready_group_off_policy_steps))
            )
        elif len(self.ready_group_off_policy_steps) > len(self.ready_groups):
            self.ready_group_off_policy_steps = self.ready_group_off_policy_steps[
                : len(self.ready_groups)
            ]

    async def aclose(self) -> None:
        self.finish_policy_update()
        for task in self.pending:
            task.cancel()
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
        self.pending.clear()
        self.pending_clients.clear()
        self.groups.clear()
        if hasattr(self, "ready_groups"):
            self.ready_groups.clear()
        if hasattr(self, "ready_group_off_policy_steps"):
            self.ready_group_off_policy_steps.clear()

    def _fill_inflight(self) -> None:
        if self.policy_update_in_progress:
            return
        while self.inflight_rollout_count < self.max_inflight_rollouts:
            if not self._schedule_next_rollout():
                break

    @property
    def policy_update_in_progress(self) -> bool:
        event = getattr(self, "_policy_update_ready", None)
        return event is not None and not event.is_set()

    def begin_policy_update(self) -> None:
        event = getattr(self, "_policy_update_ready", None)
        if event is None:
            event = asyncio.Event()
            self._policy_update_ready = event
        event.clear()

    def finish_policy_update(self) -> None:
        event = getattr(self, "_policy_update_ready", None)
        if event is not None:
            event.set()

    async def drain_policy_update_requests(self) -> None:
        """Wait until every request admitted before the update has completed."""
        if not self.policy_update_in_progress:
            raise RuntimeError("Policy request draining requires a paused scheduler.")
        pending = tuple(getattr(self, "pending", ()))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _wait_for_policy_update(self) -> None:
        event = getattr(self, "_policy_update_ready", None)
        if event is None or event.is_set():
            return
        started_at = perf_counter()
        await event.wait()
        self.policy_update_wait_seconds = getattr(
            self,
            "policy_update_wait_seconds",
            0.0,
        ) + (perf_counter() - started_at)

    def _schedule_next_rollout(self) -> bool:
        remaining_capacity = self.max_inflight_rollouts - self.inflight_rollout_count
        if remaining_capacity <= 0:
            return False

        current_policy_step = getattr(self, "policy_step", None)
        for group_id, group in list(self.groups.items()):
            if group.rollouts_to_schedule <= 0:
                continue
            if group.policy_step != current_policy_step:
                self.groups.pop(group_id, None)
                self.rejected_groups_count = (
                    getattr(self, "rejected_groups_count", 0) + 1
                )
                continue
            cost = group.rollouts_to_schedule if self.requires_group_scoring else 1
            if cost <= remaining_capacity:
                self._schedule_group_rollout(group_id, group)
                return True

        if remaining_capacity < self.rollout_count:
            return False

        record = self._next_record()
        group_id = self.next_group_id
        self.next_group_id += 1
        group = _VerifierGroupState(
            example=_verifier_example(record),
            rollouts_to_schedule=self.rollout_count,
            policy_step=current_policy_step,
        )
        self.groups[group_id] = group
        self._schedule_group_rollout(group_id, group)
        return True

    def _schedule_group_rollout(
        self,
        group_id: int,
        group: _VerifierGroupState,
    ) -> None:
        if group.pinned_client_index is None:
            group.pinned_client_index = self._least_loaded_client_index()
        client_index = group.pinned_client_index
        client = self.clients[client_index]
        if self.requires_group_scoring:
            rollout_count = group.rollouts_to_schedule
            group.rollouts_to_schedule = 0
            task = asyncio.create_task(
                _run_group(
                    self.vf,
                    self.env,
                    group.example,
                    client=client,
                    model=self.model,
                    sampling_args=self._sampling_args_for_current_policy(),
                    rollout_count=rollout_count,
                    max_retries=self.config.orchestrator.verifier_max_retries,
                    algorithm_config=self.config.algo,
                )
            )
        else:
            rollout_count = 1
            group.rollouts_to_schedule -= 1
            task = asyncio.create_task(
                _run_single_rollout(
                    self.vf,
                    self.env,
                    group.example,
                    client=client,
                    model=self.model,
                    sampling_args=self._sampling_args_for_current_policy(),
                    max_retries=self.config.orchestrator.verifier_max_retries,
                )
            )
        self.pending[task] = _PendingVerifierRequest(
            group_id=group_id,
            client_index=client_index,
            rollout_count=rollout_count,
            policy_step=group.policy_step,
        )
        self.pending_clients[task] = client_index

    def _next_record(self) -> RLExample:
        epoch, offset = divmod(self.record_cursor, len(self.records))
        if self._record_order_epoch != epoch:
            self._record_order = list(range(len(self.records)))
            if self.config.data.shuffle:
                rng = random.Random(self.config.data.seed + epoch * 1_000_003)
                rng.shuffle(self._record_order)
            self._record_order_epoch = epoch
        record = self.records[self._record_order[offset]]
        self.record_cursor += 1
        return record

    def _least_loaded_client_index(self) -> int:
        counts = [0] * len(self.clients)
        for request in self.pending.values():
            counts[request.client_index] += request.rollout_count
        return min(range(len(self.clients)), key=counts.__getitem__)

    def _sampling_args_for_current_policy(self) -> dict[str, Any]:
        cache_salt = None if self.policy_step is None else str(self.policy_step)
        return _sampling_args(self.config, cache_salt=cache_salt)


import asyncio


import json


import subprocess


import sys


from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait


from dataclasses import dataclass


from pathlib import Path


from time import perf_counter, sleep


from wavelet.configs.rl_config import RLConfig


from wavelet.data.rl import RLExample


from wavelet.inference.policy import create_policy_inference_engine


from wavelet.transport.queue import (
    FileSystemPolicyReceiver,
    FileSystemRolloutSender,
    QueueEvent,
    append_event_best_effort,
    publish_adapter_policy_snapshot,
    utc_now,
    validate_rollout_manifest,
)


from wavelet.orchestrator.policy_metadata import policy_metadata


from wavelet.orchestrator.metrics import log_eval_metrics, log_rollout_metrics


from wavelet.orchestrator.rollouts import RLOrchestrator


from wavelet.orchestrator.schedule import (
    chunks_per_step as _chunks_per_step,
    policy_step_to_load as _policy_step_to_load,
    required_policy_step as _required_policy_step,
    rollout_chunk_examples as _rollout_chunk_examples,
    rollout_groups_for_chunk as _rollout_groups_for_chunk,
    select_due_eval_envs,
    target_steps as _target_steps,
)


from wavelet.orchestrator.state_server import OrchestratorRunState, maybe_state_server


from wavelet.orchestrator.sources import RolloutSourceKind


from wavelet.utils.config import load_config


from wavelet.utils.monitoring import emit_perf


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

    from wavelet.orchestrator.envs import (
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


def _reusable_rollout_batch(
    config: RLConfig,
    sender: FileSystemRolloutSender,
    *,
    queue_step: int,
    optimizer_step: int,
    chunk_index: int | None,
):
    batch = sender.stable_batch(queue_step)
    if batch is None:
        return None
    row_count = _count_nonempty_lines(batch.path)
    validate_rollout_manifest(
        batch,
        queue_step=queue_step,
        optimizer_step=optimizer_step,
        chunk_index=chunk_index,
        rows=row_count,
        minimum_policy_step=_required_policy_step(config, optimizer_step),
        maximum_policy_step=optimizer_step,
    )
    append_event_best_effort(
        config.output_dir / "events",
        QueueEvent(
            time=utc_now(),
            kind="rollout_reused",
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            details={"path": str(batch.path)},
        ),
    )
    return batch


@dataclass(slots=True)
class RolloutScheduler:
    """Single process-mode scheduler configured by source and publish policy."""

    config: RLConfig
    orchestrator: RLOrchestrator
    inference_engine: object
    policy_receiver: FileSystemPolicyReceiver
    state: OrchestratorRunState | None = None

    def run(self, *, target_step: int, start_step: int = 0) -> int:
        schedule = resolve_rollout_schedule(self.config)
        common = {
            "config": self.config,
            "orchestrator": self.orchestrator,
            "inference_engine": self.inference_engine,
            "policy_receiver": self.policy_receiver,
            "target_step": target_step,
            "start_step": start_step,
            "state": self.state,
        }
        if (
            schedule.source is RolloutSourceKind.VERIFIER
            and schedule.publish_mode is PublishMode.STREAMING
        ):
            return asyncio.run(_run_verifier_scheduler(**common))
        if (
            schedule.source is RolloutSourceKind.NATIVE
            and schedule.publish_mode is PublishMode.STREAMING
        ):
            return _run_chunk_scheduler(**common)
        return _run_batched_scheduler(**common)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    start_step = _resume_optimizer_step(config)
    policy_receiver = FileSystemPolicyReceiver(
        config.output_dir,
        config.policy_transfer,
        start_step=start_step,
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
        result = RolloutScheduler(
            config=config,
            orchestrator=orchestrator,
            inference_engine=inference_engine,
            policy_receiver=policy_receiver,
            state=state,
        ).run(target_step=target_step, start_step=start_step)
        if state is not None:
            state.set_status("completed", phase="completed")
        return result


@dataclass
class _SchedulerStateMachine:
    config: RLConfig
    orchestrator: RLOrchestrator
    inference_engine: object
    policy_receiver: FileSystemPolicyReceiver
    state: OrchestratorRunState | None
    rollout_sender: FileSystemRolloutSender
    loaded_policy_step: int | None = None
    next_step_to_submit: int = 0
    next_step_to_publish: int = 0
    pending_policy_load: Future[tuple[int, float, float]] | None = None

    def __post_init__(self) -> None:
        self.pending: dict[Future[tuple[int, object, float, float]], int] = {}
        self.completed: dict[int, tuple[object, float, float]] = {}
        self.last_eval_steps = _initial_eval_steps(self.config)

    def submit_step(self, pool: ThreadPoolExecutor, step: int) -> None:
        future = pool.submit(
            _publish_step,
            self.orchestrator,
            self.rollout_sender,
            step,
            self.inference_engine,
            self.loaded_policy_step,
        )
        self.pending[future] = step

    def finish_policy_load(self, *, block: bool = False) -> bool:
        pending = self.pending_policy_load
        if pending is None or (not block and not pending.done()):
            return False
        policy_step, wait_seconds, load_seconds = pending.result()
        self.loaded_policy_step = policy_step
        self.pending_policy_load = None
        self._record_loaded_policy(policy_step)
        emit_perf(
            "policy_load",
            step=policy_step,
            wait_policy=wait_seconds,
            load_policy=load_seconds,
        )
        return True

    def _record_loaded_policy(self, policy_step: int) -> None:
        if self.state is not None:
            self.state.update_policy(
                loaded_step=policy_step,
                pending_load=False,
                requested_step=None,
                available_tail=self.policy_receiver.available_steps()[-20:],
            )
        _maybe_run_evals(
            self.config,
            self.orchestrator,
            policy_step=policy_step,
            rollout_step=self._rollout_step(self.next_step_to_submit),
            last_eval_steps=self.last_eval_steps,
        )

    def _rollout_step(self, queue_step: int) -> int:
        return queue_step

    def collect_done(self, done) -> None:
        for future in done:
            step = self.pending.pop(future)
            _, batch, materialize_seconds, publish_seconds = future.result()
            self.completed[step] = (batch, materialize_seconds, publish_seconds)
            if self.state is not None:
                self.state.mark_completed(
                    queue_step=step,
                    optimizer_step=step,
                    pending_count=len(self.pending),
                    completed_count=len(self.completed),
                )

    def publish_ready(self) -> bool:
        published = False
        while self.next_step_to_publish in self.completed:
            step = self.next_step_to_publish
            batch, materialize_seconds, publish_seconds = self.completed.pop(step)
            self.next_step_to_publish += 1
            if self.state is not None:
                self.state.mark_published(
                    queue_step=step,
                    optimizer_step=step,
                    path=str(batch.path),
                    next_queue_step_to_publish=self.next_step_to_publish,
                    completed_count=len(self.completed),
                )
            _sleep_for_colocated_sleep(self.config, self.inference_engine)
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

    def collect_finished_rollouts(self) -> bool:
        done = [future for future in self.pending if future.done()]
        if not done:
            return False
        self.collect_done(done)
        return self.publish_ready()

    def wait_for_one_rollout(self) -> None:
        done, _ = wait(self.pending, return_when=FIRST_COMPLETED)
        self.collect_done(done)
        self.publish_ready()

    def _start_policy_load(
        self,
        policy_pool: ThreadPoolExecutor,
        policy_step: int,
    ) -> None:
        self.pending_policy_load = policy_pool.submit(
            _load_policy_step,
            self.config,
            self.inference_engine,
            self.policy_receiver,
            policy_step,
        )
        if self.state is not None:
            self.state.update_policy(
                pending_load=True,
                requested_step=policy_step,
                available_tail=self.policy_receiver.available_steps()[-20:],
            )

    def _load_policy_now(self, policy_step: int) -> tuple[float, float]:
        loaded_step, wait_seconds, load_seconds = _load_policy_step(
            self.config,
            self.inference_engine,
            self.policy_receiver,
            policy_step,
        )
        self.loaded_policy_step = loaded_step
        self._record_loaded_policy(loaded_step)
        return wait_seconds, load_seconds

    def prepare_step(
        self,
        policy_pool: ThreadPoolExecutor,
        step: int,
    ) -> tuple[bool, float, float]:
        return self._prepare_policy(
            policy_pool,
            rollout_step=self._rollout_step(step),
            load_optional_while_pending=True,
            start_required_load_while_pending=True,
        )

    def _prepare_policy(
        self,
        policy_pool: ThreadPoolExecutor,
        *,
        rollout_step: int,
        load_optional_while_pending: bool,
        start_required_load_while_pending: bool,
    ) -> tuple[bool, float, float]:
        policy_step = _policy_step_to_load(
            self.config,
            self.policy_receiver,
            rollout_step=rollout_step,
            loaded_policy_step=self.loaded_policy_step,
        )
        if policy_step is None:
            _wake_for_colocated_sleep(self.config, self.inference_engine)
            return True, 0.0, 0.0

        required_step = _required_policy_step(self.config, rollout_step)
        must_load = (
            self.loaded_policy_step is None or self.loaded_policy_step < required_step
        )
        if not must_load:
            if self.pending_policy_load is None and (
                load_optional_while_pending or not self.pending
            ):
                if not load_optional_while_pending:
                    wait_seconds, load_seconds = self._load_policy_now(policy_step)
                    return True, wait_seconds, load_seconds
                self._start_policy_load(policy_pool, policy_step)
            return True, 0.0, 0.0
        if self.publish_ready():
            return False, 0.0, 0.0
        if self.pending_policy_load is not None:
            if not self.finish_policy_load() and self.pending:
                self.wait_for_one_rollout()
            elif self.pending_policy_load is not None:
                self.finish_policy_load(block=True)
            return False, 0.0, 0.0
        if self.pending:
            if start_required_load_while_pending:
                self._start_policy_load(policy_pool, policy_step)
            self.wait_for_one_rollout()
            return False, 0.0, 0.0
        wait_seconds, load_seconds = self._load_policy_now(policy_step)
        return True, wait_seconds, load_seconds

    def run(self, *, target_step: int, prefetch_steps: int) -> int:
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
            self._run_until_complete(
                pool,
                policy_pool,
                target_step=target_step,
                prefetch_steps=prefetch_steps,
            )
        self.loaded_policy_step = _run_final_evals(
            self.config,
            self.orchestrator,
            self.inference_engine,
            self.policy_receiver,
            target_step=target_step,
            loaded_policy_step=self.loaded_policy_step,
        )
        if self.state is not None:
            self.state.set_status("completed", phase="completed")
        return 0

    def _run_until_complete(
        self,
        pool: ThreadPoolExecutor,
        policy_pool: ThreadPoolExecutor,
        *,
        target_step: int,
        prefetch_steps: int,
    ) -> None:
        while (
            self.next_step_to_submit < target_step
            or self.pending
            or self.pending_policy_load
        ):
            self.finish_policy_load()
            if self.collect_finished_rollouts():
                continue
            submitted = self._submit_available_steps(
                pool,
                policy_pool,
                target_step=target_step,
                prefetch_steps=prefetch_steps,
            )
            if submitted or self.finish_policy_load():
                continue
            if self.pending:
                self.wait_for_one_rollout()
            elif self.pending_policy_load is not None:
                self.finish_policy_load(block=True)

    def _submit_available_steps(
        self,
        pool: ThreadPoolExecutor,
        policy_pool: ThreadPoolExecutor,
        *,
        target_step: int,
        prefetch_steps: int,
    ) -> bool:
        submitted = False
        while (
            self.next_step_to_submit < target_step
            and len(self.pending) < prefetch_steps
        ):
            step = self.next_step_to_submit
            started_at = perf_counter()
            ready, wait_seconds, load_seconds = self.prepare_step(policy_pool, step)
            if not ready:
                continue
            self.next_step_to_submit += 1
            existing = _reusable_rollout_batch(
                self.config,
                self.rollout_sender,
                queue_step=step,
                optimizer_step=step,
                chunk_index=None,
            )
            if existing is not None:
                self._record_submitted_step(step)
                self.completed[step] = (existing, 0.0, 0.0)
                self.publish_ready()
            else:
                self.submit_step(pool, step)
                self._record_submitted_step(step)
            submitted = True
            emit_perf(
                "inference_submit",
                step=step,
                wait_policy=wait_seconds,
                load_policy=load_seconds,
                submit=perf_counter() - started_at,
            )
        return submitted

    def _record_submitted_step(self, step: int) -> None:
        if self.state is None:
            return
        self.state.update_rollouts(
            next_queue_step_to_submit=self.next_step_to_submit,
        )
        self.state.mark_submitted(
            queue_step=step,
            optimizer_step=step,
            pending_count=len(self.pending),
        )


def _run_batched_scheduler(
    *,
    config: RLConfig,
    orchestrator: RLOrchestrator,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    target_step: int,
    start_step: int = 0,
    state: OrchestratorRunState | None = None,
) -> int:
    prefetch_steps = max(1, min(config.orchestrator.max_async_level, target_step))
    context = _SchedulerStateMachine(
        config=config,
        orchestrator=orchestrator,
        inference_engine=inference_engine,
        policy_receiver=policy_receiver,
        state=state,
        rollout_sender=FileSystemRolloutSender(config.output_dir, config.transport),
        next_step_to_submit=start_step,
        next_step_to_publish=start_step,
    )
    return context.run(target_step=target_step, prefetch_steps=prefetch_steps)


class _ChunkPublisherStrategy(_SchedulerStateMachine):
    def __init__(
        self,
        *,
        config: RLConfig,
        orchestrator: RLOrchestrator,
        inference_engine: object,
        policy_receiver: FileSystemPolicyReceiver,
        state: OrchestratorRunState | None,
        rollout_sender: FileSystemRolloutSender,
        chunks_per_step: int,
        chunk_examples: int,
        next_step_to_submit: int = 0,
        next_step_to_publish: int = 0,
    ) -> None:
        self.chunks_per_step = chunks_per_step
        self.chunk_examples = chunk_examples
        self.published_queue_steps: set[int] = set()
        super().__init__(
            config=config,
            orchestrator=orchestrator,
            inference_engine=inference_engine,
            policy_receiver=policy_receiver,
            state=state,
            rollout_sender=rollout_sender,
            next_step_to_submit=next_step_to_submit,
            next_step_to_publish=next_step_to_publish,
        )

    def submit_step(self, pool: ThreadPoolExecutor, queue_step: int) -> None:
        optimizer_step = queue_step // self.chunks_per_step
        chunk_index = queue_step % self.chunks_per_step
        future = pool.submit(
            _publish_native_chunk,
            self.orchestrator,
            self.rollout_sender,
            optimizer_step,
            chunk_index,
            queue_step,
            self.chunk_examples,
            self.inference_engine,
            self.loaded_policy_step,
        )
        self.pending[future] = queue_step

    def _rollout_step(self, queue_step: int) -> int:
        return queue_step // self.chunks_per_step

    def collect_done(self, done) -> None:
        for future in done:
            queue_step = self.pending.pop(future)
            _, batch, materialize_seconds, publish_seconds = future.result()
            self.completed[queue_step] = (
                batch,
                materialize_seconds,
                publish_seconds,
            )
            if self.state is not None:
                self.state.mark_completed(
                    queue_step=queue_step,
                    optimizer_step=queue_step // self.chunks_per_step,
                    chunk_index=queue_step % self.chunks_per_step,
                    pending_count=len(self.pending),
                    completed_count=len(self.completed),
                )

    def publish_ready(self) -> bool:
        published = False
        for queue_step in sorted(self.completed):
            batch, materialize_seconds, publish_seconds = self.completed.pop(queue_step)
            optimizer_step = queue_step // self.chunks_per_step
            chunk_index = queue_step % self.chunks_per_step
            self.published_queue_steps.add(queue_step)
            while self.next_step_to_publish in self.published_queue_steps:
                self.published_queue_steps.remove(self.next_step_to_publish)
                self.next_step_to_publish += 1
            if self.state is not None:
                self.state.mark_published(
                    queue_step=queue_step,
                    optimizer_step=optimizer_step,
                    chunk_index=chunk_index,
                    path=str(batch.path),
                    next_queue_step_to_publish=self.next_step_to_publish,
                    completed_count=len(self.completed),
                )
            _sleep_for_colocated_sleep(self.config, self.inference_engine)
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

    def start_policy_load_if_available(
        self,
        policy_pool: ThreadPoolExecutor,
        optimizer_step: int,
    ) -> bool:
        if self.pending_policy_load is not None or self.pending:
            return False
        policy_step = _policy_step_to_load(
            self.config,
            self.policy_receiver,
            rollout_step=optimizer_step,
            loaded_policy_step=self.loaded_policy_step,
        )
        if policy_step is None:
            return False
        self._start_policy_load(policy_pool, policy_step)
        return True

    def prepare_step(
        self,
        policy_pool: ThreadPoolExecutor,
        queue_step: int,
    ) -> tuple[bool, float, float]:
        return self._prepare_policy(
            policy_pool,
            rollout_step=self._rollout_step(queue_step),
            load_optional_while_pending=False,
            start_required_load_while_pending=False,
        )

    def run_native(self, *, target_chunks: int, max_pending_chunks: int) -> int:
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
            self._run_native_until_complete(
                pool,
                policy_pool,
                target_chunks=target_chunks,
                max_pending_chunks=max_pending_chunks,
            )
        target_step = target_chunks // self.chunks_per_step
        self.loaded_policy_step = _run_final_evals(
            self.config,
            self.orchestrator,
            self.inference_engine,
            self.policy_receiver,
            target_step=target_step,
            loaded_policy_step=self.loaded_policy_step,
        )
        if self.state is not None:
            self.state.set_status("completed", phase="completed")
        return 0

    def _run_native_until_complete(
        self,
        pool: ThreadPoolExecutor,
        policy_pool: ThreadPoolExecutor,
        *,
        target_chunks: int,
        max_pending_chunks: int,
    ) -> None:
        while (
            self.next_step_to_submit < target_chunks
            or self.pending
            or self.pending_policy_load
        ):
            self.finish_policy_load()
            self.collect_finished_rollouts()
            if self.next_step_to_submit < target_chunks and self.pending:
                self.start_policy_load_if_available(
                    policy_pool,
                    self.next_step_to_submit // self.chunks_per_step,
                )
            submitted = self._submit_available_chunks(
                pool,
                policy_pool,
                target_chunks=target_chunks,
                max_pending_chunks=max_pending_chunks,
            )
            if submitted or self.finish_policy_load():
                continue
            self._wait_for_native_progress(policy_pool, target_chunks=target_chunks)

    def _submit_available_chunks(
        self,
        pool: ThreadPoolExecutor,
        policy_pool: ThreadPoolExecutor,
        *,
        target_chunks: int,
        max_pending_chunks: int,
    ) -> bool:
        submitted = False
        while (
            self.next_step_to_submit < target_chunks
            and len(self.pending) < max_pending_chunks
        ):
            queue_step = self.next_step_to_submit
            optimizer_step = queue_step // self.chunks_per_step
            started_at = perf_counter()
            ready, wait_seconds, load_seconds = self.prepare_step(
                policy_pool,
                queue_step,
            )
            if not ready:
                continue
            self.next_step_to_submit += 1
            existing = _reusable_rollout_batch(
                self.config,
                self.rollout_sender,
                queue_step=queue_step,
                optimizer_step=optimizer_step,
                chunk_index=queue_step % self.chunks_per_step,
            )
            if existing is not None:
                self._record_submitted_chunk(queue_step)
                self.completed[queue_step] = (existing, 0.0, 0.0)
                self.publish_ready()
            else:
                self.submit_step(pool, queue_step)
                self._record_submitted_chunk(queue_step)
            submitted = True
            emit_perf(
                "inference_native_submit",
                queue_step=queue_step,
                optimizer_step=optimizer_step,
                chunk_index=queue_step % self.chunks_per_step,
                wait_policy=wait_seconds,
                load_policy=load_seconds,
                submit=perf_counter() - started_at,
            )
        return submitted

    def _record_submitted_chunk(self, queue_step: int) -> None:
        if self.state is None:
            return
        self.state.update_rollouts(
            next_queue_step_to_submit=self.next_step_to_submit,
        )
        self.state.mark_submitted(
            queue_step=queue_step,
            optimizer_step=queue_step // self.chunks_per_step,
            chunk_index=queue_step % self.chunks_per_step,
            pending_count=len(self.pending),
        )

    def _wait_for_native_progress(
        self,
        policy_pool: ThreadPoolExecutor,
        *,
        target_chunks: int,
    ) -> None:
        if self.pending:
            if self.next_step_to_submit < target_chunks:
                self.start_policy_load_if_available(
                    policy_pool,
                    self.next_step_to_submit // self.chunks_per_step,
                )
            done, _ = wait(
                self.pending,
                timeout=self.config.transport.poll_interval_seconds,
                return_when=FIRST_COMPLETED,
            )
            if done:
                self.collect_done(done)
                self.publish_ready()
        elif self.pending_policy_load is not None:
            self.finish_policy_load(block=True)


@dataclass
class _VerifierPublisherStrategy:
    config: RLConfig
    orchestrator: RLOrchestrator
    inference_engine: object
    policy_receiver: FileSystemPolicyReceiver
    scheduler: object
    rollout_sender: FileSystemRolloutSender
    state: OrchestratorRunState | None
    chunks_per_step: int
    last_eval_steps: dict[str, int]
    loaded_policy_step: int | None = None
    pending_policy_update: asyncio.Task[int] | None = None

    async def prepare_policy(self, optimizer_step: int) -> tuple[float, float]:
        """Make the required policy available without blocking optional refreshes."""
        await self._finish_ready_policy_update(optimizer_step)
        policy_step = _policy_step_to_load(
            self.config,
            self.policy_receiver,
            rollout_step=optimizer_step,
            loaded_policy_step=self.loaded_policy_step,
        )
        if policy_step is None:
            _wake_for_colocated_sleep(self.config, self.inference_engine)
            return 0.0, 0.0

        required_step = _required_policy_step(self.config, optimizer_step)
        must_load = (
            self.loaded_policy_step is None or self.loaded_policy_step < required_step
        )
        if self.pending_policy_update is None:
            if must_load:
                return await self._load_now(policy_step, optimizer_step)
            self._start_background_load(policy_step)
            return 0.0, 0.0
        if not must_load:
            return 0.0, 0.0
        return await self._wait_for_required_policy(
            policy_step,
            required_step=required_step,
            optimizer_step=optimizer_step,
        )

    async def _finish_ready_policy_update(self, optimizer_step: int) -> None:
        pending = self.pending_policy_update
        if pending is None or not pending.done():
            return
        self.pending_policy_update = None
        self.loaded_policy_step = pending.result()
        await self._record_loaded_policy(optimizer_step)
        emit_perf(
            "policy_load",
            step=self.loaded_policy_step,
            wait_policy=0.0,
            load_policy="async",
        )

    def _start_background_load(self, policy_step: int) -> None:
        self.scheduler.begin_policy_update()
        try:
            self.pending_policy_update = asyncio.create_task(
                _load_policy_and_update_scheduler(
                    self.config,
                    self.inference_engine,
                    self.policy_receiver,
                    policy_step,
                    self.scheduler,
                )
            )
        except Exception:
            self.scheduler.finish_policy_update()
            raise
        if self.state is not None:
            self.state.update_policy(
                pending_load=True,
                requested_step=policy_step,
                available_tail=self.policy_receiver.available_steps()[-20:],
            )

    async def _load_now(
        self,
        policy_step: int,
        optimizer_step: int,
    ) -> tuple[float, float]:
        started_at = perf_counter()
        previous_policy_step = self.loaded_policy_step
        self.scheduler.begin_policy_update()
        try:
            await self.scheduler.drain_policy_update_requests()
            self.loaded_policy_step = await _load_policy_async(
                self.config,
                self.inference_engine,
                self.policy_receiver,
                policy_step,
            )
            self.scheduler.set_policy_step(
                self.loaded_policy_step,
                model_name=_current_policy_model_name(self.inference_engine),
            )
            if (
                previous_policy_step is not None
                and self.loaded_policy_step != previous_policy_step
            ):
                await self.scheduler.mark_policy_update()
        finally:
            self.scheduler.finish_policy_update()
        await self._record_loaded_policy(optimizer_step)
        elapsed = perf_counter() - started_at
        return elapsed, elapsed

    async def _wait_for_required_policy(
        self,
        policy_step: int,
        *,
        required_step: int,
        optimizer_step: int,
    ) -> tuple[float, float]:
        started_at = perf_counter()
        assert self.pending_policy_update is not None
        self.loaded_policy_step = await self.pending_policy_update
        self.pending_policy_update = None
        await self._record_loaded_policy(optimizer_step)
        elapsed = perf_counter() - started_at
        emit_perf(
            "policy_load",
            step=self.loaded_policy_step,
            wait_policy=elapsed,
            load_policy="async",
        )
        if self.loaded_policy_step < required_step:
            extra_wait, _ = await self._load_now(policy_step, optimizer_step)
            elapsed += extra_wait
        return elapsed, elapsed

    async def _record_loaded_policy(self, optimizer_step: int) -> None:
        if self.state is not None:
            self.state.update_policy(
                loaded_step=self.loaded_policy_step,
                pending_load=False,
                requested_step=None,
                available_tail=self.policy_receiver.available_steps()[-20:],
            )
        await _maybe_run_evals_async(
            self.config,
            self.orchestrator,
            policy_step=self.loaded_policy_step,
            rollout_step=optimizer_step,
            last_eval_steps=self.last_eval_steps,
        )

    async def publish_chunk(
        self,
        queue_step: int,
        *,
        wait_policy_seconds: float,
        load_policy_seconds: float,
    ) -> None:
        step_started_at = perf_counter()
        optimizer_step = queue_step // self.chunks_per_step
        existing = _reusable_rollout_batch(
            self.config,
            self.rollout_sender,
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=queue_step % self.chunks_per_step,
        )
        if existing is not None:
            self._record_published_chunk(
                queue_step,
                optimizer_step=optimizer_step,
                path=existing.path,
            )
            print(existing.path)
            return
        generate_started_at = perf_counter()
        chunk_index = queue_step % self.chunks_per_step
        chunk_groups = _rollout_groups_for_chunk(self.config, chunk_index)
        records = await self.scheduler.generate_batch(target_groups=chunk_groups)
        generate_seconds = perf_counter() - generate_started_at
        rollout_policy_step = _rollout_records_policy_step(
            records,
            fallback=self.loaded_policy_step,
        )

        materialize_started_at = perf_counter()
        materialized_path = _write_materialized_records(
            self.orchestrator,
            records,
            step=queue_step,
        )
        materialize_seconds = perf_counter() - materialize_started_at
        publish_started_at = perf_counter()
        batch = self.rollout_sender.publish(
            materialized_path,
            step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=queue_step % self.chunks_per_step,
            policy_step=rollout_policy_step,
            rows=_count_nonempty_lines(materialized_path),
        )
        publish_seconds = perf_counter() - publish_started_at
        self._record_published_chunk(
            queue_step,
            optimizer_step=optimizer_step,
            path=batch.path,
        )
        log_rollout_metrics(
            self.config,
            materialized_path,
            step=optimizer_step,
            policy_step=rollout_policy_step,
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=queue_step % self.chunks_per_step,
            timings={
                "generate_completions": generate_seconds,
                "parallel_preprocess": materialize_seconds,
                "publish": publish_seconds,
                "step": perf_counter() - step_started_at,
            },
            extra_metrics=getattr(self.scheduler, "last_batch_metrics", None),
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
            pending_policy_update=int(self.pending_policy_update is not None),
            total=perf_counter() - step_started_at,
        )
        print(batch.path)

    def _record_published_chunk(
        self,
        queue_step: int,
        *,
        optimizer_step: int,
        path: Path,
    ) -> None:
        if self.state is None:
            return
        chunk_index = queue_step % self.chunks_per_step
        self.state.update_rollouts(next_queue_step_to_submit=queue_step + 1)
        self.state.mark_submitted(
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=chunk_index,
            pending_count=0,
        )
        self.state.mark_completed(
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=chunk_index,
            pending_count=0,
            completed_count=1,
        )
        self.state.mark_published(
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=chunk_index,
            path=str(path),
            next_queue_step_to_publish=queue_step + 1,
            completed_count=0,
        )

    async def finish_pending_policy(self, optimizer_step: int) -> None:
        if self.pending_policy_update is None:
            return
        self.loaded_policy_step = await self.pending_policy_update
        self.pending_policy_update = None
        await self._record_loaded_policy(optimizer_step)

    async def run_final_evals(self, target_step: int) -> None:
        if self.config.eval is None or not self.config.eval.final_eval:
            return
        final_policy_step = _final_eval_policy_step(self.config, target_step)
        if final_policy_step is None:
            return
        if (
            target_step == 0
            and self.loaded_policy_step is None
            and self.config.model.adapter_path is None
        ):
            self.loaded_policy_step = 0
            self.scheduler.set_policy_step(
                0,
                model_name=_current_policy_model_name(self.inference_engine),
            )
        elif (
            self.loaded_policy_step is None
            or self.loaded_policy_step < final_policy_step
        ):
            policy = await asyncio.to_thread(
                self.policy_receiver.wait_for_step,
                final_policy_step,
            )
            _wake_for_colocated_sleep(
                self.config,
                self.inference_engine,
                tags=["weights"],
            )
            self.inference_engine.load_policy(policy.step_dir, step=policy.step)
            _wake_for_colocated_sleep(
                self.config,
                self.inference_engine,
                tags=["kv_cache"],
            )
            self.loaded_policy_step = policy.step
            self.scheduler.set_policy_step(
                policy.step,
                model_name=_current_policy_model_name(self.inference_engine),
            )
        else:
            _wake_for_colocated_sleep(self.config, self.inference_engine)
        await _run_evals_async(
            self.config,
            self.orchestrator,
            policy_step=self.loaded_policy_step,
            rollout_step=target_step,
            envs=self.config.eval.env,
        )
        _sleep_for_colocated_sleep(self.config, self.inference_engine)

    async def close(self) -> None:
        if self.pending_policy_update is not None:
            self.pending_policy_update.cancel()
            await asyncio.gather(
                self.pending_policy_update,
                return_exceptions=True,
            )
        await self.scheduler.aclose()
        await _teardown_cached_verifier_envs()


def _run_chunk_scheduler(
    *,
    config: RLConfig,
    orchestrator: RLOrchestrator,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    target_step: int,
    start_step: int = 0,
    state: OrchestratorRunState | None = None,
) -> int:
    chunks_per_step = _chunks_per_step(config)
    target_chunks = target_step * chunks_per_step
    configured_limit = (
        config.orchestrator.max_pending_rollout_chunks
        or config.orchestrator.max_async_level * chunks_per_step
    )
    max_pending_chunks = max(1, min(configured_limit, target_chunks))
    start_queue_step = start_step * chunks_per_step
    context = _ChunkPublisherStrategy(
        config=config,
        orchestrator=orchestrator,
        inference_engine=inference_engine,
        policy_receiver=policy_receiver,
        state=state,
        rollout_sender=FileSystemRolloutSender(config.output_dir, config.transport),
        chunks_per_step=chunks_per_step,
        chunk_examples=_rollout_chunk_examples(config),
        next_step_to_submit=start_queue_step,
        next_step_to_publish=start_queue_step,
    )
    return context.run_native(
        target_chunks=target_chunks,
        max_pending_chunks=max_pending_chunks,
    )


async def _run_verifier_scheduler(
    *,
    config: RLConfig,
    orchestrator: RLOrchestrator,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    target_step: int,
    start_step: int = 0,
    state: OrchestratorRunState | None = None,
) -> int:
    from wavelet.orchestrator.scheduler import VerifierRolloutScheduler

    examples_per_step = config.orchestrator.examples_per_step
    if examples_per_step is None:
        raise ValueError("orchestrator.examples_per_step is required.")
    scheduler = VerifierRolloutScheduler(
        orchestrator,
        start_record_cursor=start_step * examples_per_step,
    )
    chunks_per_step = _chunks_per_step(config)
    context = _VerifierPublisherStrategy(
        config=config,
        orchestrator=orchestrator,
        inference_engine=inference_engine,
        policy_receiver=policy_receiver,
        scheduler=scheduler,
        rollout_sender=FileSystemRolloutSender(config.output_dir, config.transport),
        state=state,
        chunks_per_step=chunks_per_step,
        last_eval_steps=_initial_eval_steps(config),
    )
    target_chunks = target_step * chunks_per_step
    try:
        for queue_step in range(start_step * chunks_per_step, target_chunks):
            optimizer_step = queue_step // chunks_per_step
            wait_seconds, load_seconds = await context.prepare_policy(optimizer_step)
            await context.publish_chunk(
                queue_step,
                wait_policy_seconds=wait_seconds,
                load_policy_seconds=load_seconds,
            )
        await context.finish_pending_policy(target_step)
        await context.run_final_evals(target_step)
        if state is not None:
            state.set_status("completed", phase="completed")
        return 0
    finally:
        await context.close()


def _initial_eval_steps(config: RLConfig) -> dict[str, int]:
    if config.eval is None:
        return {}
    return {env.resolved_name: -1 for env in config.eval.env}


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
    scheduler.begin_policy_update()
    try:
        await scheduler.drain_policy_update_requests()
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
    finally:
        scheduler.finish_policy_update()


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


def _rollout_records_policy_step(
    records: list[RLExample],
    *,
    fallback: int | None,
) -> int | None:
    """Return the oldest policy that contributed trajectories to a batch."""
    policy_steps = [
        value
        for record in records
        if isinstance(record.metadata, dict)
        for value in [record.metadata.get("policy_step")]
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    return min(policy_steps) if policy_steps else fallback


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


def _run_final_evals(
    config: RLConfig,
    orchestrator: RLOrchestrator,
    inference_engine,
    policy_receiver: FileSystemPolicyReceiver,
    *,
    target_step: int,
    loaded_policy_step: int | None,
) -> int | None:
    """Load the final policy when necessary and run configured final evals."""
    if config.eval is None or not config.eval.final_eval:
        return loaded_policy_step
    final_policy_step = _final_eval_policy_step(config, target_step)
    if final_policy_step is None:
        return loaded_policy_step

    if (
        target_step == 0
        and loaded_policy_step is None
        and config.model.adapter_path is None
    ):
        loaded_policy_step = 0
    elif loaded_policy_step is None or loaded_policy_step < final_policy_step:
        policy = policy_receiver.wait_for_step(final_policy_step)
        _wake_for_colocated_sleep(config, inference_engine, tags=["weights"])
        inference_engine.load_policy(policy.step_dir, step=policy.step)
        _wake_for_colocated_sleep(config, inference_engine, tags=["kv_cache"])
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
    return loaded_policy_step


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

    from wavelet.orchestrator.envs import evaluate_env

    for env in envs:
        metrics = evaluate_env(
            orchestrator,
            env,
            step=rollout_step,
            policy_step=policy_step,
        )
        log_eval_metrics(
            config,
            metrics,
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

    from wavelet.orchestrator.envs import evaluate_env_async

    for env in envs:
        metrics = await evaluate_env_async(
            orchestrator,
            env,
            step=rollout_step,
            policy_step=policy_step,
        )
        log_eval_metrics(
            config,
            metrics,
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

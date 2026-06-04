from __future__ import annotations

import asyncio
import json
import math
import os
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from wavelet.configs.rl_config import RLEvalEnvConfig
from wavelet.data.rl_dataset import RLExample, load_rl_records
from wavelet.orchestrator.agent_trajectory import TokenSegment, merge_token_segments
from wavelet.orchestrator.advantage import (
    group_reward_advantages,
    length_penalty_cost_for_output,
    output_completion_token_count,
    output_tool_response_token_count,
)
from wavelet.orchestrator.eval_utils import pass_at_k
from wavelet.orchestrator.patches import apply_verifier_openai_patches
from wavelet.orchestrator.rollout_metadata import rollout_task_harness_metadata
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.schedule import (
    rollout_chunk_examples as _rollout_chunk_examples,
)
from wavelet.utils.perf import emit_perf


_ENV_CACHE: dict[tuple[str, str], Any] = {}
_OPENAI_PATCHES_APPLIED = False
_RATE_LIMIT_ERROR_MARKERS = (
    "gousagelimiterror",
    "rate limit",
    "usage limit",
    "too many requests",
)


def _ensure_verifier_openai_patches() -> None:
    global _OPENAI_PATCHES_APPLIED
    if _OPENAI_PATCHES_APPLIED:
        return
    apply_verifier_openai_patches()
    _OPENAI_PATCHES_APPLIED = True


@dataclass(slots=True)
class _PendingVerifierRequest:
    group_id: int
    client_index: int
    rollout_count: int
    off_policy_steps: int = 0


@dataclass(slots=True)
class _VerifierGroupState:
    example: dict[str, Any]
    rollouts_to_schedule: int
    completed_outputs: list[dict[str, Any]] = field(default_factory=list)
    pinned_client_index: int | None = None


def generate_rollouts(
    orchestrator: RLOrchestrator,
    records: list[RLExample],
    _inference_engine,
) -> list[RLExample]:
    try:
        import verifiers as vf
    except ImportError as exc:
        raise ImportError(
            "Verifier rollouts require the 'verifiers' extra. Install with "
            "`uv sync --extra verifiers`."
        ) from exc
    _ensure_verifier_openai_patches()

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
    os.environ.setdefault(config.orchestrator.verifier_api_key_var, "EMPTY")
    clients = [
        vf.ClientConfig(
            client_idx=client_index,
            client_type=config.orchestrator.verifier_client_type,
            api_base_url=base_url,
            api_key_var=config.orchestrator.verifier_api_key_var,
            extra_headers=extra_headers,
        )
        for client_index, (base_url, extra_headers) in enumerate(
            _verifier_client_routes(base_urls, config.inference.vllm.data_parallel_size)
        )
    ]
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
            advantage_epsilon=config.orchestrator.advantage_epsilon,
            normalize_group_advantages=config.orchestrator.normalize_group_advantages,
            length_penalty=config.orchestrator.length_penalty,
            env_name=_env_name(env, fallback=env_id),
        )
    )
    rollout_seconds = perf_counter() - rollout_started_at
    convert_started_at = perf_counter()
    _assign_rollout_advantages(outputs, config)
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

    def __init__(self, orchestrator: RLOrchestrator) -> None:
        try:
            import verifiers as vf
        except ImportError as exc:
            raise ImportError(
                "Verifier rollouts require the 'verifiers' extra. Install with "
                "`uv sync --extra verifiers`."
            ) from exc
        _ensure_verifier_openai_patches()

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
        os.environ.setdefault(config.orchestrator.verifier_api_key_var, "EMPTY")
        base_urls = _verifier_base_urls(config)
        self.clients = [
            vf.ClientConfig(
                client_idx=client_index,
                client_type=config.orchestrator.verifier_client_type,
                api_base_url=base_url,
                api_key_var=config.orchestrator.verifier_api_key_var,
                extra_headers=extra_headers,
            )
            for client_index, (base_url, extra_headers) in enumerate(
                _verifier_client_routes(
                    base_urls,
                    config.inference.vllm.data_parallel_size,
                )
            )
        ]
        if not self.clients:
            raise ValueError("At least one verifier client is required.")
        self.records = load_rl_records(config.data)
        if not self.records:
            raise ValueError("Verifier scheduler requires at least one train record.")
        self.rng = random.Random(config.data.seed)
        self.record_offset = self.rng.randrange(len(self.records))
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
            return max(
                len(self.clients),
                math.ceil(explicit_rollouts / self.rollout_count),
            )
        base_groups = self.target_groups
        pending_chunk_limit = self.config.orchestrator.max_pending_rollout_chunks
        if pending_chunk_limit is not None:
            base_groups = _rollout_chunk_examples(self.config) * pending_chunk_limit
        oversampled_groups = math.ceil(
            base_groups * self.config.orchestrator.oversampling_factor
        )
        return max(
            len(self.clients),
            oversampled_groups,
        )

    @property
    def max_inflight_rollouts(self) -> int:
        return max(len(self.clients), self.max_inflight_groups * self.rollout_count)

    @property
    def inflight_rollout_count(self) -> int:
        return sum(info.rollout_count for info in self.pending.values())

    async def generate_batch(
        self, *, target_groups: int | None = None
    ) -> list[RLExample]:
        started_at = perf_counter()
        target_groups = self.target_groups if target_groups is None else target_groups
        outputs: list[dict[str, Any]] = []
        accepted_groups = 0
        rejected_groups = 0
        completed_groups = 0
        attempt = 0
        max_completed_groups = target_groups * (
            self.config.orchestrator.zero_advantage_max_retries + 1
        )
        if not hasattr(self, "ready_groups"):
            self.ready_groups = []
        self._sync_ready_group_ages()

        try:
            drained_completed, drained_rejected = self._drain_completed_groups_to_ready(
                target_groups=target_groups,
                outputs=outputs,
                accepted_groups=accepted_groups,
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
                    records = [
                        record
                        for output in outputs
                        for record in _records_from_output(output)
                    ]
                    records = _mark_zero_advantage_records_metric_only(
                        records,
                        self.config,
                    )
                    if _has_trainable_rollout_record(records):
                        break
                    attempt += 1
                    if completed_groups >= max_completed_groups:
                        raise RuntimeError(
                            "Verifier scheduler could not produce enough trainable "
                            "rollout groups after "
                            f"{completed_groups} completed group(s): accepted "
                            f"{accepted_groups}, rejected {rejected_groups}. "
                            "Increase orchestrator.zero_advantage_max_retries, "
                            "relax filtering, or check reward/model behavior."
                        )
                    outputs = []
                    accepted_groups = 0

                self._fill_inflight()
                done, _ = await asyncio.wait(
                    self.pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    request = self.pending.pop(task, None)
                    self.pending_clients.pop(task, None)
                    if request is None:
                        continue
                    group = self.groups.get(request.group_id)
                    if group is None:
                        continue
                    group_outputs = _completed_group_outputs(task)
                    if len(group_outputs) < request.rollout_count:
                        if self.requires_group_scoring:
                            group.completed_outputs.clear()
                            group.rollouts_to_schedule = self.rollout_count
                        else:
                            group.rollouts_to_schedule += request.rollout_count - len(
                                group_outputs
                            )
                    group.completed_outputs.extend(group_outputs)
                    if len(group.completed_outputs) < self.rollout_count:
                        continue

                    completed_groups += 1
                    group_outputs = group.completed_outputs
                    self.groups.pop(request.group_id, None)
                    _stamp_env_name(
                        group_outputs,
                        getattr(self, "env_name", "verifier"),
                    )
                    _assign_completed_group_advantages(group_outputs, self.config)
                    if _is_complete_training_group(
                        group_outputs,
                        expected_rollouts=self.rollout_count,
                    ):
                        if accepted_groups < target_groups:
                            outputs.extend(group_outputs)
                            accepted_groups += 1
                        else:
                            self.ready_groups.append(group_outputs)
                            self.ready_group_off_policy_steps.append(0)
                    else:
                        rejected_groups += 1
                    if (
                        completed_groups >= max_completed_groups
                        and accepted_groups < target_groups
                    ):
                        if accepted_groups > 0:
                            break
                        raise RuntimeError(
                            "Verifier scheduler could not produce enough trainable "
                            "rollout groups after "
                            f"{completed_groups} completed group(s): accepted "
                            f"{accepted_groups}, rejected {rejected_groups}. "
                            "Increase orchestrator.zero_advantage_max_retries, "
                            "relax filtering, or check reward/model behavior."
                        )
        except Exception:
            await self.aclose()
            raise

        drained_completed, drained_rejected = self._drain_completed_groups_to_ready(
            target_groups=target_groups,
            outputs=outputs,
            accepted_groups=accepted_groups,
        )
        completed_groups += drained_completed
        rejected_groups += drained_rejected
        self._fill_inflight()
        convert_started_at = perf_counter()
        records = [
            record for output in outputs for record in _records_from_output(output)
        ]
        records = _mark_zero_advantage_records_metric_only(records, self.config)
        convert_seconds = perf_counter() - convert_started_at
        emit_perf(
            "verifier_scheduler",
            attempts=attempt + 1,
            accepted_groups=accepted_groups,
            rejected_groups=rejected_groups,
            completed_groups=completed_groups,
            inflight_rollouts=self.inflight_rollout_count,
            records=len(records),
            convert=convert_seconds,
            total=perf_counter() - started_at,
        )
        return records

    def _drain_completed_groups_to_ready(
        self,
        *,
        target_groups: int,
        outputs: list[dict[str, Any]],
        accepted_groups: int,
    ) -> tuple[int, int]:
        completed_groups = 0
        rejected_groups = 0
        for task in [task for task in self.pending if task.done()]:
            request = self.pending.pop(task, None)
            self.pending_clients.pop(task, None)
            if request is None:
                continue
            group = self.groups.get(request.group_id)
            if group is None:
                continue
            group_outputs = _completed_group_outputs(task)
            if len(group_outputs) < request.rollout_count:
                if self.requires_group_scoring:
                    group.completed_outputs.clear()
                    group.rollouts_to_schedule = self.rollout_count
                else:
                    group.rollouts_to_schedule += request.rollout_count - len(
                        group_outputs
                    )
            group.completed_outputs.extend(group_outputs)
            if len(group.completed_outputs) < self.rollout_count:
                continue

            completed_groups += 1
            group_outputs = group.completed_outputs
            self.groups.pop(request.group_id, None)
            _stamp_env_name(group_outputs, getattr(self, "env_name", "verifier"))
            _assign_completed_group_advantages(group_outputs, self.config)
            if _is_complete_training_group(
                group_outputs,
                expected_rollouts=self.rollout_count,
            ):
                if accepted_groups < target_groups:
                    outputs.extend(group_outputs)
                    accepted_groups += 1
                else:
                    self.ready_groups.append(group_outputs)
                    self.ready_group_off_policy_steps.append(0)
            else:
                rejected_groups += 1
        self.rejected_groups_count = (
            getattr(self, "rejected_groups_count", 0) + rejected_groups
        )
        return completed_groups, rejected_groups

    async def mark_policy_update(self) -> int:
        max_off_policy_steps = self.config.orchestrator.max_off_policy_steps
        cancelled_rollouts = self._age_ready_groups(max_off_policy_steps)
        if not self.pending:
            self.cancelled_rollouts_count += cancelled_rollouts
            return cancelled_rollouts

        stale_group_ids = {
            request.group_id
            for request in self.pending.values()
            if request.off_policy_steps >= max_off_policy_steps
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
            request.off_policy_steps += 1

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        self.cancelled_rollouts_count += cancelled_rollouts
        return cancelled_rollouts

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
            if off_policy_steps >= max_off_policy_steps:
                dropped_rollouts += len(group_outputs)
                continue
            kept_groups.append(group_outputs)
            kept_ages.append(off_policy_steps + 1)
        self.ready_groups = kept_groups
        self.ready_group_off_policy_steps = kept_ages
        return dropped_rollouts

    def _sync_ready_group_ages(self) -> None:
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
        while self.inflight_rollout_count < self.max_inflight_rollouts:
            if not self._schedule_next_rollout():
                break

    def _schedule_next_rollout(self) -> bool:
        remaining_capacity = self.max_inflight_rollouts - self.inflight_rollout_count
        if remaining_capacity <= 0:
            return False

        for group_id, group in self.groups.items():
            if group.rollouts_to_schedule <= 0:
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
                    normalize_group_advantages=(
                        self.config.orchestrator.normalize_group_advantages
                    ),
                    advantage_epsilon=self.config.orchestrator.advantage_epsilon,
                    length_penalty=self.config.orchestrator.length_penalty,
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
        )
        self.pending_clients[task] = client_index

    def _next_record(self) -> RLExample:
        return self.rng.choice(self.records)

    def _least_loaded_client_index(self) -> int:
        counts = [0] * len(self.clients)
        for request in self.pending.values():
            counts[request.client_index] += request.rollout_count
        return min(range(len(self.clients)), key=counts.__getitem__)

    def _sampling_args_for_current_policy(self) -> dict[str, Any]:
        cache_salt = None if self.policy_step is None else str(self.policy_step)
        return _sampling_args(self.config, cache_salt=cache_salt)


def evaluate_env(
    orchestrator: RLOrchestrator,
    env_config: RLEvalEnvConfig,
    *,
    step: int,
    policy_step: int,
) -> dict[str, float]:
    try:
        import verifiers as vf
    except ImportError as exc:
        raise ImportError(
            "Verifier evals require the 'verifiers' extra. Install with "
            "`uv sync --extra verifiers`."
        ) from exc
    _ensure_verifier_openai_patches()

    config = orchestrator.config
    env, _env_cache_hit = _load_cached_env(
        vf,
        env_config.id,
        env_config.args,
        _verifier_extra_env_kwargs(config),
    )
    examples = env.get_eval_dataset(n=env_config.num_examples).to_list()
    base_urls = _verifier_base_urls(config)
    os.environ.setdefault(config.orchestrator.verifier_api_key_var, "EMPTY")
    clients = [
        vf.ClientConfig(
            client_idx=client_index,
            client_type="openai_chat_completions",
            api_base_url=base_url,
            api_key_var=config.orchestrator.verifier_api_key_var,
            extra_headers=extra_headers,
        )
        for client_index, (base_url, extra_headers) in enumerate(
            _verifier_client_routes(base_urls, config.inference.vllm.data_parallel_size)
        )
    ]
    if not clients:
        raise ValueError("At least one verifier eval client is required.")

    started_at = perf_counter()
    outputs = asyncio.run(
        _run_eval_examples(
            vf,
            env,
            examples,
            clients=clients,
            model=config.orchestrator.verifier_model or config.model.name,
            sampling_args=_sampling_args_with_cache_salt(
                env_config.sampling.to_sampling_args(),
                cache_salt=str(policy_step),
            ),
            rollouts_per_example=env_config.rollouts_per_example,
            max_retries=env_config.max_retries,
        )
    )
    elapsed = perf_counter() - started_at
    env_name = env_config.resolved_name
    output_path = config.output_dir / "evals" / f"step-{step:06d}" / f"{env_name}.jsonl"
    _write_eval_rollouts(output_path, outputs)
    metrics = _eval_metrics(
        env_name,
        outputs,
        total_rollouts=len(examples) * env_config.rollouts_per_example,
        elapsed_seconds=elapsed,
        rollouts_per_example=env_config.rollouts_per_example,
    )
    metrics["progress/policy_step"] = float(policy_step)
    metrics["step"] = float(step)
    _append_eval_metrics(config.output_dir / "eval_metrics.jsonl", metrics)
    return metrics


async def evaluate_env_async(
    orchestrator: RLOrchestrator,
    env_config: RLEvalEnvConfig,
    *,
    step: int,
    policy_step: int,
) -> dict[str, float]:
    try:
        import verifiers as vf
    except ImportError as exc:
        raise ImportError(
            "Verifier evals require the 'verifiers' extra. Install with "
            "`uv sync --extra verifiers`."
        ) from exc
    _ensure_verifier_openai_patches()

    config = orchestrator.config
    env, _env_cache_hit = _load_cached_env(
        vf,
        env_config.id,
        env_config.args,
        _verifier_extra_env_kwargs(config),
    )
    examples = env.get_eval_dataset(n=env_config.num_examples).to_list()
    base_urls = _verifier_base_urls(config)
    os.environ.setdefault(config.orchestrator.verifier_api_key_var, "EMPTY")
    clients = [
        vf.ClientConfig(
            client_idx=client_index,
            client_type="openai_chat_completions",
            api_base_url=base_url,
            api_key_var=config.orchestrator.verifier_api_key_var,
            extra_headers=extra_headers,
        )
        for client_index, (base_url, extra_headers) in enumerate(
            _verifier_client_routes(base_urls, config.inference.vllm.data_parallel_size)
        )
    ]
    if not clients:
        raise ValueError("At least one verifier eval client is required.")

    started_at = perf_counter()
    outputs = await _run_eval_examples(
        vf,
        env,
        examples,
        clients=clients,
        model=config.orchestrator.verifier_model or config.model.name,
        sampling_args=_sampling_args_with_cache_salt(
            env_config.sampling.to_sampling_args(),
            cache_salt=str(policy_step),
        ),
        rollouts_per_example=env_config.rollouts_per_example,
        max_retries=env_config.max_retries,
    )
    elapsed = perf_counter() - started_at
    env_name = env_config.resolved_name
    output_path = config.output_dir / "evals" / f"step-{step:06d}" / f"{env_name}.jsonl"
    _write_eval_rollouts(output_path, outputs)
    metrics = _eval_metrics(
        env_name,
        outputs,
        total_rollouts=len(examples) * env_config.rollouts_per_example,
        elapsed_seconds=elapsed,
        rollouts_per_example=env_config.rollouts_per_example,
    )
    metrics["progress/policy_step"] = float(policy_step)
    metrics["step"] = float(step)
    _append_eval_metrics(config.output_dir / "eval_metrics.jsonl", metrics)
    return metrics


async def _run_eval_examples(
    vf,
    env,
    examples: list[dict[str, Any]],
    *,
    clients: list[Any],
    model: str,
    sampling_args: dict[str, Any],
    rollouts_per_example: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    tasks = []
    for example_index, example in enumerate(examples):
        client = clients[example_index % len(clients)]
        for _ in range(rollouts_per_example):
            tasks.append(
                env.run_rollout(
                    vf.RolloutInput(**example),
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                    max_retries=max_retries,
                    state_columns=["trajectory", "sampling_args"],
                )
            )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        outputs.append(dict(result))
    return outputs


def _eval_metrics(
    env_name: str,
    outputs: list[dict[str, Any]],
    *,
    total_rollouts: int,
    elapsed_seconds: float,
    rollouts_per_example: int,
) -> dict[str, float]:
    prefix = f"eval/{env_name}"
    rewards = [float(output["reward"]) for output in outputs if "reward" in output]
    failed = max(total_rollouts - len(rewards), 0)
    metrics = {
        f"{prefix}/failed_rollouts": failed / max(total_rollouts, 1),
        f"{prefix}/time": elapsed_seconds,
    }
    if not outputs:
        return metrics

    completion_lengths = [_completion_len(output) for output in outputs]
    truncations = [bool(output.get("is_truncated")) for output in outputs]
    no_responses = [not bool(output.get("completion")) for output in outputs]
    if rewards:
        metrics[f"{prefix}/avg@{rollouts_per_example}"] = sum(rewards) / len(rewards)
    if completion_lengths:
        metrics[f"{prefix}/completion_len/mean"] = sum(completion_lengths) / len(
            completion_lengths
        )
        metrics[f"{prefix}/completion_len/min"] = float(min(completion_lengths))
        metrics[f"{prefix}/completion_len/max"] = float(max(completion_lengths))
    metrics[f"{prefix}/is_truncated/mean"] = sum(truncations) / len(truncations)
    metrics[f"{prefix}/no_response/mean"] = sum(no_responses) / len(no_responses)
    if rewards and set(rewards).issubset({0.0, 1.0}):
        by_example: dict[str, list[float]] = {}
        for output in outputs:
            if "reward" not in output:
                continue
            by_example.setdefault(str(output.get("example_id")), []).append(
                float(output["reward"])
            )
        pass_metrics: dict[str, list[float]] = {}
        for group_rewards in by_example.values():
            for key, value in pass_at_k(group_rewards).items():
                pass_metrics.setdefault(key, []).append(value)
        for key, values in pass_metrics.items():
            metrics[f"{prefix}/{key}"] = sum(values) / len(values)
    return metrics


def _completion_len(output: dict[str, Any]) -> float:
    trajectory = output.get("trajectory") or []
    if trajectory:
        return float(
            sum(
                len((step.get("tokens") or {}).get("completion_ids") or [])
                for step in trajectory
            )
        )
    completion = output.get("completion") or []
    return float(len(str(completion).split()))


def _write_eval_rollouts(path: Path, outputs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for output in outputs:
            handle.write(json.dumps(output, default=str) + "\n")


def _append_eval_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")


def _verifier_client_routes(
    base_urls: list[str],
    data_parallel_size: int,
) -> list[tuple[str, dict[str, str]]]:
    routes: list[tuple[str, dict[str, str]]] = []
    for base_url in base_urls:
        for dp_rank in range(data_parallel_size):
            headers = (
                {"X-data-parallel-rank": str(dp_rank)} if data_parallel_size > 1 else {}
            )
            routes.append((base_url, headers))
    return routes


def _verifier_base_urls(config) -> list[str]:
    base_urls = config.orchestrator.verifier_base_url
    if base_urls is None:
        http = config.inference.http
        ports = http.ports or [http.port]
        return [f"http://{http.host}:{port}/v1" for port in ports]
    if isinstance(base_urls, str):
        return [base_urls]
    return list(base_urls)


def _verifier_model(config, inference_engine: Any | None = None) -> str:
    policy_model_name = getattr(inference_engine, "policy_model_name", None)
    if isinstance(policy_model_name, str) and policy_model_name:
        return policy_model_name
    return config.orchestrator.verifier_model or config.model.name


def _verifier_extra_env_kwargs(config) -> dict[str, Any]:
    rollout_seq_len = config.inference.vllm.max_model_len or config.data.seq_len
    kwargs: dict[str, Any] = {
        "max_seq_len": rollout_seq_len,
        "max_total_completion_tokens": (
            config.orchestrator.verifier_max_total_completion_tokens
        ),
    }
    if config.orchestrator.verifier_timeout_seconds is not None:
        kwargs["timeout_seconds"] = config.orchestrator.verifier_timeout_seconds
    return kwargs


def _load_cached_env(
    vf,
    env_id: str,
    env_args: dict[str, Any],
    extra_env_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, bool]:
    extra_env_kwargs = extra_env_kwargs or {}
    cache_key = (
        env_id,
        json.dumps(env_args, sort_keys=True, default=str),
        json.dumps(extra_env_kwargs, sort_keys=True, default=str),
    )
    cached = _ENV_CACHE.get(cache_key)
    if cached is not None:
        return cached, True
    env = vf.load_environment(env_id, **env_args)
    if extra_env_kwargs:
        set_kwargs = getattr(env, "set_kwargs", None)
        if callable(set_kwargs):
            set_kwargs(**extra_env_kwargs)
        else:
            for key, value in extra_env_kwargs.items():
                setattr(env, key, value)
    _patch_env_response_messages(vf, env)
    _ENV_CACHE[cache_key] = env
    return env, False


async def _run_all(
    vf,
    env,
    records: list[RLExample],
    *,
    clients: list[Any],
    model: str,
    sampling_args: dict[str, Any],
    rollout_count: int,
    max_retries: int,
    target_groups: int | None,
    filter_zero_advantage: bool,
    advantage_epsilon: float,
    normalize_group_advantages: bool,
    length_penalty: object | None,
    env_name: str = "verifier",
) -> list[dict[str, Any]]:
    if not clients:
        raise ValueError("At least one verifier client is required.")
    if target_groups is None or len(records) <= target_groups:
        if getattr(env, "requires_group_scoring", False):
            tasks = []
            for record_index, record in enumerate(records):
                example = _verifier_example(record)
                client = clients[record_index % len(clients)]
                tasks.append(
                    _run_group(
                        vf,
                        env,
                        example,
                        client=client,
                        model=model,
                        sampling_args=sampling_args,
                        rollout_count=rollout_count,
                        max_retries=max_retries,
                        normalize_group_advantages=normalize_group_advantages,
                        advantage_epsilon=advantage_epsilon,
                        length_penalty=length_penalty,
                    )
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            outputs = [
                output
                for result in results
                if not isinstance(result, Exception)
                for output in result
            ]
            _stamp_env_name(outputs, env_name)
            return outputs

        tasks = []
        for record_index, record in enumerate(records):
            example = _verifier_example(record)
            client = clients[record_index % len(clients)]
            for _ in range(rollout_count):
                tasks.append(
                    env.run_rollout(
                        vf.RolloutInput(**example),
                        client=client,
                        model=model,
                        sampling_args=sampling_args,
                        max_retries=max_retries,
                        state_columns=["trajectory", "sampling_args"],
                    )
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = _successful_rollout_outputs(results, require_trainable=False)
        _stamp_env_name(outputs, env_name)
        return outputs

    group_tasks: list[asyncio.Task[list[dict[str, Any]]]] = []
    for record_index, record in enumerate(records):
        example = _verifier_example(record)
        client = clients[record_index % len(clients)]
        task = asyncio.create_task(
            _run_group(
                vf,
                env,
                example,
                client=client,
                model=model,
                sampling_args=sampling_args,
                rollout_count=rollout_count,
                max_retries=max_retries,
                normalize_group_advantages=normalize_group_advantages,
                advantage_epsilon=advantage_epsilon,
                length_penalty=length_penalty,
            )
        )
        group_tasks.append(task)

    outputs: list[dict[str, Any]] = []
    accepted_groups = 0
    pending = set(group_tasks)
    try:
        while pending and accepted_groups < target_groups:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                group_outputs = _completed_group_outputs(task)
                _stamp_env_name(group_outputs, env_name)
                if _is_usable_training_group(
                    group_outputs,
                    expected_rollouts=rollout_count,
                    filter_zero_advantage=filter_zero_advantage,
                    advantage_epsilon=advantage_epsilon,
                ):
                    outputs.extend(group_outputs)
                    accepted_groups += 1
                    if accepted_groups >= target_groups:
                        break
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    return outputs


async def _run_group(
    vf,
    env,
    example: dict[str, Any],
    *,
    client: Any,
    model: str,
    sampling_args: dict[str, Any],
    rollout_count: int,
    max_retries: int,
    normalize_group_advantages: bool,
    advantage_epsilon: float,
    length_penalty: object | None,
) -> list[dict[str, Any]]:
    if getattr(env, "requires_group_scoring", False):
        run_group = getattr(env, "run_group", None)
        if not callable(run_group):
            raise ValueError(
                "Verifier environment requires group scoring but does not expose "
                "run_group()."
            )
        group_inputs = [vf.RolloutInput(**example) for _ in range(rollout_count)]
        result = await run_group(
            group_inputs,
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_retries=max_retries,
            state_columns=["trajectory", "sampling_args"],
        )
        outputs = _successful_rollout_outputs(list(result))
        _assign_group_advantages(
            outputs,
            normalize_group_advantages=normalize_group_advantages,
            advantage_epsilon=advantage_epsilon,
            length_penalty=length_penalty,
        )
        return outputs

    tasks = [
        env.run_rollout(
            vf.RolloutInput(**example),
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_retries=max_retries,
            state_columns=["trajectory", "sampling_args"],
        )
        for _ in range(rollout_count)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs = _successful_rollout_outputs(results)
    _assign_group_advantages(
        outputs,
        normalize_group_advantages=normalize_group_advantages,
        advantage_epsilon=advantage_epsilon,
        length_penalty=length_penalty,
    )
    return outputs


def _env_name(env: Any, *, fallback: str) -> str:
    name = getattr(env, "name", None)
    if isinstance(name, str) and name:
        return name
    return fallback


def _stamp_env_name(outputs: list[dict[str, Any]], env_name: str) -> None:
    for output in outputs:
        output.setdefault("env_name", env_name)


async def _run_single_rollout(
    vf,
    env,
    example: dict[str, Any],
    *,
    client: Any,
    model: str,
    sampling_args: dict[str, Any],
    max_retries: int,
) -> list[dict[str, Any]]:
    result = await env.run_rollout(
        vf.RolloutInput(**example),
        client=client,
        model=model,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=["trajectory", "sampling_args"],
    )
    return _successful_rollout_outputs([result])


def _successful_rollout_outputs(
    results: list[Any],
    *,
    require_trainable: bool = True,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            _raise_if_external_rate_limit(result)
            continue
        try:
            output = dict(result)
        except (TypeError, ValueError):
            continue
        error = output.get("error")
        if error is not None:
            _raise_if_external_rate_limit(error)
            continue
        if "reward" not in output:
            continue
        if require_trainable and not _has_trainable_trajectory(output):
            continue
        outputs.append(output)
    return outputs


def _assign_completed_group_advantages(outputs: list[dict[str, Any]], config) -> None:
    if config.orchestrator.advantage_mode == "reward":
        for output in outputs:
            output["advantage"] = float(output["reward"])
        return
    if config.orchestrator.advantage_mode != "group_reward":
        return
    _assign_group_advantages(
        outputs,
        normalize_group_advantages=config.orchestrator.normalize_group_advantages,
        advantage_epsilon=config.orchestrator.advantage_epsilon,
        length_penalty=config.orchestrator.length_penalty,
    )


def _completed_group_outputs(
    task: asyncio.Task[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    try:
        return task.result()
    except Exception as exc:
        _raise_if_external_rate_limit(exc)
        return []


def _raise_if_external_rate_limit(error: Any) -> None:
    text = str(error)
    lowered = text.lower()
    if not any(marker in lowered for marker in _RATE_LIMIT_ERROR_MARKERS):
        return
    if "429" not in lowered and "limit" not in lowered:
        return
    raise RuntimeError(
        "Verifier reward provider rate limit exceeded. Retry after the provider "
        "limit resets, enable paid usage, or reduce judge usage/concurrency. "
        f"Original error: {_truncate_error(text)}"
    )


def _truncate_error(text: str, *, max_chars: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _assign_group_advantages(
    outputs: list[dict[str, Any]],
    *,
    normalize_group_advantages: bool,
    advantage_epsilon: float,
    length_penalty: object | None,
) -> None:
    if not outputs:
        return
    rewards = [float(output["reward"]) for output in outputs]
    costs = (
        [length_penalty_cost_for_output(output, length_penalty) for output in outputs]
        if length_penalty is not None
        else None
    )
    advantages = group_reward_advantages(
        rewards,
        costs=costs,
        normalize=normalize_group_advantages,
        epsilon=advantage_epsilon,
    )
    for output, advantage in zip(outputs, advantages, strict=True):
        output["advantage"] = advantage


def _has_trainable_advantage(
    outputs: list[dict[str, Any]],
    *,
    filter_zero_advantage: bool,
    advantage_epsilon: float,
) -> bool:
    if not filter_zero_advantage:
        return True
    return any(
        output.get("advantage") is not None
        and abs(float(output["advantage"])) > advantage_epsilon
        for output in outputs
    )


def _is_usable_training_group(
    outputs: list[dict[str, Any]],
    *,
    expected_rollouts: int,
    filter_zero_advantage: bool,
    advantage_epsilon: float,
) -> bool:
    if len(outputs) != expected_rollouts:
        return False
    if not all(_has_trainable_trajectory(output) for output in outputs):
        return False
    return _has_trainable_advantage(
        outputs,
        filter_zero_advantage=filter_zero_advantage,
        advantage_epsilon=advantage_epsilon,
    )


def _is_complete_training_group(
    outputs: list[dict[str, Any]],
    *,
    expected_rollouts: int,
) -> bool:
    return len(outputs) == expected_rollouts and all(
        _has_trainable_trajectory(output) for output in outputs
    )


def _has_trainable_rollout_record(records: list[RLExample]) -> bool:
    return any(
        record.loss_mask is not None
        and any(bool(item) for item in record.loss_mask)
        and not (record.metadata or {}).get("_wavelet_filtered_rollout")
        for record in records
    )


def _mark_zero_advantage_records_metric_only(
    records: list[RLExample],
    config,
) -> list[RLExample]:
    if not config.orchestrator.filter_zero_advantage:
        return records
    if config.orchestrator.advantage_mode != "group_reward":
        return records
    epsilon = config.orchestrator.advantage_epsilon
    marked: list[RLExample] = []
    for record in records:
        if record.advantage is not None and abs(float(record.advantage)) > epsilon:
            marked.append(record)
            continue
        metadata = dict(record.metadata or {})
        metadata["_wavelet_filtered_rollout"] = True
        loss_mask = (
            [False] * len(record.loss_mask)
            if record.loss_mask is not None
            else record.loss_mask
        )
        marked.append(
            replace(
                record,
                advantage=0.0,
                loss_mask=loss_mask,
                inference_logprobs=[],
                teacher_logprobs=[] if record.teacher_logprobs is not None else None,
                temperatures=[],
                metadata=metadata,
            )
        )
    return marked


def _has_trainable_trajectory(output: dict[str, Any]) -> bool:
    trajectory = output.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return False
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        tokens = step.get("tokens")
        if not isinstance(tokens, dict):
            continue
        completion_mask = tokens.get("completion_mask")
        if isinstance(completion_mask, list) and any(
            bool(item) for item in completion_mask
        ):
            return True
    return False


def _patch_env_response_messages(vf, env) -> None:
    env_response = getattr(env, "env_response", None)
    if not callable(env_response):
        return

    async def wrapped_env_response(*args, **kwargs):
        response = await env_response(*args, **kwargs)
        return [_coerce_vf_message(vf, message) for message in response]

    setattr(env, "env_response", wrapped_env_response)


def _coerce_vf_message(vf, message: Any):
    if not isinstance(message, dict):
        return message
    role = message.get("role")
    content = message.get("content", "")
    if role == "system":
        return vf.SystemMessage(content=content)
    if role == "user":
        return vf.UserMessage(content=content)
    if role == "assistant":
        return vf.AssistantMessage(content=content)
    if role == "tool":
        return vf.ToolMessage(
            tool_call_id=str(message.get("tool_call_id", "")),
            content=content,
        )
    return message


def _verifier_example(record: RLExample) -> dict[str, Any]:
    metadata = record.metadata or {}
    example = metadata.get("verifier_example")
    if isinstance(example, dict):
        return example
    example_id = metadata.get("example_id")
    return {
        "example_id": example_id if example_id is not None else 0,
        "prompt": record.prompt,
    }


def _sampling_args(config, *, cache_salt: str | None = None) -> dict[str, Any]:
    sampling = config.inference.sampling
    args: dict[str, Any] = {
        "temperature": sampling.temperature if sampling.do_sample else 0.0,
        "top_p": sampling.top_p,
        "logprobs": True,
    }
    if sampling.max_completion_tokens is not None:
        args["max_completion_tokens"] = sampling.max_completion_tokens
    extra_body: dict[str, Any] = dict(sampling.extra_body)
    extra_body["return_token_ids"] = True
    if cache_salt is not None:
        extra_body["cache_salt"] = cache_salt
    if sampling.top_k != 0:
        extra_body["top_k"] = sampling.top_k
    extra_body["min_p"] = sampling.min_p
    if sampling.min_tokens > 0:
        extra_body["min_tokens"] = sampling.min_tokens
    if sampling.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = sampling.repetition_penalty
    if sampling.seed is not None:
        args["seed"] = sampling.seed
    args["extra_body"] = extra_body
    return args


def _sampling_args_with_cache_salt(
    sampling_args: dict[str, Any],
    *,
    cache_salt: str | None,
) -> dict[str, Any]:
    if cache_salt is None:
        return sampling_args
    copied = dict(sampling_args)
    extra_body = dict(copied.get("extra_body") or {})
    extra_body["cache_salt"] = cache_salt
    copied["extra_body"] = extra_body
    return copied


def _assign_rollout_advantages(outputs: list[dict[str, Any]], config) -> None:
    if config.orchestrator.advantage_mode == "reward":
        for output in outputs:
            output["advantage"] = float(output["reward"])
        return
    if config.orchestrator.advantage_mode != "group_reward":
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for output in outputs:
        grouped.setdefault(_output_group_key(output), []).append(output)
    for group in grouped.values():
        rewards = [float(output["reward"]) for output in group]
        penalty = config.orchestrator.length_penalty
        costs = (
            [length_penalty_cost_for_output(output, penalty) for output in group]
            if penalty is not None
            else None
        )
        advantages = group_reward_advantages(
            rewards,
            costs=costs,
            normalize=config.orchestrator.normalize_group_advantages,
            epsilon=config.orchestrator.advantage_epsilon,
        )
        for output, advantage in zip(group, advantages, strict=True):
            output["advantage"] = advantage


def _records_from_output(output: dict[str, Any]) -> list[RLExample]:
    temperature = float((output.get("sampling_args") or {}).get("temperature", 1.0))
    group_key = _output_group_key(output)
    records: list[RLExample] = []
    samples = _interleave_output(output, temperature)
    for sample_index, sample in enumerate(samples):
        trainable_indexes = [
            index for index, trainable in enumerate(sample["loss_mask"]) if trainable
        ]
        if not trainable_indexes:
            continue
        trajectory = output.get("trajectory") or []
        first_step = trajectory[0] if trajectory else {}
        last_step = trajectory[-1] if trajectory else {}
        inference_logprobs = [
            float(sample["inference_logprobs"][index]) for index in trainable_indexes
        ]
        temperatures = [
            float(sample["temperatures"][index]) for index in trainable_indexes
        ]
        records.append(
            RLExample(
                prompt=_mask_prompt_history(first_step.get("prompt") or []),
                completion=_messages(last_step.get("completion") or []),
                target_completion=None,
                input_ids=sample["input_ids"],
                target_ids=sample["target_ids"],
                loss_mask=sample["loss_mask"],
                advantage=float(output["advantage"])
                if output.get("advantage") is not None
                else None,
                reward=float(output["reward"]),
                inference_logprobs=inference_logprobs,
                temperatures=temperatures,
                metadata={
                    "group_key": group_key,
                    "rollout_key": f"{group_key}:{sample_index}",
                    "stop_condition": output.get("stop_condition"),
                    "is_truncated": output.get("is_truncated"),
                    "completion_token_count": output_completion_token_count(output),
                    "tool_response_token_count": output_tool_response_token_count(
                        output
                    ),
                    "turn_count": len(output.get("trajectory") or []),
                    "_wavelet_rollout_count": 1 if sample_index == 0 else 0,
                    **rollout_task_harness_metadata(
                        output,
                        group_key=group_key,
                        sample_index=sample_index,
                    ),
                },
                source=str(output.get("env_name") or output.get("task") or "verifier"),
            )
        )
    return records


def _output_group_key(output: dict[str, Any]) -> str:
    env_name = str(output.get("env_name") or output.get("task") or "verifier")
    example_id = str(output.get("example_id", "unknown"))
    return json.dumps(
        {"env_name": env_name, "example_id": example_id},
        sort_keys=True,
        separators=(",", ":"),
    )


def _interleave_output(
    output: dict[str, Any],
    temperature: float,
) -> list[dict[str, list[Any]]]:
    trajectory = output.get("trajectory") or []
    if not trajectory:
        return []
    has_error = output.get("error") is not None
    segments = [
        _step_token_segment(output, step, index)
        for index, step in enumerate(trajectory)
    ]
    return [
        sample.as_dict()
        for sample in merge_token_segments(
            segments,
            temperature=temperature,
            mask_outputs=has_error,
        )
    ]


def _step_token_segment(
    output: dict[str, Any], step: dict[str, Any], index: int
) -> TokenSegment:
    tokens = step.get("tokens")
    if tokens is None:
        raise ValueError(
            f"Verifier rollout for example {output.get('example_id')} step {index} "
            "is missing token data."
        )
    return TokenSegment(
        prompt_ids=[int(token_id) for token_id in tokens["prompt_ids"]],
        prompt_loss_mask=[bool(value) for value in tokens["prompt_mask"]],
        output_ids=[int(token_id) for token_id in tokens["completion_ids"]],
        output_loss_mask=[bool(value) for value in tokens["completion_mask"]],
        output_logprobs=[float(value) for value in tokens["completion_logprobs"]],
        turn_id=str(step.get("turn_id", index)),
    )


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(message) for message in messages]


def _mask_prompt_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    masked = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant":
            item["step_loss_mask"] = 0
        masked.append(item)
    return masked

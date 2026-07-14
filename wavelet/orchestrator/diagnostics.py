from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample, load_rl_records
from wavelet.orchestrator.metrics import RolloutMetricInputs, rollout_metrics
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.schedule import (
    chunks_per_step,
    required_policy_step,
    rollout_chunk_examples,
    target_steps,
)


@dataclass(frozen=True)
class OrchestratorProbe:
    timings: dict[str, float]
    records_available: int
    records_selected: int
    records_scored: int
    records_trainable: int
    rollouts_per_example: int
    examples_per_step: int | None
    metrics: dict[str, float]
    output_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def orchestrator_debug_state(config: RLConfig) -> dict[str, Any]:
    schedule: dict[str, Any] = {
        "target_steps": target_steps(config),
        "examples_per_step": config.orchestrator.examples_per_step,
        "rollouts_per_example": config.orchestrator.rollouts_per_example,
        "max_async_level": config.orchestrator.max_async_level,
        "max_off_policy_steps": config.orchestrator.max_off_policy_steps,
        "required_policy_step_at_rollout_0": required_policy_step(config, 0),
        "required_policy_step_at_rollout_1": required_policy_step(config, 1),
    }
    if config.orchestrator.examples_per_step is not None:
        schedule["rollout_chunk_examples"] = rollout_chunk_examples(config)
        schedule["chunks_per_step"] = chunks_per_step(config)
    return {
        "algo": config.algo.model_dump(mode="json", exclude_none=True),
        "data": {
            "source": config.data.source,
            "path": str(config.data.path),
            "seed": config.data.seed,
            "seq_len": config.data.seq_len,
            "batch_size": config.data.batch_size,
            "micro_batch_size": config.data.micro_batch_size,
        },
        "orchestrator": {
            "enabled": config.orchestrator.enabled,
            "custom_rollout_function": config.orchestrator.custom_rollout_function,
            "verifier_env_id": config.orchestrator.verifier_env_id,
            "verifier_model": config.orchestrator.verifier_model,
            "verifier_client_type": config.orchestrator.verifier_client_type,
            "filter_zero_advantage": config.orchestrator.filter_zero_advantage,
            "zero_advantage_max_retries": config.orchestrator.zero_advantage_max_retries,
            "oversampling_factor": config.orchestrator.oversampling_factor,
        },
        "reward": config.reward.model_dump(mode="json"),
        "schedule": schedule,
        "transport": config.transport.model_dump(mode="json", exclude_none=True),
        "output_dir": str(config.output_dir),
    }


def sample_orchestrator_records(
    config: RLConfig,
    *,
    step: int | None,
    retry: int = 0,
) -> dict[str, Any]:
    orchestrator = RLOrchestrator(config)
    started_at = time.perf_counter()
    all_records = load_rl_records(config.data)
    load_seconds = time.perf_counter() - started_at
    started_at = time.perf_counter()
    records = orchestrator._select_step_records(  # noqa: SLF001
        all_records,
        seed=orchestrator._step_seed(step=step, retry=retry),  # noqa: SLF001
    )
    select_seconds = time.perf_counter() - started_at
    seconds = load_seconds + select_seconds
    return {
        "records_available": len(all_records),
        "records": len(records),
        "timings": {
            "load_records": load_seconds,
            "select_records": select_seconds,
            "total": seconds,
        },
        "records_per_second": len(records) / max(seconds, 1e-9),
        "sample": [_record_summary(record) for record in records[:5]],
    }


def probe_orchestrator(
    config: RLConfig,
    *,
    step: int | None,
    retry: int = 0,
    inference_engine: Any = None,
    write: bool = False,
) -> OrchestratorProbe:
    orchestrator = RLOrchestrator(config)
    timings: dict[str, float] = {}

    started_at = time.perf_counter()
    all_records = load_rl_records(config.data)
    timings["load_records"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    selected_records = orchestrator._select_step_records(  # noqa: SLF001
        all_records,
        seed=orchestrator._step_seed(step=step, retry=retry),  # noqa: SLF001
    )
    timings["select_records"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    scored_records = orchestrator._generate_and_score(  # noqa: SLF001
        selected_records,
        inference_engine=inference_engine,
    )
    timings["generate_score"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    trainable_records = orchestrator._filter_zero_advantage_records(scored_records)  # noqa: SLF001
    timings["filter_zero_advantage"] = time.perf_counter() - started_at

    output_path = None
    if write:
        started_at = time.perf_counter()
        output_path = str(orchestrator._write_records(trainable_records, step=step))  # noqa: SLF001
        timings["write"] = time.perf_counter() - started_at

    timings["total"] = sum(timings.values())
    rows = [orchestrator._serialize_record(record) for record in trainable_records]  # noqa: SLF001
    metrics = rollout_metrics(
        RolloutMetricInputs(
            rows=rows,
            rollouts_per_example=config.orchestrator.rollouts_per_example,
            step=step or 0,
            timings=timings,
        )
    )
    return OrchestratorProbe(
        timings=timings,
        records_available=len(all_records),
        records_selected=len(selected_records),
        records_scored=len(scored_records),
        records_trainable=len(trainable_records),
        rollouts_per_example=config.orchestrator.rollouts_per_example,
        examples_per_step=config.orchestrator.examples_per_step,
        metrics=metrics,
        output_path=output_path,
    )


def with_orchestrator_limits(
    config: RLConfig,
    *,
    examples: int | None,
    rollouts: int | None,
) -> RLConfig:
    updates: dict[str, Any] = {}
    if examples is not None:
        updates["examples_per_step"] = examples
    if rollouts is not None:
        updates["rollouts_per_example"] = rollouts
    if not updates:
        return config
    return config.model_copy(
        update={
            "orchestrator": config.orchestrator.model_copy(update=updates),
        }
    )


def _record_summary(record: RLExample) -> dict[str, Any]:
    return {
        "source": record.source,
        "prompt_turns": len(record.prompt),
        "completion_turns": len(record.completion),
        "reward": record.reward,
        "advantage": record.advantage,
        "has_input_ids": record.input_ids is not None,
        "trainable_tokens": sum(record.loss_mask or []),
        "metadata": record.metadata or {},
    }

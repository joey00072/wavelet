"""Verifier environments, clients, rollout materialization, and evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from wavelet.configs.rl_config import RLAlgorithmConfig, RLEvalEnvConfig
from wavelet.data.rl import RLExample
from wavelet.orchestrator.advantage import (
    output_completion_token_count,
    output_tool_response_token_count,
)
from wavelet.orchestrator.agent_trajectory import TokenSegment, merge_token_segments
from wavelet.orchestrator.algorithms import (
    algorithm_epsilon,
    algorithm_scope,
    build_algorithm,
    score_algorithm_records,
    uses_group_advantages,
)
from wavelet.orchestrator.eval_utils import pass_at_k
from wavelet.orchestrator.patches import apply_verifier_openai_patches
from wavelet.orchestrator.rollout_metadata import rollout_task_harness_metadata
from wavelet.orchestrator.rollouts import RLOrchestrator

_ENV_CACHE: dict[tuple[str, str], Any] = {}
logger = logging.getLogger(__name__)


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


def _load_verifiers(feature: str):
    try:
        import verifiers as vf
    except ImportError as exc:
        raise ImportError(
            f"Verifier {feature} require the 'verifiers' extra. Install with "
            "`uv sync --extra verifiers`."
        ) from exc
    _ensure_verifier_openai_patches()
    return vf


def evaluate_env(
    orchestrator: RLOrchestrator,
    env_config: RLEvalEnvConfig,
    *,
    step: int,
    policy_step: int,
) -> dict[str, float]:
    return asyncio.run(
        _evaluate_env_async(
            orchestrator,
            env_config,
            step=step,
            policy_step=policy_step,
        )
    )


async def evaluate_env_async(
    orchestrator: RLOrchestrator,
    env_config: RLEvalEnvConfig,
    *,
    step: int,
    policy_step: int,
) -> dict[str, float]:
    return await _evaluate_env_async(
        orchestrator,
        env_config,
        step=step,
        policy_step=policy_step,
    )


async def _evaluate_env_async(
    orchestrator: RLOrchestrator,
    env_config: RLEvalEnvConfig,
    *,
    step: int,
    policy_step: int,
) -> dict[str, float]:
    vf = _load_verifiers("evals")

    config = orchestrator.config
    env, _env_cache_hit = _load_cached_env(
        vf,
        env_config.id,
        env_config.args,
        _verifier_extra_env_kwargs(config),
    )
    examples = env.get_eval_dataset(n=env_config.num_examples).to_list()
    clients = _verifier_clients(
        vf,
        config,
        client_type="openai_chat_completions",
        client_label="verifier eval",
    )

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
    _prune_eval_rollout_sets(
        config.output_dir / "evals",
        keep_last=config.eval.keep_last_rollout_sets,
    )
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
    example_ids: list[str] = []
    for example_index, example in enumerate(examples):
        client = clients[example_index % len(clients)]
        for _ in range(rollouts_per_example):
            example_ids.append(
                str(example.get("example_id", example.get("id", example_index)))
            )
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
    for example_id, result in zip(example_ids, results, strict=True):
        if isinstance(result, Exception):
            _raise_if_external_rate_limit(result)
            outputs.append(
                {
                    "example_id": example_id,
                    "error": _truncate_error(str(result)),
                    "completion": [],
                }
            )
            continue
        try:
            output = dict(result)
        except (TypeError, ValueError) as exc:
            outputs.append(
                {
                    "example_id": example_id,
                    "error": f"Invalid verifier result: {exc}",
                    "completion": [],
                }
            )
            continue
        output.setdefault("example_id", example_id)
        outputs.append(output)
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

    if total_rollouts > 0:
        metrics[f"{prefix}/avg@{rollouts_per_example}"] = sum(rewards) / total_rollouts
    if rewards:
        metrics[f"{prefix}/effective/avg@{rollouts_per_example}"] = sum(rewards) / len(
            rewards
        )
    if not outputs:
        return metrics

    completion_lengths = [_completion_len(output) for output in outputs]
    truncations = [bool(output.get("is_truncated")) for output in outputs]
    no_responses = [not bool(output.get("completion")) for output in outputs]
    if completion_lengths:
        metrics[f"{prefix}/completion_len/mean"] = sum(completion_lengths) / len(
            completion_lengths
        )
        metrics[f"{prefix}/completion_len/min"] = float(min(completion_lengths))
        metrics[f"{prefix}/completion_len/max"] = float(max(completion_lengths))
    metrics[f"{prefix}/is_truncated/mean"] = sum(truncations) / len(truncations)
    metrics[f"{prefix}/no_response/mean"] = sum(no_responses) / len(no_responses)
    if set(rewards).issubset({0.0, 1.0}):
        by_example: dict[str, list[float]] = {}
        for output in outputs:
            by_example.setdefault(str(output.get("example_id")), []).append(
                float(output.get("reward", 0.0))
            )
        pass_metrics: dict[str, list[float]] = {}
        for group_rewards in by_example.values():
            for key, value in pass_at_k(group_rewards).items():
                pass_metrics.setdefault(key, []).append(value)
        for key, values in pass_metrics.items():
            metrics[f"{prefix}/{key}"] = sum(values) / len(values)
        effective_by_example: dict[str, list[float]] = {}
        for output in outputs:
            if "reward" not in output:
                continue
            effective_by_example.setdefault(str(output.get("example_id")), []).append(
                float(output["reward"])
            )
        effective_pass_metrics: dict[str, list[float]] = {}
        for group_rewards in effective_by_example.values():
            for key, value in pass_at_k(group_rewards).items():
                effective_pass_metrics.setdefault(key, []).append(value)
        for key, values in effective_pass_metrics.items():
            metrics[f"{prefix}/effective/{key}"] = sum(values) / len(values)
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


def _prune_eval_rollout_sets(eval_dir: Path, *, keep_last: int | None) -> list[Path]:
    """Retain only the newest evaluation rollout directories."""
    if keep_last is None or not eval_dir.exists():
        return []

    step_dirs: list[tuple[int, Path]] = []
    for candidate in eval_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith("step-"):
            continue
        try:
            step = int(candidate.name.removeprefix("step-"))
        except ValueError:
            continue
        step_dirs.append((step, candidate))

    removed: list[Path] = []
    for _, path in sorted(step_dirs)[:-keep_last]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


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


def _verifier_clients(
    vf,
    config,
    *,
    base_urls: list[str] | None = None,
    client_type: str | None = None,
    client_label: str = "verifier",
) -> list[Any]:
    os.environ.setdefault(config.orchestrator.verifier_api_key_var, "EMPTY")
    routes = _verifier_client_routes(
        base_urls or _verifier_base_urls(config),
        config.inference.vllm.data_parallel_size,
    )
    clients = [
        vf.ClientConfig(
            client_idx=index,
            client_type=client_type or config.orchestrator.verifier_client_type,
            api_base_url=base_url,
            api_key_var=config.orchestrator.verifier_api_key_var,
            extra_headers=headers,
        )
        for index, (base_url, headers) in enumerate(routes)
    ]
    if not clients:
        raise ValueError(f"At least one {client_label} client is required.")
    return clients


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
    algorithm_config: RLAlgorithmConfig,
    env_name: str = "verifier",
) -> list[dict[str, Any]]:
    if not clients:
        raise ValueError("At least one verifier client is required.")
    if target_groups is None or len(records) <= target_groups:
        return await _run_complete_record_set(
            vf,
            env,
            records,
            clients=clients,
            model=model,
            sampling_args=sampling_args,
            rollout_count=rollout_count,
            max_retries=max_retries,
            algorithm_config=algorithm_config,
            env_name=env_name,
        )
    return await _run_until_target_groups(
        vf,
        env,
        records,
        clients=clients,
        model=model,
        sampling_args=sampling_args,
        rollout_count=rollout_count,
        max_retries=max_retries,
        target_groups=target_groups,
        filter_zero_advantage=filter_zero_advantage,
        advantage_epsilon=advantage_epsilon,
        algorithm_config=algorithm_config,
        env_name=env_name,
    )


async def _run_complete_record_set(
    vf,
    env,
    records: list[RLExample],
    *,
    clients: list[Any],
    model: str,
    sampling_args: dict[str, Any],
    rollout_count: int,
    max_retries: int,
    algorithm_config: RLAlgorithmConfig,
    env_name: str,
) -> list[dict[str, Any]]:
    if getattr(env, "requires_group_scoring", False):
        tasks = [
            _run_group(
                vf,
                env,
                _verifier_example(record),
                client=clients[index % len(clients)],
                model=model,
                sampling_args=sampling_args,
                rollout_count=rollout_count,
                max_retries=max_retries,
                algorithm_config=algorithm_config,
            )
            for index, record in enumerate(records)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = [
            output
            for result in results
            if not isinstance(result, Exception)
            for output in result
        ]
    else:
        tasks = [
            env.run_rollout(
                vf.RolloutInput(**_verifier_example(record)),
                client=clients[index % len(clients)],
                model=model,
                sampling_args=sampling_args,
                max_retries=max_retries,
                state_columns=["trajectory", "sampling_args"],
            )
            for index, record in enumerate(records)
            for _ in range(rollout_count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = _successful_rollout_outputs(results, require_trainable=False)
    _stamp_env_name(outputs, env_name)
    return outputs


async def _run_until_target_groups(
    vf,
    env,
    records: list[RLExample],
    *,
    clients: list[Any],
    model: str,
    sampling_args: dict[str, Any],
    rollout_count: int,
    max_retries: int,
    target_groups: int,
    filter_zero_advantage: bool,
    advantage_epsilon: float,
    algorithm_config: RLAlgorithmConfig,
    env_name: str,
) -> list[dict[str, Any]]:
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
                algorithm_config=algorithm_config,
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
    algorithm_config: RLAlgorithmConfig,
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
        _assign_group_advantages(outputs, algorithm_config=algorithm_config)
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
    _assign_group_advantages(outputs, algorithm_config=algorithm_config)
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
            logger.warning("Verifier rollout failed: %r", result)
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
    _assign_group_advantages(outputs, algorithm_config=config.algo)


def _completed_group_outputs(
    task: asyncio.Task[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    try:
        return task.result()
    except Exception as exc:
        _raise_if_external_rate_limit(exc)
        logger.warning("Verifier group rollout failed: %s", exc, exc_info=True)
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
    algorithm_config: RLAlgorithmConfig,
) -> None:
    if not outputs:
        return
    algorithm = build_algorithm(algorithm_config)
    records = [_algorithm_record_from_output(output) for output in outputs]
    scored = score_algorithm_records(
        algorithm,
        records,
        scope=algorithm_scope(algorithm_config),
    )
    for output, record in zip(outputs, scored, strict=True):
        output["advantage"] = record.advantage


def _algorithm_record_from_output(output: dict[str, Any]) -> RLExample:
    return RLExample(
        prompt=[],
        completion=[],
        advantage=(
            float(output["advantage"]) if output.get("advantage") is not None else None
        ),
        reward=float(output["reward"]),
        metadata={
            "completion_token_count": output_completion_token_count(output),
            "tool_response_token_count": output_tool_response_token_count(output),
            "turn_count": len(output.get("trajectory") or []),
        },
        source=str(output.get("env_name") or output.get("task") or "verifier"),
    )


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
    if not uses_group_advantages(config.algo):
        return records
    epsilon = algorithm_epsilon(config.algo)
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
    grouped: dict[str, list[dict[str, Any]]] = {}
    for output in outputs:
        grouped.setdefault(_output_group_key(output), []).append(output)
    for group in grouped.values():
        _assign_group_advantages(
            group,
            algorithm_config=config.algo,
        )


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
        metadata = {
            "group_key": group_key,
            "rollout_key": f"{group_key}:{sample_index}",
            "stop_condition": output.get("stop_condition"),
            "is_truncated": output.get("is_truncated"),
            "completion_token_count": output_completion_token_count(output),
            "tool_response_token_count": output_tool_response_token_count(output),
            "turn_count": len(output.get("trajectory") or []),
            "_wavelet_rollout_count": 1 if sample_index == 0 else 0,
            **rollout_task_harness_metadata(
                output,
                group_key=group_key,
                sample_index=sample_index,
            ),
        }
        policy_step = output.get("_wavelet_policy_step")
        if isinstance(policy_step, int) and not isinstance(policy_step, bool):
            metadata["policy_step"] = policy_step
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
                metadata=metadata,
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

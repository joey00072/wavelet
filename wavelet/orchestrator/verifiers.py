from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from wavelet.configs.rl_config import RLEvalEnvConfig
from wavelet.data.rl_dataset import RLExample, load_rl_records
from wavelet.orchestrator.advantage import (
    group_reward_advantages,
    length_penalty_cost_for_output,
    output_completion_token_count,
    output_tool_response_token_count,
)
from wavelet.orchestrator.eval_utils import pass_at_k
from wavelet.orchestrator.rollouts import RLOrchestrator


_ENV_CACHE: dict[tuple[str, str], Any] = {}


def _perf_enabled() -> bool:
    return os.environ.get("WAVELET_PERF_LOG", "").lower() in {"1", "true", "yes", "on"}


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

    config = orchestrator.config
    env_id = config.orchestrator.verifier_env_id
    if env_id is None:
        raise ValueError("orchestrator.verifier_env_id is required.")
    base_urls = _verifier_base_urls(config)
    model = config.orchestrator.verifier_model or config.model.name
    env_started_at = perf_counter()
    env, env_cache_hit = _load_cached_env(
        vf,
        env_id,
        config.orchestrator.verifier_env_args,
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
    sampling_args = _sampling_args(config)
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
        )
    )
    rollout_seconds = perf_counter() - rollout_started_at
    convert_started_at = perf_counter()
    _assign_rollout_advantages(outputs, config)
    records = [
        record
        for output in outputs
        for record in _records_from_output(output)
    ]
    convert_seconds = perf_counter() - convert_started_at
    if _perf_enabled():
        print(
            "WAVELET_PERF verifier_rollouts "
            f"env_load={env_load_seconds:.3f} env_cache_hit={int(env_cache_hit)} "
            f"rollout={rollout_seconds:.3f} convert={convert_seconds:.3f} "
            f"outputs={len(outputs)} records={len(records)}",
            flush=True,
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
        )
        self.model = config.orchestrator.verifier_model or config.model.name
        self.sampling_args = _sampling_args(config)
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
        self.pending: dict[asyncio.Task[list[dict[str, Any]]], int] = {}
        self.pending_clients: dict[asyncio.Task[list[dict[str, Any]]], int] = {}

    @property
    def max_inflight_groups(self) -> int:
        async_level = max(1, self.config.orchestrator.max_async_level)
        return max(
            len(self.clients),
            self.target_groups * async_level,
        )

    async def generate_batch(self, *, target_groups: int | None = None) -> list[RLExample]:
        started_at = perf_counter()
        target_groups = self.target_groups if target_groups is None else target_groups
        outputs: list[dict[str, Any]] = []
        accepted_groups = 0
        rejected_groups = 0
        completed_groups = 0
        max_completed_groups = target_groups * (
            self.config.orchestrator.zero_advantage_max_retries + 1
        )

        try:
            while accepted_groups < target_groups:
                self._fill_inflight()
                done, _ = await asyncio.wait(
                    self.pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    self.pending.pop(task, None)
                    self.pending_clients.pop(task, None)
                    completed_groups += 1
                    group_outputs = _completed_group_outputs(task)
                    if _is_usable_training_group(
                        group_outputs,
                        expected_rollouts=self.rollout_count,
                        filter_zero_advantage=(
                            self.config.orchestrator.filter_zero_advantage
                        ),
                        advantage_epsilon=self.config.orchestrator.advantage_epsilon,
                    ):
                        outputs.extend(group_outputs)
                        accepted_groups += 1
                        if accepted_groups >= target_groups:
                            break
                    else:
                        rejected_groups += 1
                    if (
                        completed_groups >= max_completed_groups
                        and accepted_groups < target_groups
                    ):
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

        self._fill_inflight()
        convert_started_at = perf_counter()
        _assign_rollout_advantages(outputs, self.config)
        records = [
            record
            for output in outputs
            for record in _records_from_output(output)
        ]
        records = self.orchestrator._filter_zero_advantage_records(records)
        convert_seconds = perf_counter() - convert_started_at
        if _perf_enabled():
            print(
                "WAVELET_PERF verifier_scheduler "
                f"accepted_groups={accepted_groups} "
                f"rejected_groups={rejected_groups} "
                f"completed_groups={completed_groups} "
                f"inflight_groups={len(self.pending)} "
                f"records={len(records)} "
                f"convert={convert_seconds:.3f} "
                f"total={perf_counter() - started_at:.3f}",
                flush=True,
            )
        return records

    async def aclose(self) -> None:
        for task in self.pending:
            task.cancel()
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
        self.pending.clear()
        self.pending_clients.clear()

    def _fill_inflight(self) -> None:
        while len(self.pending) < self.max_inflight_groups:
            self._schedule_group()

    def _schedule_group(self) -> None:
        record = self._next_record()
        client_index = self._least_loaded_client_index()
        task = asyncio.create_task(
            _run_group(
                self.vf,
                self.env,
                _verifier_example(record),
                client=self.clients[client_index],
                model=self.model,
                sampling_args=self.sampling_args,
                rollout_count=self.rollout_count,
                max_retries=self.config.orchestrator.verifier_max_retries,
                normalize_group_advantages=(
                    self.config.orchestrator.normalize_group_advantages
                ),
                advantage_epsilon=self.config.orchestrator.advantage_epsilon,
                length_penalty=self.config.orchestrator.length_penalty,
            )
        )
        self.pending[task] = self.next_group_id
        self.pending_clients[task] = client_index
        self.next_group_id += 1

    def _next_record(self) -> RLExample:
        record = self.records[self.record_offset % len(self.records)]
        self.record_offset += 1
        return record

    def _least_loaded_client_index(self) -> int:
        counts = [0] * len(self.clients)
        for client_index in self.pending_clients.values():
            counts[client_index] += 1
        return min(range(len(self.clients)), key=counts.__getitem__)


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

    config = orchestrator.config
    env, _env_cache_hit = _load_cached_env(vf, env_config.id, env_config.args)
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
            sampling_args=env_config.sampling.to_sampling_args(),
            rollouts_per_example=env_config.rollouts_per_example,
            max_retries=env_config.max_retries,
        )
    )
    elapsed = perf_counter() - started_at
    env_name = env_config.resolved_name
    output_path = (
        config.output_dir
        / "evals"
        / f"step-{step:06d}"
        / f"{env_name}.jsonl"
    )
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

    config = orchestrator.config
    env, _env_cache_hit = _load_cached_env(vf, env_config.id, env_config.args)
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
        sampling_args=env_config.sampling.to_sampling_args(),
        rollouts_per_example=env_config.rollouts_per_example,
        max_retries=env_config.max_retries,
    )
    elapsed = perf_counter() - started_at
    env_name = env_config.resolved_name
    output_path = (
        config.output_dir
        / "evals"
        / f"step-{step:06d}"
        / f"{env_name}.jsonl"
    )
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
                {"X-data-parallel-rank": str(dp_rank)}
                if data_parallel_size > 1
                else {}
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


def _load_cached_env(vf, env_id: str, env_args: dict[str, Any]) -> tuple[Any, bool]:
    cache_key = (
        env_id,
        json.dumps(env_args, sort_keys=True, default=str),
    )
    cached = _ENV_CACHE.get(cache_key)
    if cached is not None:
        return cached, True
    env = vf.load_environment(env_id, **env_args)
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
) -> list[dict[str, Any]]:
    if not clients:
        raise ValueError("At least one verifier client is required.")
    if target_groups is None or len(records) <= target_groups:
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
        return _successful_rollout_outputs(results)

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


def _successful_rollout_outputs(results: list[Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        try:
            output = dict(result)
        except (TypeError, ValueError):
            continue
        if output.get("error") is not None:
            continue
        if "reward" not in output:
            continue
        outputs.append(output)
    return outputs


def _completed_group_outputs(
    task: asyncio.Task[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    try:
        return task.result()
    except Exception:
        return []


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
    return _has_trainable_advantage(
        outputs,
        filter_zero_advantage=filter_zero_advantage,
        advantage_epsilon=advantage_epsilon,
    )


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


def _sampling_args(config) -> dict[str, Any]:
    sampling = config.inference.sampling
    args: dict[str, Any] = {
        "temperature": sampling.temperature if sampling.do_sample else 0.0,
        "top_p": sampling.top_p,
        "max_completion_tokens": sampling.max_completion_tokens,
        "logprobs": True,
    }
    extra_body: dict[str, Any] = dict(sampling.extra_body)
    extra_body["return_token_ids"] = True
    if sampling.top_k != 0:
        extra_body["top_k"] = sampling.top_k
    extra_body["min_p"] = sampling.min_p
    if sampling.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = sampling.repetition_penalty
    if sampling.seed is not None:
        args["seed"] = sampling.seed
    args["extra_body"] = extra_body
    return args


def _assign_rollout_advantages(outputs: list[dict[str, Any]], config) -> None:
    if config.orchestrator.advantage_mode == "reward":
        for output in outputs:
            output["advantage"] = float(output["reward"])
        return
    if config.orchestrator.advantage_mode != "group_reward":
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for output in outputs:
        grouped.setdefault(str(output.get("example_id", "unknown")), []).append(output)
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
    example_id = str(output.get("example_id", "unknown"))
    records: list[RLExample] = []
    for sample_index, sample in enumerate(_interleave_output(output, temperature)):
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
                    "group_key": example_id,
                    "rollout_key": f"{example_id}:{sample_index}",
                    "stop_condition": output.get("stop_condition"),
                    "is_truncated": output.get("is_truncated"),
                    "completion_token_count": output_completion_token_count(output),
                    "tool_response_token_count": output_tool_response_token_count(
                        output
                    ),
                    "turn_count": len(output.get("trajectory") or []),
                },
                source=str(output.get("env_name") or output.get("task") or "verifier"),
            )
        )
    return records


def _interleave_output(
    output: dict[str, Any],
    temperature: float,
) -> list[dict[str, list[Any]]]:
    trajectory = output.get("trajectory") or []
    if not trajectory:
        return []
    has_error = output.get("error") is not None
    prepared = [_step_tokens(output, step, index) for index, step in enumerate(trajectory)]

    def make_sample(tokens: dict[str, Any]) -> dict[str, list[Any]]:
        prompt_ids = list(tokens["prompt_ids"])
        completion_ids = list(tokens["completion_ids"])
        completion_mask = (
            [False] * len(tokens["completion_mask"])
            if has_error
            else [bool(value) for value in tokens["completion_mask"]]
        )
        return _shift_sample(
            prompt_ids,
            [bool(value) for value in tokens["prompt_mask"]],
            completion_ids,
            completion_mask,
            [float(value) for value in tokens["completion_logprobs"]],
            temperature,
        )

    def extend_sample(
        sample: dict[str, list[Any]],
        prefix_len: int,
        tokens: dict[str, Any],
    ) -> dict[str, list[Any]]:
        prefix_ids = list(tokens["prompt_ids"])[:prefix_len]
        new_prompt_ids = list(tokens["prompt_ids"])[prefix_len:]
        completion_ids = list(tokens["completion_ids"])
        completion_mask = (
            [False] * len(tokens["completion_mask"])
            if has_error
            else [bool(value) for value in tokens["completion_mask"]]
        )
        extension = _shift_sample(
            [],
            [],
            new_prompt_ids + completion_ids,
            [False] * len(new_prompt_ids) + completion_mask,
            [0.0] * len(new_prompt_ids)
            + [float(value) for value in tokens["completion_logprobs"]],
            temperature,
        )
        extension_ids = new_prompt_ids + completion_ids
        if prefix_ids and extension_ids:
            sample["input_ids"].append(prefix_ids[-1])
            sample["target_ids"].append(extension_ids[0])
            sample["loss_mask"].append(False)
            sample["inference_logprobs"].append(0.0)
            sample["temperatures"].append(temperature)
        for key in (
            "input_ids",
            "target_ids",
            "loss_mask",
            "inference_logprobs",
            "temperatures",
        ):
            sample[key].extend(extension[key])
        return sample

    active: list[tuple[list[int], dict[str, list[Any]]]] = []
    first = prepared[0]
    active.append((first["prompt_ids"] + first["completion_ids"], make_sample(first)))
    for tokens in prepared[1:]:
        prompt_ids = tokens["prompt_ids"]
        for index, (prefix, sample) in enumerate(active):
            if prompt_ids[: len(prefix)] == prefix:
                active[index] = (
                    tokens["prompt_ids"] + tokens["completion_ids"],
                    extend_sample(sample, len(prefix), tokens),
                )
                break
        else:
            active.append((tokens["prompt_ids"] + tokens["completion_ids"], make_sample(tokens)))
    return [sample for _, sample in active]


def _step_tokens(output: dict[str, Any], step: dict[str, Any], index: int) -> dict[str, Any]:
    tokens = step.get("tokens")
    if tokens is None:
        raise ValueError(
            f"Verifier rollout for example {output.get('example_id')} step {index} "
            "is missing token data."
        )
    return {
        "prompt_ids": [int(token_id) for token_id in tokens["prompt_ids"]],
        "prompt_mask": [bool(value) for value in tokens["prompt_mask"]],
        "completion_ids": [int(token_id) for token_id in tokens["completion_ids"]],
        "completion_mask": [bool(value) for value in tokens["completion_mask"]],
        "completion_logprobs": [
            float(value) for value in tokens["completion_logprobs"]
        ],
    }


def _shift_sample(
    prompt_ids: list[int],
    prompt_mask: list[bool],
    completion_ids: list[int],
    completion_mask: list[bool],
    completion_logprobs: list[float],
    temperature: float,
) -> dict[str, list[Any]]:
    full_ids = prompt_ids + completion_ids
    full_mask = prompt_mask + completion_mask
    full_logprobs = [0.0] * len(prompt_ids) + completion_logprobs
    full_temperatures = [temperature] * len(full_ids)
    if len(full_ids) < 2:
        return {
            "input_ids": [],
            "target_ids": [],
            "loss_mask": [],
            "inference_logprobs": [],
            "temperatures": [],
        }
    return {
        "input_ids": full_ids[:-1],
        "target_ids": full_ids[1:],
        "loss_mask": full_mask[1:],
        "inference_logprobs": full_logprobs[1:],
        "temperatures": full_temperatures[1:],
    }


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

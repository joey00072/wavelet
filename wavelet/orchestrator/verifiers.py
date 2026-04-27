from __future__ import annotations

import asyncio
import json
import os
from time import perf_counter
from typing import Any

from wavelet.data.rl_dataset import RLExample
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
    base_url = config.orchestrator.verifier_base_url
    if base_url is None:
        http = config.inference.http
        base_url = f"http://{http.host}:{http.port}/v1"
    model = config.orchestrator.verifier_model or config.model.name
    env_started_at = perf_counter()
    env, env_cache_hit = _load_cached_env(
        vf,
        env_id,
        config.orchestrator.verifier_env_args,
    )
    env_load_seconds = perf_counter() - env_started_at
    os.environ.setdefault(config.orchestrator.verifier_api_key_var, "EMPTY")
    client = vf.ClientConfig(
        client_type=config.orchestrator.verifier_client_type,
        api_base_url=base_url,
        api_key_var=config.orchestrator.verifier_api_key_var,
    )
    sampling_args = _sampling_args(config)
    rollout_count = config.orchestrator.rollouts_per_example or 1

    rollout_started_at = perf_counter()
    outputs = asyncio.run(
        _run_all(
            vf,
            env,
            records,
            client=client,
            model=model,
            sampling_args=sampling_args,
            rollout_count=rollout_count,
            max_retries=config.orchestrator.verifier_max_retries,
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
    client,
    model: str,
    sampling_args: dict[str, Any],
    rollout_count: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    tasks = []
    for record in records:
        example = _verifier_example(record)
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
    return await asyncio.gather(*tasks)


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
    extra_body: dict[str, Any] = {"return_token_ids": True}
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
        mean = sum(rewards) / len(rewards)
        std = 0.0
        if config.orchestrator.normalize_group_advantages:
            variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
            std = variance**0.5
        for output, reward in zip(group, rewards, strict=True):
            advantage = reward - mean
            if (
                config.orchestrator.normalize_group_advantages
                and std > config.orchestrator.advantage_epsilon
            ):
                advantage /= std
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

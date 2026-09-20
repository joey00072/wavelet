from __future__ import annotations

from math import ceil
from typing import Protocol

from wavelet.configs.config import RLConfig, RLEvalEnvConfig
from wavelet.orchestrator.eval_utils import compute_eval_policy_step


class PolicyStepReceiver(Protocol):
    def available_steps(self) -> list[int]: ...


def target_steps(config: RLConfig) -> int:
    return 1 if config.max_steps is None else config.max_steps


def rollout_chunk_examples(config: RLConfig) -> int:
    if config.orchestrator.token_batch_size is not None:
        return 1
    configured = config.orchestrator.rollout_chunk_examples
    if configured is not None:
        return configured
    examples_per_step = config.orchestrator.examples_per_step
    if examples_per_step is None:
        return 1
    async_level = max(config.orchestrator.max_async_level, 1)
    return max(1, ceil(examples_per_step / async_level))


def chunks_per_step(config: RLConfig) -> int:
    if config.orchestrator.token_batch_size is not None:
        return 1
    examples_per_step = config.orchestrator.examples_per_step
    if examples_per_step is None:
        raise ValueError("orchestrator.examples_per_step is required.")
    return max(ceil(examples_per_step / rollout_chunk_examples(config)), 1)


def rollout_groups_for_chunk(config: RLConfig, chunk_index: int) -> int:
    """Return the exact group count for one optimizer-step chunk."""
    if config.orchestrator.token_batch_size is not None:
        raise ValueError("Token-based rollout batches have a dynamic group count.")
    examples_per_step = config.orchestrator.examples_per_step
    if examples_per_step is None:
        raise ValueError("orchestrator.examples_per_step is required.")
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative.")
    chunk_examples = rollout_chunk_examples(config)
    remaining = examples_per_step - chunk_index * chunk_examples
    if remaining <= 0:
        raise ValueError(
            f"chunk_index {chunk_index} exceeds the configured optimizer batch."
        )
    return min(chunk_examples, remaining)


def max_policy_lag(config: RLConfig) -> int:
    """Maximum policy age allowed by both freshness constraints."""
    async_level = config.orchestrator.max_async_level
    async_lag = max(async_level - 1, 0)
    off_policy_steps = config.orchestrator.max_off_policy_steps
    return min(async_lag, off_policy_steps)


def required_policy_step(config: RLConfig, rollout_step: int) -> int:
    """Oldest policy step allowed for a rollout under the async window."""
    return max(rollout_step - max_policy_lag(config), 0)


def retained_policy_snapshots(config: RLConfig) -> int:
    """Keep selected policies alive while admitted rollout requests drain."""
    # The trainer can consume the previously published batch during a refresh.
    # Retain the freshness window plus that export and the selected snapshot.
    minimum = (
        ceil(max_policy_lag(config) / config.policy_transfer.export_every_steps) + 2
    )
    return max(config.policy_transfer.keep_last, minimum)


def next_exported_policy_step(config: RLConfig, required_step: int) -> int:
    if required_step <= 0 and config.policy_transfer.export_initial:
        return 0
    interval = config.policy_transfer.export_every_steps
    return ((max(required_step, 1) + interval - 1) // interval) * interval


def latest_exported_policy_step_at_or_before(config: RLConfig, step: int) -> int | None:
    if step <= 0:
        return 0 if config.policy_transfer.export_initial else None
    interval = config.policy_transfer.export_every_steps
    exported_step = (step // interval) * interval
    if exported_step > 0:
        return exported_step
    return 0 if config.policy_transfer.export_initial else None


def policy_step_to_load(
    config: RLConfig,
    policy_receiver: PolicyStepReceiver,
    *,
    rollout_step: int,
    loaded_policy_step: int | None,
) -> int | None:
    required_step = required_policy_step(config, rollout_step)
    available_steps = policy_receiver.available_steps()
    available_in_window = [
        step
        for step in available_steps
        if step >= required_step
        and step <= rollout_step
        and (loaded_policy_step is None or step > loaded_policy_step)
    ]
    if available_in_window:
        return max(available_in_window)
    if loaded_policy_step is None or loaded_policy_step < required_step:
        next_step = next_exported_policy_step(config, required_step)
        latest_allowed = latest_exported_policy_step_at_or_before(
            config,
            rollout_step,
        )
        if latest_allowed is None or latest_allowed < required_step:
            return next_step
        return min(next_step, latest_allowed)
    return None


def select_due_eval_envs(
    config: RLConfig,
    *,
    policy_step: int,
    last_eval_steps: dict[str, int],
) -> list[RLEvalEnvConfig]:
    if config.eval is None:
        return []

    envs: list[RLEvalEnvConfig] = []
    for env in config.eval.env:
        eval_step = compute_eval_policy_step(
            policy_step=policy_step,
            last_eval_step=last_eval_steps[env.resolved_name],
            interval=env.interval,
            eval_base_model=config.eval.eval_base_model,
        )
        if eval_step is None:
            continue
        # The currently loaded policy is what evaluation actually measures.
        # A scheduler can jump over an interval boundary after resume or an
        # asynchronous export, so retaining the nominal boundary would make a
        # later final eval repeat the same loaded policy.
        last_eval_steps[env.resolved_name] = policy_step
        envs.append(env)
    return envs

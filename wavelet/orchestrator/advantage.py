from __future__ import annotations

import math
from typing import Any

from wavelet.configs.rl_config import (
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from wavelet.data.rl_dataset import RLExample


def group_reward_advantages(
    rewards: list[float],
    *,
    costs: list[float] | None = None,
    normalize: bool = False,
    epsilon: float = 1e-6,
) -> list[float]:
    if not rewards:
        return []
    if costs is None:
        advantages = _center(rewards)
    else:
        advantages = _efficiency_shaped_advantages(rewards, costs)
    if normalize:
        variance = sum(value * value for value in advantages) / len(advantages)
        std = math.sqrt(variance)
        if std > epsilon:
            advantages = [value / std for value in advantages]
    return advantages


def length_penalty_cost_for_record(record: RLExample, penalty: object) -> float:
    if isinstance(penalty, TokensLengthPenaltyConfig):
        metadata = record.metadata or {}
        completion_tokens = metadata.get("completion_token_count")
        if not isinstance(completion_tokens, (int, float)):
            completion_tokens = (
                sum(bool(value) for value in record.loss_mask)
                if record.loss_mask is not None
                else 0
            )
        return (
            penalty.completion_weight * float(completion_tokens)
            + penalty.tool_response_weight
            * float(_metadata_number(metadata, "tool_response_token_count"))
        )
    if isinstance(penalty, TurnsLengthPenaltyConfig):
        return float(_metadata_number(record.metadata or {}, "turn_count", default=1.0))
    return 0.0


def length_penalty_cost_for_output(output: dict[str, Any], penalty: object) -> float:
    if isinstance(penalty, TokensLengthPenaltyConfig):
        return (
            penalty.completion_weight * float(_output_completion_token_count(output))
            + penalty.tool_response_weight * float(output_tool_response_token_count(output))
        )
    if isinstance(penalty, TurnsLengthPenaltyConfig):
        return float(len(output.get("trajectory") or []))
    return 0.0


def output_completion_token_count(output: dict[str, Any]) -> int:
    return _output_completion_token_count(output)


def output_tool_response_token_count(output: dict[str, Any]) -> int:
    metrics = output.get("metrics") or {}
    if not isinstance(metrics, dict):
        return 0
    for key, value in metrics.items():
        if key.endswith("total_tool_response_tokens") and isinstance(
            value, (int, float)
        ):
            return int(value)
    return 0


def _efficiency_shaped_advantages(
    rewards: list[float],
    costs: list[float],
) -> list[float]:
    if len(rewards) != len(costs):
        raise ValueError("rewards and costs must have the same length")
    max_reward = max(rewards)
    if max_reward <= 0:
        return _center(rewards)

    correct = [reward >= max_reward for reward in rewards]
    correct_costs = [cost for cost, is_correct in zip(costs, correct) if is_correct]
    mean_correct_cost = (
        sum(correct_costs) / len(correct_costs) if correct_costs else 0.0
    )
    if mean_correct_cost <= 0:
        return _center(rewards)

    shaped_rewards: list[float] = []
    for reward, cost, is_correct in zip(rewards, costs, correct):
        bonus = min(max(1.0 - cost / mean_correct_cost, 0.0), 1.0)
        shaped_rewards.append(reward * (1.0 + bonus if is_correct else 1.0))
    return _center(shaped_rewards)


def _center(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def _metadata_number(
    metadata: dict[str, Any],
    key: str,
    *,
    default: float = 0.0,
) -> float:
    value = metadata.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _output_completion_token_count(output: dict[str, Any]) -> int:
    total = 0
    for step in output.get("trajectory") or []:
        if not isinstance(step, dict):
            continue
        tokens = step.get("tokens") or {}
        if not isinstance(tokens, dict):
            continue
        total += len(tokens.get("completion_ids") or [])
    return total

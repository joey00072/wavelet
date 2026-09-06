from __future__ import annotations

import math
from typing import Any

from wavelet.configs.rl_config import (
    LinearLengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TruncationLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from wavelet.data.rl import RLExample


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
        return penalty.completion_weight * float(
            completion_tokens
        ) + penalty.tool_response_weight * float(
            _metadata_number(metadata, "tool_response_token_count")
        )
    if isinstance(penalty, TurnsLengthPenaltyConfig):
        return float(_metadata_number(record.metadata or {}, "turn_count", default=1.0))
    return 0.0


def linearly_penalized_rewards(
    records: list[RLExample],
    rewards: list[float],
    penalty: LinearLengthPenaltyConfig,
) -> list[float]:
    """Apply group-normalized linear costs scaled by the group's mean reward."""
    if len(records) != len(rewards):
        raise ValueError("records and rewards must have the same length")
    if not rewards:
        return []
    outputs = [_record_length(record, "completion_token_count") for record in records]
    inputs = [_record_input_tokens(record) for record in records]
    turns = [_record_length(record, "turn_count", default=1.0) for record in records]
    output_max = max(max(outputs), 1.0)
    input_max = max(max(inputs), 1.0)
    turn_max = max(max(turns), 1.0)
    pass_rate = sum(rewards) / len(rewards)
    return [
        reward
        - pass_rate
        * (
            penalty.num_output_tokens_weight * output / output_max
            + penalty.num_input_tokens_weight * input_tokens / input_max
            + penalty.num_turns_weight * turn_count / turn_max
        )
        for reward, output, input_tokens, turn_count in zip(
            rewards, outputs, inputs, turns, strict=True
        )
    ]


def truncation_penalized_rewards(
    records: list[RLExample],
    rewards: list[float],
    penalty: TruncationLengthPenaltyConfig,
) -> list[float]:
    """Subtract a fixed reward penalty from max-length-truncated rollouts."""
    if len(records) != len(rewards):
        raise ValueError("records and rewards must have the same length")
    return [
        reward - penalty.penalty if _record_is_truncated(record) else reward
        for record, reward in zip(records, rewards, strict=True)
    ]


def output_completion_token_count(output: dict[str, Any]) -> int:
    return _output_token_total(output, "completion_ids")


def output_input_token_count(output: dict[str, Any]) -> int:
    return _output_token_total(output, "prompt_ids")


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


def _record_length(
    record: RLExample,
    key: str,
    *,
    default: float = 0.0,
) -> float:
    metadata = record.metadata or {}
    value = _metadata_number(metadata, key, default=default)
    if value > 0.0 or key != "completion_token_count":
        return value
    if record.loss_mask is not None:
        return float(sum(bool(item) for item in record.loss_mask))
    return value


def _record_input_tokens(record: RLExample) -> float:
    metadata = record.metadata or {}
    value = _metadata_number(metadata, "input_token_count")
    if value > 0.0:
        return value
    if record.input_ids is None:
        return 0.0
    return max(
        float(len(record.input_ids)) - _record_length(record, "completion_token_count"),
        0.0,
    )


def _record_is_truncated(record: RLExample) -> bool:
    metadata = record.metadata or {}
    if "is_truncated" in metadata:
        return bool(metadata["is_truncated"])
    rollout = metadata.get("rollout")
    return isinstance(rollout, dict) and bool(rollout.get("is_truncated"))


def _output_token_total(output: dict[str, Any], key: str) -> int:
    """Sum the ``key`` token lists across every trajectory step of an output."""
    total = 0
    for step in output.get("trajectory") or []:
        if not isinstance(step, dict):
            continue
        tokens = step.get("tokens") or {}
        if isinstance(tokens, dict):
            total += len(tokens.get(key) or [])
    return total

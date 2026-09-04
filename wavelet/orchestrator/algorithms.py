from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from wavelet.configs.rl_config import (
    AlgorithmScope,
    CustomAlgorithmConfig,
    GRPOAlgorithmConfig,
    LengthPenaltyConfig,
    MaxRLAlgorithmConfig,
    OPDAlgorithmConfig,
    OPSDAlgorithmConfig,
    PassthroughAlgorithmConfig,
    RewardAlgorithmConfig,
    RLAlgorithmConfig,
    SFTDistillAlgorithmConfig,
)
from wavelet.data.rl import RLExample
from wavelet.orchestrator.advantage import (
    group_reward_advantages,
    length_penalty_cost_for_record,
)
from wavelet.orchestrator.custom_algorithms import (
    load_custom_algorithm,
)
from wavelet.orchestrator.custom_algorithms import (
    register_algorithm as register_algorithm,  # noqa: PLC0414 - public re-export
)


class Algorithm(Protocol):
    """Synchronous credit-assignment hooks used by every rollout source."""

    def score_rollout(self, record: RLExample) -> RLExample: ...

    def score_group(self, records: list[RLExample]) -> list[RLExample]: ...


class BaseAlgorithm:
    """No-op base class for algorithms that implement only one hook."""

    def score_rollout(self, record: RLExample) -> RLExample:
        return record

    def score_group(self, records: list[RLExample]) -> list[RLExample]:
        return records


class PassthroughAlgorithm(BaseAlgorithm):
    pass


class RewardAlgorithm(BaseAlgorithm):
    def score_rollout(self, record: RLExample) -> RLExample:
        if record.advantage is not None:
            return record
        return replace(record, advantage=record.reward)


@dataclass(frozen=True, slots=True)
class GRPOAlgorithm(BaseAlgorithm):
    normalize_advantages: bool = False
    epsilon: float = 1e-6
    length_penalty: LengthPenaltyConfig | None = None

    def score_group(self, records: list[RLExample]) -> list[RLExample]:
        """Assign group-relative advantages without mutating the input records."""
        rewards = _required_rewards(records, algorithm="GRPO")
        costs = (
            [
                length_penalty_cost_for_record(record, self.length_penalty)
                for record in records
            ]
            if self.length_penalty is not None
            else None
        )
        advantages = group_reward_advantages(
            rewards,
            costs=costs,
            normalize=self.normalize_advantages,
            epsilon=self.epsilon,
        )
        return [
            replace(record, advantage=advantage)
            for record, advantage in zip(records, advantages, strict=True)
        ]


class MaxRLAlgorithm(BaseAlgorithm):
    def score_group(self, records: list[RLExample]) -> list[RLExample]:
        rewards = _required_rewards(records, algorithm="MaxRL")
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        advantages = (
            [0.0] * len(rewards)
            if mean_reward <= 0.0
            else [(reward - mean_reward) / mean_reward for reward in rewards]
        )
        return [
            replace(record, advantage=advantage)
            for record, advantage in zip(records, advantages, strict=True)
        ]


class OPDAlgorithm(BaseAlgorithm):
    """Route policy samples to reverse-KL distillation against a teacher."""

    def score_rollout(self, record: RLExample) -> RLExample:
        return replace(record, advantage=None, ref_kl_weight=1.0)


class OPSDAlgorithm(BaseAlgorithm):
    """Route policy samples to demo-conditioned self-distillation."""

    def score_rollout(self, record: RLExample) -> RLExample:
        return replace(record, advantage=None, ref_kl_weight=1.0)


class SFTDistillAlgorithm(BaseAlgorithm):
    """Route frozen-teacher samples to token-level cross entropy."""

    def score_rollout(self, record: RLExample) -> RLExample:
        return replace(record, advantage=None, ce_weight=1.0)


def _required_rewards(
    records: list[RLExample],
    *,
    algorithm: str,
) -> list[float]:
    missing = [index for index, record in enumerate(records) if record.reward is None]
    if missing:
        indexes = ", ".join(str(index) for index in missing)
        raise ValueError(
            f"{algorithm} requires a reward for every rollout; missing at "
            f"index(es): {indexes}."
        )
    return [float(record.reward) for record in records if record.reward is not None]


def build_algorithm(config: RLAlgorithmConfig) -> Algorithm:
    """Construct the runtime selected by a validated algorithm config."""
    if isinstance(config, PassthroughAlgorithmConfig):
        return PassthroughAlgorithm()
    if isinstance(config, RewardAlgorithmConfig):
        return RewardAlgorithm()
    if isinstance(config, GRPOAlgorithmConfig):
        return GRPOAlgorithm(
            normalize_advantages=config.normalize_advantages,
            epsilon=config.epsilon,
            length_penalty=config.length_penalty,
        )
    if isinstance(config, MaxRLAlgorithmConfig):
        return MaxRLAlgorithm()
    if isinstance(config, OPDAlgorithmConfig):
        return OPDAlgorithm()
    if isinstance(config, OPSDAlgorithmConfig):
        return OPSDAlgorithm()
    if isinstance(config, SFTDistillAlgorithmConfig):
        return SFTDistillAlgorithm()
    if isinstance(config, CustomAlgorithmConfig):
        return load_custom_algorithm(
            config.file,
            config.algorithm,
            kwargs=config.kwargs,
        )
    raise TypeError(f"Unsupported algorithm config: {type(config).__name__}")


def score_algorithm_records(
    algorithm: Algorithm,
    records: list[RLExample],
    *,
    scope: AlgorithmScope,
    group_key: Callable[[RLExample], str] | None = None,
) -> list[RLExample]:
    """Run the configured hooks and preserve record order across groups."""
    scored = list(records)
    if scope in {"rollout", "both"}:
        scored = [
            _require_example(algorithm.score_rollout(record), hook="score_rollout")
            for record in scored
        ]
    if scope == "rollout" or not scored:
        return scored
    if scope == "group" and all(record.advantage is not None for record in scored):
        return scored

    indexes_by_group: dict[str, list[int]] = {}
    for index, record in enumerate(scored):
        key = group_key(record) if group_key is not None else "group"
        indexes_by_group.setdefault(key, []).append(index)

    for indexes in indexes_by_group.values():
        group = [scored[index] for index in indexes]
        group_result = algorithm.score_group(group)
        if not isinstance(group_result, list):
            raise TypeError("score_group must return list[RLExample].")
        if len(group_result) != len(group):
            raise ValueError(
                "score_group must return one RLExample for every input record "
                f"({len(group_result)} != {len(group)})."
            )
        for index, record in zip(indexes, group_result, strict=True):
            scored[index] = _require_example(record, hook="score_group")
    return scored


def _require_example(value: object, *, hook: str) -> RLExample:
    if not isinstance(value, RLExample):
        raise TypeError(f"{hook} must return RLExample values.")
    return value


def algorithm_epsilon(config: RLAlgorithmConfig) -> float:
    """Return the zero-advantage threshold for the selected algorithm."""
    if isinstance(config, (GRPOAlgorithmConfig, CustomAlgorithmConfig)):
        return config.epsilon
    return 1e-6


def algorithm_scope(config: RLAlgorithmConfig) -> AlgorithmScope:
    """Return which scoring hooks the orchestrator must execute."""
    if isinstance(config, CustomAlgorithmConfig):
        return config.scope
    if isinstance(config, (GRPOAlgorithmConfig, MaxRLAlgorithmConfig)):
        return "group"
    return "rollout"


def uses_group_advantages(config: RLAlgorithmConfig) -> bool:
    """Return whether complete rollout groups are required before scoring."""
    return algorithm_scope(config) in {"group", "both"}


def algorithm_loss_component(config: RLAlgorithmConfig) -> str:
    """Return the token-loss component owned by the algorithm."""
    if isinstance(config, SFTDistillAlgorithmConfig):
        return "ce"
    if isinstance(config, (OPDAlgorithmConfig, OPSDAlgorithmConfig)):
        return "ref_kl"
    return "rl"

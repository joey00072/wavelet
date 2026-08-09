from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from wavelet.configs.rl_config import (
    AlgorithmScope,
    CustomAlgorithmConfig,
    GRPOAlgorithmConfig,
    LengthPenaltyConfig,
    MaxRLAlgorithmConfig,
    OPDAlgorithmConfig,
    PassthroughAlgorithmConfig,
    RewardAlgorithmConfig,
    RLAlgorithmConfig,
    RLConfig,
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
    register_algorithm as register_algorithm,
)
from wavelet.orchestrator.reference import ReferenceScorer, VLLMReferenceScorer

ActionLossType = Literal["rl", "ce", "ref_kl"]


class Algorithm(Protocol):
    """Synchronous credit-assignment hooks used by every rollout source."""

    action_loss_type: ActionLossType

    def score_rollout(self, record: RLExample) -> RLExample: ...

    def score_group(self, records: list[RLExample]) -> list[RLExample]: ...


class BaseAlgorithm:
    """No-op base class for algorithms that implement only one hook."""

    action_loss_type: ActionLossType = "rl"

    def setup(self) -> None:
        return None

    def close(self) -> None:
        return None

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
    """Attach frozen-teacher prefill logprobs for reverse-KL training."""

    action_loss_type = "ref_kl"

    def __init__(
        self,
        config: OPDAlgorithmConfig,
        *,
        scorer: ReferenceScorer | None = None,
    ) -> None:
        self.config = config
        self.scorer = scorer or VLLMReferenceScorer(config.teacher)

    def score_rollout(self, record: RLExample) -> RLExample:
        if record.ref_logprobs is not None and record.ref_kl_weights is not None:
            return record
        input_ids = record.input_ids
        target_ids = record.target_ids
        loss_mask = record.loss_mask
        if input_ids is None or target_ids is None or loss_mask is None:
            raise ValueError(
                "OPD requires pretokenized input_ids, target_ids, and loss_mask."
            )
        if not (len(input_ids) == len(target_ids) == len(loss_mask)):
            raise ValueError(
                "OPD token streams must have identical lengths "
                f"({len(input_ids)}, {len(target_ids)}, {len(loss_mask)})."
            )
        if not target_ids:
            raise ValueError("OPD requires at least one token to score.")
        for index in range(len(input_ids) - 1):
            if input_ids[index + 1] != target_ids[index]:
                raise ValueError(
                    "OPD requires causal shifted input_ids and target_ids."
                )

        full_token_ids = [*input_ids, target_ids[-1]]
        full_logprobs = self.scorer.score(full_token_ids)
        if len(full_logprobs) != len(full_token_ids):
            raise ValueError(
                "OPD teacher logprobs must align with the full token sequence "
                f"({len(full_logprobs)} != {len(full_token_ids)})."
            )
        ref_logprobs = [
            float(logprob)
            for logprob, trainable in zip(
                full_logprobs[1:],
                loss_mask,
                strict=True,
            )
            if trainable
        ]
        return replace(
            record,
            advantage=None,
            ref_logprobs=ref_logprobs,
            rl_weights=0.0,
            ce_weights=0.0,
            ref_kl_weights=1.0,
        )


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
        return OPDAlgorithm(config)
    if isinstance(config, CustomAlgorithmConfig):
        return load_custom_algorithm(
            config.file,
            config.algorithm,
            kwargs=config.kwargs,
        )
    raise TypeError(f"Unsupported algorithm config: {type(config).__name__}")


def algorithm_config_for_source(
    config: RLConfig,
    source_name: str,
) -> RLAlgorithmConfig:
    """Resolve a source-local algorithm, falling back to the run default."""
    for source in config.orchestrator.train_sources:
        if source.name == source_name and source.algo is not None:
            return source.algo
    return config.algo


def score_records_by_source(
    config: RLConfig,
    records: list[RLExample],
    *,
    group_key: Callable[[RLExample], str] | None = None,
) -> list[RLExample]:
    """Apply each source's algorithm while preserving global record order."""
    scored = list(records)
    indexes_by_source: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        indexes_by_source.setdefault(record.source, []).append(index)
    for source_name, indexes in indexes_by_source.items():
        algorithm_config = algorithm_config_for_source(config, source_name)
        source_records = [records[index] for index in indexes]
        source_scored = score_algorithm_records(
            build_algorithm(algorithm_config),
            source_records,
            scope=algorithm_scope(algorithm_config),
            group_key=group_key,
        )
        for index, record in zip(indexes, source_scored, strict=True):
            scored[index] = record
    return scored


def score_algorithm_records(
    algorithm: Algorithm,
    records: list[RLExample],
    *,
    scope: AlgorithmScope,
    group_key: Callable[[RLExample], str] | None = None,
) -> list[RLExample]:
    """Run one algorithm lifecycle and preserve record order across groups."""
    setup = getattr(algorithm, "setup", None)
    close = getattr(algorithm, "close", None)
    if callable(setup):
        setup()
    try:
        scored = _score_algorithm_records(
            algorithm,
            records,
            scope=scope,
            group_key=group_key,
        )
        return [_apply_action_loss_type(algorithm, record) for record in scored]
    finally:
        if callable(close):
            close()


def _score_algorithm_records(
    algorithm: Algorithm,
    records: list[RLExample],
    *,
    scope: AlgorithmScope,
    group_key: Callable[[RLExample], str] | None = None,
) -> list[RLExample]:
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


def _apply_action_loss_type(algorithm: Algorithm, record: RLExample) -> RLExample:
    if any(
        weight is not None
        for weight in (
            record.rl_weights,
            record.ce_weights,
            record.ref_kl_weights,
        )
    ):
        return record
    action_loss_type = getattr(algorithm, "action_loss_type", "rl")
    if action_loss_type not in {"rl", "ce", "ref_kl"}:
        raise ValueError(
            "algorithm.action_loss_type must be 'rl', 'ce', or 'ref_kl'; "
            f"got {action_loss_type!r}."
        )
    if action_loss_type == "rl":
        return record
    return replace(
        record,
        rl_weights=float(action_loss_type == "rl"),
        ce_weights=float(action_loss_type == "ce"),
        ref_kl_weights=float(action_loss_type == "ref_kl"),
    )


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


def requires_tokenized_records(config: RLAlgorithmConfig) -> bool:
    """Return whether scoring must wait for the serialized token trajectory."""
    return isinstance(config, OPDAlgorithmConfig)

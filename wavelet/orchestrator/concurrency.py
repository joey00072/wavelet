from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from wavelet.configs.config import RLAdaptiveConcurrencyConfig

logger = logging.getLogger(__name__)

ConcurrencySignal = Literal["clear", "soft", "hard"]
_SIGNAL_SEVERITY: dict[ConcurrencySignal, int] = {
    "clear": 0,
    "soft": 1,
    "hard": 2,
}


@dataclass(frozen=True, slots=True)
class EngineLoadSample:
    """Load and capacity signals from one inference engine."""

    replica: str
    kv_cache_usage: float
    running: int
    waiting: int
    preemptions_delta: int
    kv_capacity_tokens: int | None = None
    max_model_len: int | None = None
    waiting_capacity: int | None = None
    role: Literal["prefill", "decode"] | None = None


@dataclass(frozen=True, slots=True)
class ConcurrencyDecision:
    limit: int
    cancel_rollouts: int = 0
    reason: str | None = None


class AdaptiveConcurrencyController:
    """Adapt the in-flight rollout cap to observed inference-engine pressure."""

    def __init__(
        self,
        config: RLAdaptiveConcurrencyConfig,
        *,
        fallback_limit: int,
        minimum_burst: int,
        fallback_cost: int | None = None,
    ) -> None:
        self.config = config
        self.floor = max(config.min_inflight, minimum_burst)
        configured_ceiling = (
            config.max_inflight
            if config.max_inflight is not None
            else fallback_limit
        )
        self.ceiling = max(self.floor, configured_ceiling)
        self.fallback_cost = max(fallback_cost or fallback_limit, 1)

        initial = config.initial_inflight or self.floor
        self.cap = self._clamp(float(initial))
        self.limit = int(self.cap)
        self.bootstrapped = config.initial_inflight is not None
        self.engine_max_len: int | None = None
        self.capacity_by_replica: dict[str, int] = {}

        self.turnover = 0.0
        self.signal: ConcurrencySignal = "clear"
        self.can_grow = False
        self.can_grow_until = 0.0
        self.previous_waiting: dict[str, int] = {}
        self.queue_overload_polls = 0
        self.trim_cooldown = 0
        self.escalation_grace = 0
        self.draining = False
        self.escalated = False
        self.adjustments = 0

    @property
    def capacity(self) -> int | None:
        """Total reported KV capacity across decode replicas."""
        return sum(self.capacity_by_replica.values()) or None

    def record_episode(self, *, tokens: int, inflight: int) -> ConcurrencyDecision:
        """Advance the turnover clock after one rollout completes."""
        active = max(inflight, 1)
        fraction = 1.0 / active
        self.turnover += fraction
        if (
            tokens > 0
            and self.can_grow
            and time.monotonic() < self.can_grow_until
            and active >= self.config.binding_fraction * self.limit
        ):
            self.cap = self._clamp(
                self.cap * self.config.growth_factor_per_turnover**fraction
            )
            return self._apply_limit(int(self.cap))
        return ConcurrencyDecision(limit=self.limit)

    def observe(
        self,
        samples: list[EngineLoadSample],
        *,
        inflight: int,
    ) -> ConcurrencyDecision:
        """Classify one engine poll and resize the rollout cap when needed."""
        if not samples:
            return ConcurrencyDecision(limit=self.limit)

        for sample in samples:
            if sample.kv_capacity_tokens and sample.role != "prefill":
                self.capacity_by_replica[sample.replica] = sample.kv_capacity_tokens
            if sample.max_model_len:
                self.engine_max_len = max(
                    self.engine_max_len or 0,
                    sample.max_model_len,
                )

        decode_samples = [sample for sample in samples if sample.role != "prefill"]
        if not decode_samples:
            return ConcurrencyDecision(limit=self.limit)

        signal: ConcurrencySignal = "clear"
        max_usage = 0.0
        total_running = 0
        total_queued = 0
        preempted = False
        for sample in decode_samples:
            if sample.preemptions_delta > 0:
                preempted = True
                signal = "hard"
            if sample.waiting > 0 and self.previous_waiting.get(sample.replica, 0) > 0:
                signal = self._worst(signal, "soft")
            max_usage = max(max_usage, sample.kv_cache_usage)
            total_running += sample.running
            total_queued += (
                sample.waiting_capacity
                if sample.waiting_capacity is not None
                else sample.waiting
            )
        self.previous_waiting = {
            sample.replica: sample.waiting for sample in decode_samples
        }

        queue_over_threshold = (
            total_running > 0
            and total_queued > self.config.max_waiting_requests
            and total_queued > self.config.queue_ratio * total_running
        )
        if queue_over_threshold:
            self.queue_overload_polls += 1
        else:
            self.queue_overload_polls = 0
        queue_overload = (
            self.queue_overload_polls >= self.config.queue_persistence_polls
        )
        if queue_overload or max_usage > self.config.growth_kv_cache_usage:
            signal = self._worst(signal, "hard" if queue_overload else "soft")
        self.signal = signal

        if (
            self.draining
            and inflight <= self.limit
            and not preempted
            and not queue_overload
        ):
            self.draining = False
            self.escalation_grace = self.config.escalation_grace_polls
        if not self.draining and self.escalated:
            self.escalation_grace -= 1
            if self.escalation_grace <= 0:
                self.escalated = False
        self.trim_cooldown = max(0, self.trim_cooldown - 1)
        self.can_grow = (
            signal == "clear" and total_queued == 0 and not self.draining
        )
        self.can_grow_until = time.monotonic() + (
            self.config.poll_interval_seconds * self.config.growth_gate_polls
        )

        if not self.bootstrapped and self.capacity is not None:
            self.bootstrapped = True
            episode_cost = float(self.engine_max_len or self.fallback_cost)
            self.cap = self._clamp(self.capacity / episode_cost)
            decision = self._apply_limit(int(self.cap))
            logger.info(
                "Derived initial verifier concurrency %s from %s KV-cache "
                "tokens / %.0f tokens per rollout.",
                self.limit,
                self.capacity,
                episode_cost,
            )
            if self.draining:
                return decision

        if self.draining:
            return ConcurrencyDecision(limit=self.limit)

        if preempted or queue_overload:
            cut_fraction = (
                self.config.escalated_decrease_factor if self.escalated else None
            )
            if queue_overload:
                self.queue_overload_polls = 0
                target = int(
                    self._clamp(
                        inflight
                        * (cut_fraction or self.config.queue_decrease_factor)
                    )
                )
                reason = "queue overload"
            else:
                target = int(
                    self._clamp(
                        inflight
                        * (cut_fraction or self.config.preemption_decrease_factor)
                    )
                )
                reason = "preemptions"
            decision = self._resize_down(
                target,
                inflight=inflight,
                reason=reason,
                cancel=True,
            )
            self.draining = True
            self.escalated = True
            return decision

        if (
            max_usage > self.config.soft_kv_cache_usage
            and inflight > 0
            and self.trim_cooldown == 0
        ):
            hard = max_usage > self.config.hard_kv_cache_usage
            target = int(
                self._clamp(
                    inflight * self.config.target_kv_cache_usage / max_usage
                )
            )
            decision = self._resize_down(
                target,
                inflight=inflight,
                reason=(
                    f"KV headroom (usage {max_usage:.2f}, "
                    f"{'hard' if hard else 'soft'} trim)"
                ),
                cancel=hard,
            )
            self.trim_cooldown = self.config.decrease_cooldown_polls
            return decision

        return ConcurrencyDecision(limit=self.limit)

    def _resize_down(
        self,
        target: int,
        *,
        inflight: int,
        reason: str,
        cancel: bool,
    ) -> ConcurrencyDecision:
        target = min(target, self.limit)
        self.cap = float(target)
        decision = self._apply_limit(target, reason=reason)
        if cancel and inflight > target:
            return ConcurrencyDecision(
                limit=decision.limit,
                cancel_rollouts=inflight - target,
                reason=reason,
            )
        return decision

    def _apply_limit(
        self,
        target: int,
        *,
        reason: str | None = None,
    ) -> ConcurrencyDecision:
        if target == self.limit:
            return ConcurrencyDecision(limit=self.limit)
        previous = self.limit
        self.limit = target
        self.adjustments += 1
        if reason is not None:
            logger.info(
                "Adjusted verifier concurrency %s -> %s (%s); "
                "turnover=%.1f signal=%s.",
                previous,
                target,
                reason,
                self.turnover,
                self.signal,
            )
        return ConcurrencyDecision(limit=target, reason=reason)

    def _clamp(self, value: float) -> float:
        return min(max(value, float(self.floor)), float(self.ceiling))

    @staticmethod
    def _worst(
        left: ConcurrencySignal,
        right: ConcurrencySignal,
    ) -> ConcurrencySignal:
        return max(left, right, key=_SIGNAL_SEVERITY.__getitem__)

    def metrics(self) -> dict[str, float]:
        return {
            "generation/concurrency/limit": float(self.limit),
            "generation/concurrency/turnover": self.turnover,
            "generation/concurrency/capacity_tokens": float(self.capacity or 0),
            "generation/concurrency/adjustments": float(self.adjustments),
            "generation/concurrency/signal": float(
                _SIGNAL_SEVERITY[self.signal]
            ),
        }

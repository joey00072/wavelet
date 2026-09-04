from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wavelet.configs.config import RLAdaptiveConcurrencyConfig


@dataclass(frozen=True, slots=True)
class EngineLoadSample:
    """Load signals from one inference-server metrics endpoint."""

    replica: str
    kv_cache_usage: float
    running: int
    waiting: int
    preemptions_delta: int


@dataclass(frozen=True, slots=True)
class ConcurrencyDecision:
    limit: int
    cancel_rollouts: int = 0
    reason: str | None = None


class AdaptiveConcurrencyController:
    """AIMD controller for the verifier scheduler's rollout cap."""

    def __init__(
        self,
        config: RLAdaptiveConcurrencyConfig,
        *,
        fallback_limit: int,
        minimum_burst: int,
    ) -> None:
        self.config = config
        self.floor = max(config.min_inflight, minimum_burst)
        self.ceiling = max(
            self.floor,
            config.max_inflight if config.max_inflight is not None else fallback_limit,
        )
        initial = config.initial_inflight
        if initial is None:
            initial = fallback_limit
        self.limit = min(max(initial, self.floor), self.ceiling)
        self.signal = "fallback"
        self.adjustments = 0
        self.queue_overload_polls = 0
        self.decrease_cooldown_polls = 0

    def observe(
        self,
        samples: list[EngineLoadSample],
        *,
        inflight: int,
    ) -> ConcurrencyDecision:
        if not samples:
            self.signal = "fallback"
            return ConcurrencyDecision(limit=self.limit)

        max_usage = max(sample.kv_cache_usage for sample in samples)
        waiting = sum(sample.waiting for sample in samples)
        preemptions = sum(sample.preemptions_delta for sample in samples)
        if waiting > self.config.max_waiting_requests:
            self.queue_overload_polls += 1
        else:
            self.queue_overload_polls = 0
        queue_overload = (
            self.queue_overload_polls >= self.config.queue_persistence_polls
        )
        if self.decrease_cooldown_polls > 0:
            self.decrease_cooldown_polls -= 1
            self.signal = "steady"
            return ConcurrencyDecision(limit=self.limit)
        hard_overload = (
            preemptions > 0
            or queue_overload
            or max_usage >= self.config.hard_kv_cache_usage
        )
        if hard_overload:
            target = self._decreased_limit(max_usage=max_usage, inflight=inflight)
            reason = (
                "preemptions"
                if preemptions > 0
                else "request queue"
                if queue_overload
                else "KV cache pressure"
            )
            decision = self._resize(
                target,
                cancel_rollouts=max(inflight - target, 0),
                reason=reason,
                signal="hard",
            )
            self._start_decrease_cooldown()
            return decision

        if max_usage > self.config.target_kv_cache_usage:
            decision = self._resize(
                self._decreased_limit(max_usage=max_usage, inflight=inflight),
                reason="KV cache headroom",
                signal="soft",
            )
            self._start_decrease_cooldown()
            return decision

        cap_is_binding = inflight >= max(
            self.limit - self.config.additive_increase,
            self.floor,
        )
        if (
            waiting == 0
            and max_usage <= self.config.growth_kv_cache_usage
            and cap_is_binding
        ):
            return self._resize(
                self.limit + self.config.additive_increase,
                reason="available KV cache headroom",
                signal="clear",
            )

        self.signal = "steady"
        return ConcurrencyDecision(limit=self.limit)

    def _decreased_limit(self, *, max_usage: float, inflight: int) -> int:
        multiplicative = int(self.limit * self.config.decrease_factor)
        target = multiplicative
        if max_usage > 0.0 and inflight > 0:
            target = min(
                target,
                int(inflight * self.config.target_kv_cache_usage / max_usage),
            )
        return max(target, self.floor)

    def _start_decrease_cooldown(self) -> None:
        self.decrease_cooldown_polls = self.config.decrease_cooldown_polls

    def _resize(
        self,
        target: int,
        *,
        reason: str,
        signal: str,
        cancel_rollouts: int = 0,
    ) -> ConcurrencyDecision:
        target = min(max(target, self.floor), self.ceiling)
        self.signal = signal
        if target == self.limit:
            return ConcurrencyDecision(
                limit=self.limit,
                cancel_rollouts=cancel_rollouts,
                reason=reason if cancel_rollouts > 0 else None,
            )
        self.limit = target
        self.adjustments += 1
        return ConcurrencyDecision(
            limit=target,
            cancel_rollouts=cancel_rollouts,
            reason=reason,
        )

    def metrics(self) -> dict[str, float]:
        signals = {
            "fallback": 0.0,
            "clear": 1.0,
            "steady": 2.0,
            "soft": 3.0,
            "hard": 4.0,
        }
        return {
            "generation/concurrency/limit": float(self.limit),
            "generation/concurrency/adjustments": float(self.adjustments),
            "generation/concurrency/signal": signals[self.signal],
        }

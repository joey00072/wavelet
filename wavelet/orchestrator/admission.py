from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class _AdmissionPermit:
    window: int
    cost: int


T = TypeVar("T")


class RolloutAdmissionController:
    """Pace verifier dispatch while allowing steady-state replacements."""

    def __init__(
        self,
        *,
        max_inflight: int,
        minimum_burst: int,
        tasks_per_minute: int | None = None,
        window_seconds: float = 5.0,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be at least 1.")
        if minimum_burst < 1:
            raise ValueError("minimum_burst must be at least 1.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if tasks_per_minute is not None and tasks_per_minute < 1:
            raise ValueError("tasks_per_minute must be at least 1.")

        self.burst_limit = max(minimum_burst, max_inflight // 10)
        self.window_seconds = window_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._window_started_at = clock()
        self._window = 0
        self._admissions = 0
        self._rate_capacity = tasks_per_minute
        self._rate_per_second = (
            None if tasks_per_minute is None else tasks_per_minute / 60.0
        )
        self._rate_tokens = (
            None if tasks_per_minute is None else float(tasks_per_minute)
        )
        self._rate_updated_at = clock()
        # The controller can survive consecutive asyncio.run() calls in the
        # synchronous orchestrator and remains safe if callers use worker loops.
        self._lock = Lock()

    async def run(self, *, cost: int, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one request after admission and refund only natural completion."""
        permit = await self.acquire(cost)
        try:
            result = await operation()
        except asyncio.CancelledError:
            # Cancellation is usually load shedding or staleness cleanup. Keeping
            # its admission charged prevents a cancellation wave from refilling
            # the dispatcher immediately.
            raise
        except BaseException:
            self.release(permit)
            raise
        self.release(permit)
        return result

    async def acquire(self, cost: int = 1) -> _AdmissionPermit:
        """Wait for global rate and burst capacity, then return a permit."""
        if cost < 1:
            raise ValueError("Admission cost must be at least 1.")
        if cost > self.burst_limit:
            raise ValueError(
                f"Admission cost {cost} exceeds burst limit {self.burst_limit}."
            )

        for _ in range(cost):
            await self._acquire_rate_token()

        while True:
            with self._lock:
                now = self._clock()
                elapsed = now - self._window_started_at
                if elapsed >= self.window_seconds:
                    windows_elapsed = max(1, int(elapsed // self.window_seconds))
                    self._window_started_at += windows_elapsed * self.window_seconds
                    self._window += windows_elapsed
                    self._admissions = 0
                if self._admissions + cost <= self.burst_limit:
                    self._admissions += cost
                    return _AdmissionPermit(window=self._window, cost=cost)
                delay = max(self.window_seconds - elapsed, 0.0)
            await self._sleeper(delay)

    async def _acquire_rate_token(self) -> None:
        if self._rate_capacity is None or self._rate_per_second is None:
            return
        while True:
            with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._rate_updated_at)
                self._rate_tokens = min(
                    float(self._rate_capacity),
                    float(self._rate_tokens) + elapsed * self._rate_per_second,
                )
                self._rate_updated_at = now
                if self._rate_tokens >= 1.0:
                    self._rate_tokens -= 1.0
                    return
                delay = (1.0 - self._rate_tokens) / self._rate_per_second
            await self._sleeper(delay)

    def release(self, permit: _AdmissionPermit) -> None:
        """Refund a naturally completed request in its admission window."""
        with self._lock:
            if permit.window == self._window:
                self._admissions = max(0, self._admissions - permit.cost)

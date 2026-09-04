from __future__ import annotations

import asyncio

import pytest

from wavelet.orchestrator.admission import RolloutAdmissionController


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def test_admission_controller_smooths_net_growth_by_window() -> None:
    clock = _Clock()
    controller = RolloutAdmissionController(
        max_inflight=100,
        minimum_burst=8,
        clock=clock,
        sleeper=clock.sleep,
    )

    async def exercise() -> None:
        first = await controller.acquire(8)
        second = await controller.acquire(3)

        assert clock.sleeps == [5.0]
        # A completion from the previous window must not refund admissions in
        # the current window.
        controller.release(first)
        third = await controller.acquire(7)
        controller.release(second)
        controller.release(third)

    asyncio.run(exercise())

    assert controller.burst_limit == 10
    assert clock.sleeps == [5.0]


def test_natural_completion_refunds_admission_without_sleep() -> None:
    clock = _Clock()
    controller = RolloutAdmissionController(
        max_inflight=8,
        minimum_burst=2,
        clock=clock,
        sleeper=clock.sleep,
    )

    async def exercise() -> None:
        assert await controller.run(cost=2, operation=lambda: _run_value(1)) == 1
        assert await controller.run(cost=2, operation=lambda: _run_value(2)) == 2

    asyncio.run(exercise())

    assert clock.sleeps == []


async def _run_value(value: int) -> int:
    return value


async def _run_none() -> None:
    return None


def test_cancelled_request_does_not_refund_admission() -> None:
    controller = RolloutAdmissionController(max_inflight=1, minimum_burst=1)

    async def exercise() -> None:
        started = asyncio.Event()

        async def wait_forever() -> None:
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(controller.run(cost=1, operation=wait_forever))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert controller._admissions == 1

    asyncio.run(exercise())


def test_rate_limiter_charges_each_rollout_before_dispatch() -> None:
    clock = _Clock()
    controller = RolloutAdmissionController(
        max_inflight=1,
        minimum_burst=1,
        tasks_per_minute=1,
        clock=clock,
        sleeper=clock.sleep,
    )

    asyncio.run(controller.run(cost=1, operation=_run_none))
    asyncio.run(controller.run(cost=1, operation=_run_none))

    assert clock.sleeps == [60.0]

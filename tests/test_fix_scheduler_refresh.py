from __future__ import annotations

import asyncio
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.scheduler import (
    VerifierRolloutScheduler,
    _PendingVerifierRequest,
    _VerifierGroupState,
)

from test_verifiers_rollouts import _bare_scheduler


def _scheduler(
    config: RLConfig,
    *,
    policy_step: int | None,
    rollout_step: int | None = None,
    rollout_count: int = 1,
) -> VerifierRolloutScheduler:
    return _bare_scheduler(
        config=config,
        policy_step=policy_step,
        rollout_step=rollout_step,
        rollout_count=rollout_count,
        env_name="env",
    )


def test_group_interrupted_by_policy_swap_is_cancelled_not_rejected() -> None:
    config = RLConfig(orchestrator={"max_async_level": 2, "max_off_policy_steps": 1})
    scheduler = _scheduler(config, policy_step=10, rollout_step=10, rollout_count=2)
    scheduler.groups = {
        0: _VerifierGroupState(
            example={"example_id": 0},
            rollouts_to_schedule=0,
            policy_step=9,
            completed_outputs=[{"example_id": 0, "reward": 0.0, "error": None}],
        )
    }

    async def run() -> tuple[int, int, int]:
        async def rollout() -> list[dict[str, Any]]:
            return [{"example_id": 0, "reward": 1.0, "error": None}]

        task = asyncio.create_task(rollout())
        await task
        scheduler.pending[task] = _PendingVerifierRequest(
            group_id=0, client_index=0, rollout_count=2, policy_step=9
        )
        return scheduler._consume_completed_task(
            task, target_groups=1, outputs=[], accepted_groups=0
        )

    result = asyncio.run(run())

    # Not a reward-filter rejection: the retry budget must not be consumed.
    assert result == (0, 0, 0)
    assert scheduler.groups == {}
    assert scheduler.cancelled_rollouts_count == 2


def test_discard_stale_requests_cancels_doomed_work_before_a_swap() -> None:
    config = RLConfig(orchestrator={"max_async_level": 2, "max_off_policy_steps": 1})
    scheduler = _scheduler(config, policy_step=9, rollout_step=9, rollout_count=2)

    async def run() -> tuple[int, int, list[int]]:
        stale = asyncio.create_task(asyncio.sleep(30))
        fresh = asyncio.create_task(asyncio.sleep(30))
        scheduler.pending[stale] = _PendingVerifierRequest(
            group_id=0, client_index=0, rollout_count=2, policy_step=8
        )
        scheduler.pending[fresh] = _PendingVerifierRequest(
            group_id=1, client_index=0, rollout_count=2, policy_step=9
        )
        scheduler.groups = {
            0: _VerifierGroupState(example={}, rollouts_to_schedule=0, policy_step=8),
            1: _VerifierGroupState(example={}, rollouts_to_schedule=0, policy_step=9),
        }
        try:
            cancelled = await scheduler.discard_stale_requests(10)
        finally:
            fresh.cancel()
            await asyncio.gather(fresh, return_exceptions=True)
        remaining = [request.group_id for request in scheduler.pending.values()]
        return cancelled, scheduler.rollout_step, remaining

    cancelled, rollout_step, remaining = asyncio.run(run())

    # Policy 8 cannot serve rollout step 10 under a one-step lag.
    assert cancelled == 2
    assert rollout_step == 10
    assert remaining == [1]
    assert 0 not in scheduler.groups
    assert scheduler.cancelled_rollouts_count == 2

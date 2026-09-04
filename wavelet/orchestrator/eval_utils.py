from __future__ import annotations

import math


def compute_eval_policy_step(
    *,
    policy_step: int,
    last_eval_step: int,
    interval: int,
    eval_base_model: bool = True,
) -> int | None:
    if policy_step <= last_eval_step:
        return None
    highest_interval_step = (policy_step // interval) * interval
    if highest_interval_step <= last_eval_step:
        return None
    if highest_interval_step == 0:
        if policy_step == 0 and eval_base_model and last_eval_step < 0:
            return 0
        return None
    return highest_interval_step


def pass_at_k(rewards: list[float]) -> dict[str, float]:
    """Return unbiased at-least-one and all-correct metrics for binary rewards."""
    n = len(rewards)
    c = sum(reward == 1.0 for reward in rewards)
    if n == 0:
        return {}
    ks = [2**index for index in range(n.bit_length())]
    return {
        key: value
        for k in ks
        for key, value in (
            (f"pass@{k}", _pass_at_k(n, c, k)),
            (f"pass^{k}", _pass_power_k(n, c, k)),
        )
    }


def _pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _pass_power_k(n: int, c: int, k: int) -> float:
    return math.comb(c, k) / math.comb(n, k)

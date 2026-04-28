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
    if policy_step == 0:
        return 0 if eval_base_model and last_eval_step < 0 else None
    if policy_step % interval == 0:
        return policy_step
    return None


def pass_at_k(rewards: list[float]) -> dict[str, float]:
    n = len(rewards)
    c = sum(reward == 1.0 for reward in rewards)
    if n == 0:
        return {}
    ks = [2**index for index in range(n.bit_length())]
    return {f"pass@{k}": _pass_at_k(n, c, k) for k in ks}


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))

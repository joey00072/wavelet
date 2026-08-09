from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from wavelet.data.rl import RLExample


VERIFIER_ROLLOUT_FUNCTION = "wavelet.orchestrator.verifiers:generate_rollouts"


class RolloutSourceKind(StrEnum):
    NATIVE = "native"
    VERIFIER = "verifier"
    CUSTOM = "custom"


class RolloutSource(Protocol):
    """Source seam shared by rollout schedulers."""

    async def rollout_group(
        self,
        example: RLExample,
        n: int,
        sampling: dict[str, object],
    ) -> list[RLExample]: ...


RolloutGroupFunction = Callable[
    [RLExample, int, dict[str, object]],
    list[RLExample] | Awaitable[list[RLExample]],
]


@dataclass(slots=True)
class CallableRolloutSource:
    """Adapts native, verifier, and custom group functions to one async seam."""

    rollout_function: RolloutGroupFunction

    async def rollout_group(
        self,
        example: RLExample,
        n: int,
        sampling: dict[str, object],
    ) -> list[RLExample]:
        result = self.rollout_function(example, n, sampling)
        if inspect.isawaitable(result):
            return await result
        return result


class NativeSource(CallableRolloutSource):
    """Native policy-inference rollout source."""


class VerifierSource(CallableRolloutSource):
    """Verifier-environment rollout source."""


class CustomSource(CallableRolloutSource):
    """User-provided rollout-function source."""


def source_kind(custom_rollout_function: str | None) -> RolloutSourceKind:
    if custom_rollout_function is None:
        return RolloutSourceKind.NATIVE
    if custom_rollout_function == VERIFIER_ROLLOUT_FUNCTION:
        return RolloutSourceKind.VERIFIER
    return RolloutSourceKind.CUSTOM

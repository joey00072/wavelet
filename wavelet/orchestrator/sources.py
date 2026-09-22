from __future__ import annotations

from enum import StrEnum

VERIFIER_ROLLOUT_FUNCTION = "wavelet.orchestrator.verifiers:generate_rollouts"


class RolloutSourceKind(StrEnum):
    NATIVE = "native"
    VERIFIER = "verifier"
    CUSTOM = "custom"


def source_kind(custom_rollout_function: str | None) -> RolloutSourceKind:
    if custom_rollout_function is None:
        return RolloutSourceKind.NATIVE
    if custom_rollout_function == VERIFIER_ROLLOUT_FUNCTION:
        return RolloutSourceKind.VERIFIER
    return RolloutSourceKind.CUSTOM

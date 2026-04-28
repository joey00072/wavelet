from __future__ import annotations

from wavelet.configs.rl_config import RLConfig
from wavelet.entrypoints.rl_inference import (
    _next_exported_policy_step,
    _policy_step_to_load,
    _required_policy_step,
)


class _PolicyReceiver:
    def __init__(self, steps: list[int]) -> None:
        self.steps = steps

    def available_steps(self) -> list[int]:
        return self.steps


def _config() -> RLConfig:
    return RLConfig(orchestrator={"max_async_level": 1, "max_off_policy_steps": 8})


def test_required_policy_step_respects_async_window() -> None:
    config = _config()

    assert _required_policy_step(config, 0) == 0
    assert _required_policy_step(config, 1) == 0
    assert _required_policy_step(config, 2) == 1


def test_policy_selection_does_not_wait_for_current_rollout_step() -> None:
    policy_step = _policy_step_to_load(
        _config(),
        _PolicyReceiver([0, 1]),  # type: ignore[arg-type]
        rollout_step=2,
        loaded_policy_step=0,
    )

    assert policy_step == 1


def test_policy_selection_loads_newest_available_policy() -> None:
    policy_step = _policy_step_to_load(
        _config(),
        _PolicyReceiver([0, 1, 2, 3]),  # type: ignore[arg-type]
        rollout_step=3,
        loaded_policy_step=1,
    )

    assert policy_step == 3


def test_policy_selection_reuses_loaded_policy_inside_async_window() -> None:
    policy_step = _policy_step_to_load(
        _config(),
        _PolicyReceiver([0]),  # type: ignore[arg-type]
        rollout_step=1,
        loaded_policy_step=0,
    )

    assert policy_step is None


def test_policy_selection_waits_for_next_exported_step() -> None:
    config = RLConfig(
        orchestrator={"max_async_level": 2, "max_off_policy_steps": 8},
        policy_transfer={"export_every_steps": 2},
    )

    assert _next_exported_policy_step(config, 1) == 2
    assert _policy_step_to_load(
        config,
        _PolicyReceiver([0]),  # type: ignore[arg-type]
        rollout_step=3,
        loaded_policy_step=0,
    ) == 2

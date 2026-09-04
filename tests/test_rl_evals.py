from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.entrypoints import rl_evals


def test_standalone_evals_run_every_environment_once(monkeypatch) -> None:
    run_evals = AsyncMock()
    monkeypatch.setattr(rl_evals, "_run_evals_async", run_evals)
    config = RLConfig(
        max_steps=12,
        eval={"env": [{"id": "first"}, {"id": "second"}]},
    )

    asyncio.run(rl_evals.run_standalone_evals(config))

    call = run_evals.await_args
    effective_config = call.args[0]
    assert effective_config.max_steps == 0
    assert (
        effective_config.orchestrator.custom_rollout_function
        == "wavelet.orchestrator.verifiers:generate_rollouts"
    )
    assert [env.id for env in call.kwargs["envs"]] == ["first", "second"]
    assert call.kwargs["policy_step"] == 0
    assert call.kwargs["rollout_step"] == 0


def test_standalone_evals_require_environment() -> None:
    with pytest.raises(ValueError, match="at least one eval.env"):
        asyncio.run(rl_evals.run_standalone_evals(RLConfig(max_steps=0)))


def test_evals_main_forces_eval_only_config(monkeypatch) -> None:
    loaded = RLConfig(max_steps=0, eval={"env": [{"id": "demo"}]})
    captured: list[str] = []

    def fake_load_config(_config_type, argv):
        captured.extend(argv)
        return loaded

    monkeypatch.setattr(rl_evals, "load_config", fake_load_config)
    monkeypatch.setattr(rl_evals, "setup_config_logger", lambda *_args: None)
    monkeypatch.setattr(rl_evals, "finish_orchestrator_wandb", lambda: None)
    run = AsyncMock()
    monkeypatch.setattr(rl_evals, "run_standalone_evals", run)

    assert rl_evals.main(["@", "run.yaml"]) == 0
    assert captured == ["@", "run.yaml", "--max-steps", "0"]
    run.assert_awaited_once_with(loaded)

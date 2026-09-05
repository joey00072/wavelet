from __future__ import annotations

import asyncio
import json
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


def test_eval_rollouts_serialize_message_models_structurally(tmp_path) -> None:
    from wavelet.orchestrator.envs import _write_eval_rollouts

    class Message:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

        def model_dump(self, *, mode: str = "json", exclude_none: bool = False) -> dict:
            assert mode == "json" and exclude_none
            return {"role": self.role, "content": self.content}

    path = tmp_path / "evals" / "step-000000" / "env.jsonl"
    _write_eval_rollouts(
        path,
        [{"example_id": "a", "prompt": [Message("user", "hi")], "reward": 1.0}],
    )

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["prompt"] == [{"role": "user", "content": "hi"}]

from __future__ import annotations

import asyncio
import sys

from wavelet.configs.rl_config import RLConfig
from wavelet.monitor import finish_orchestrator_wandb, setup_config_logger
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.scheduler import _run_evals_async
from wavelet.utils.config import load_config

_VERIFIER_ROLLOUT_FUNCTION = "wavelet.orchestrator.verifiers:generate_rollouts"


def _standalone_config(config: RLConfig) -> RLConfig:
    """Return an eval-only config without requiring a training rollout source."""
    if config.eval is None or not config.eval.env:
        raise ValueError("Standalone evals require at least one eval.env entry.")
    orchestrator = config.orchestrator.model_copy(
        update={"custom_rollout_function": _VERIFIER_ROLLOUT_FUNCTION}
    )
    return config.model_copy(update={"orchestrator": orchestrator, "max_steps": 0})


async def run_standalone_evals(config: RLConfig) -> None:
    """Evaluate every configured environment once against the served model."""
    config = _standalone_config(config)
    await _run_evals_async(
        config,
        RLOrchestrator(config),
        policy_step=0,
        rollout_step=0,
        envs=config.eval.env,
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Force eval-only validation even when reusing a training config. Appending
    # the override makes the command authoritative if that file sets max_steps.
    config = load_config(RLConfig, [*argv, "--max-steps", "0"])
    setup_config_logger("evals", config)
    try:
        asyncio.run(run_standalone_evals(config))
    finally:
        finish_orchestrator_wandb()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

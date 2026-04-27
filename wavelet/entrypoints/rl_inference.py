from __future__ import annotations

import os
import sys
from time import perf_counter

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.policy import create_policy_inference_engine
from wavelet.orchestrator.queue import FileSystemPolicyReceiver
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.utils.config import load_config


def _perf_enabled() -> bool:
    return os.environ.get("WAVELET_PERF_LOG", "").lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    policy_receiver = FileSystemPolicyReceiver(
        config.output_dir,
        config.policy_transfer,
    )
    inference_engine = create_policy_inference_engine(config)
    inference_engine.setup()
    orchestrator = RLOrchestrator(config)
    target_step = config.max_steps or 1
    loaded_policy_step: int | None = None
    for step in range(target_step):
        step_started_at = perf_counter()
        should_wait_for_policy = (
            loaded_policy_step is None
            or step - loaded_policy_step > config.orchestrator.max_async_level
            or step - loaded_policy_step > config.orchestrator.max_off_policy_steps
        )
        if not should_wait_for_policy and step in policy_receiver.available_steps():
            should_wait_for_policy = True
        if should_wait_for_policy:
            wait_started_at = perf_counter()
            policy = policy_receiver.wait_for_step(step)
            wait_policy_seconds = perf_counter() - wait_started_at
            load_started_at = perf_counter()
            inference_engine.load_policy(policy.step_dir, step=policy.step)
            load_policy_seconds = perf_counter() - load_started_at
            loaded_policy_step = policy.step
        else:
            wait_policy_seconds = 0.0
            load_policy_seconds = 0.0
        publish_started_at = perf_counter()
        batch = orchestrator.publish(
            step=step,
            inference_engine=inference_engine,
        )
        publish_seconds = perf_counter() - publish_started_at
        total_seconds = perf_counter() - step_started_at
        if _perf_enabled():
            print(
                "WAVELET_PERF inference_step "
                f"step={step} wait_policy={wait_policy_seconds:.3f} "
                f"load_policy={load_policy_seconds:.3f} "
                f"publish={publish_seconds:.3f} total={total_seconds:.3f}",
                flush=True,
            )
        print(batch.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

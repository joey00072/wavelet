from __future__ import annotations

import sys

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.policy import create_policy_inference_engine
from wavelet.orchestrator.queue import FileSystemPolicyReceiver
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.utils.config import load_config


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
    for step in range(target_step):
        if step > 0 or config.policy_transfer.export_initial:
            policy = policy_receiver.wait_for_step(step)
            inference_engine.load_policy(policy.step_dir, step=policy.step)
        batch = orchestrator.publish(
            step=step,
            inference_engine=inference_engine,
        )
        print(batch.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

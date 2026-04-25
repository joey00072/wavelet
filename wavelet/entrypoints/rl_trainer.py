from __future__ import annotations

import sys

from wavelet.orchestrator.queue import FileSystemRolloutReceiver
from wavelet.configs.rl_config import RLConfig
from wavelet.trainer.rl_trainer import RLTrainer
from wavelet.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    trainer = RLTrainer(config)
    trainer.setup()
    if config.orchestrator.enabled:
        trainer.export_policy(step=trainer.step)
        receiver = FileSystemRolloutReceiver(
            config.output_dir,
            config.transport,
            start_step=trainer.step,
        )
        try:
            target_step = config.max_steps or 1
            while trainer.step < target_step:
                batch = receiver.wait()
                trainer.load_rollout_path(batch.path)
                trainer.train_until(trainer.step + 1)
                trainer.export_policy(step=trainer.step)
        except Exception:
            trainer.finalize(status="failed")
            raise
        trainer.finalize(status="completed")
    else:
        trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())

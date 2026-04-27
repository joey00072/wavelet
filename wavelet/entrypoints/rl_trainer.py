from __future__ import annotations

import os
import sys
from time import perf_counter

from wavelet.orchestrator.queue import FileSystemRolloutReceiver
from wavelet.configs.rl_config import RLConfig
from wavelet.trainer.rl_trainer import RLTrainer
from wavelet.utils.config import load_config


def _perf_enabled() -> bool:
    return os.environ.get("WAVELET_PERF_LOG", "").lower() in {"1", "true", "yes", "on"}


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
                loop_started_at = perf_counter()
                wait_started_at = perf_counter()
                batch = receiver.wait()
                wait_seconds = perf_counter() - wait_started_at
                load_started_at = perf_counter()
                trainer.load_rollout_path(batch.path)
                load_seconds = perf_counter() - load_started_at
                train_started_at = perf_counter()
                trainer.train_until(trainer.step + 1)
                train_seconds = perf_counter() - train_started_at
                export_started_at = perf_counter()
                trainer.export_policy(step=trainer.step)
                export_seconds = perf_counter() - export_started_at
                total_seconds = perf_counter() - loop_started_at
                if _perf_enabled():
                    print(
                        "WAVELET_PERF trainer_step "
                        f"step={trainer.step} wait_batch={wait_seconds:.3f} "
                        f"load_rollout={load_seconds:.3f} "
                        f"train={train_seconds:.3f} "
                        f"export_policy={export_seconds:.3f} total={total_seconds:.3f}",
                        flush=True,
                    )
        except Exception:
            trainer.finalize(status="failed")
            raise
        trainer.finalize(status="completed")
    else:
        trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())

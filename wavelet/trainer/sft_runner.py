from __future__ import annotations

import os
import sys
import time

from wavelet.configs.sft import SFTConfig
from wavelet.trainer.sft import SFTTrainer
from wavelet.utils.config import load_config
from wavelet.utils.pathing import (
    get_config_dir,
    resolve_resume_checkpoint,
    validate_output_dir,
)
from wavelet.utils.serialization import dump_yaml


def _distributed_local_rank() -> int | None:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return None
    return int(os.environ.get("LOCAL_RANK", "0"))


def _wait_for_main_config(config_path, *, timeout_seconds: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not config_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for rank 0 to write '{config_path}'."
            )
        time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    config = load_config(SFTConfig, argv)
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    local_rank = _distributed_local_rank()
    config_path = get_config_dir(config.output_dir) / "sft.yaml"
    if local_rank in {None, 0}:
        validate_output_dir(
            config.output_dir,
            resuming=resuming,
            clean=config.clean_output_dir,
        )
        if resuming:
            assert config.ckpt is not None
            resolve_resume_checkpoint(config.output_dir, config.ckpt.resume_step)
        dump_yaml(
            config_path,
            config.model_dump(mode="json", exclude_none=True),
        )
    else:
        _wait_for_main_config(config_path)

    if config.dry_run:
        print("Dry run - configuration loaded successfully")
        print(config.model_dump_json(indent=2))
        return 0

    trainer = SFTTrainer(config)
    trainer.setup()
    trainer.train()

    return 0


if __name__ == "__main__":
    sys.exit(main())

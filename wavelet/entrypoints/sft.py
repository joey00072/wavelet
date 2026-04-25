from __future__ import annotations

import sys

from wavelet.configs.sft import SFTConfig
from wavelet.trainer.sft import SFTTrainer
from wavelet.utils.config import load_config
from wavelet.utils.pathing import (
    get_config_dir,
    resolve_resume_checkpoint,
    validate_output_dir,
)
from wavelet.utils.serialization import dump_yaml


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    config = load_config(SFTConfig, argv)
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    validate_output_dir(
        config.output_dir,
        resuming=resuming,
        clean=config.clean_output_dir,
    )
    if resuming:
        assert config.ckpt is not None
        resolve_resume_checkpoint(config.output_dir, config.ckpt.resume_step)
    dump_yaml(
        get_config_dir(config.output_dir) / "sft.yaml",
        config.model_dump(mode="json", exclude_none=True),
    )

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

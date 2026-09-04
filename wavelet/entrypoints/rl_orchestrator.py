from __future__ import annotations

import sys

from wavelet.configs.rl_config import RLConfig
from wavelet.monitor import setup_config_logger
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    setup_config_logger("rl_orchestrator", config)
    published = RLOrchestrator(config).run(max_steps=config.max_steps)
    for batch in published:
        print(batch.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

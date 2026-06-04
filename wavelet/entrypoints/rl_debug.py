from __future__ import annotations

import importlib
import sys


DEBUG_COMMANDS = {
    "preflight": (
        "wavelet.entrypoints.rl_preflight_debug",
        "Validate cheap RL launch prerequisites",
    ),
    "inference": (
        "wavelet.entrypoints.rl_inference_debug",
        "Inspect and probe RL inference",
    ),
    "orchestrator": (
        "wavelet.entrypoints.rl_orchestrator_debug",
        "Inspect and benchmark RL orchestration",
    ),
    "trainer": (
        "wavelet.entrypoints.rl_trainer_debug",
        "Inspect trainer inputs and replay artifacts",
    ),
}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: wavelet debug <subcommand> [args]")
        print("Subcommands:")
        for command, (_, description) in DEBUG_COMMANDS.items():
            print(f"  {command:<14} {description}")
        return 1

    command = argv[0]
    if command not in DEBUG_COMMANDS:
        print(f"Unknown debug subcommand: {command}")
        return 1

    module_name, _ = DEBUG_COMMANDS[command]
    module = importlib.import_module(module_name)
    return module.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import importlib
import sys

PUBLIC_COMMANDS = {
    "rl": ("wavelet.entrypoints.rl_launcher", "Run reinforcement learning launcher"),
    "sft": ("wavelet.entrypoints.sft", "Run supervised fine-tuning"),
    "debug": ("wavelet.entrypoints.rl_debug", "Inspect and probe RL subsystems"),
    "evals": (
        "wavelet.entrypoints.rl_evals",
        "Evaluate a served policy without starting training",
    ),
    "convert-checkpoint": (
        "wavelet.tools.convert_checkpoint",
        "Convert a DCP checkpoint to Hugging Face safetensors",
    ),
    "convert-traces": (
        "wavelet.tools.convert_traces",
        "Convert trace JSONL files to a Hugging Face dataset",
    ),
    "benchmark": (
        "wavelet.tools.benchmark",
        "Run and compare training performance benchmarks",
    ),
    "rl-trainer": ("wavelet.entrypoints.rl_trainer", "Run RL trainer"),
    "rl-orchestrator": ("wavelet.entrypoints.rl_orchestrator", "Run RL orchestrator"),
    "rl-inference": (
        "wavelet.entrypoints.rl_inference",
        "Run RL inference annotation stage",
    ),
    "inference-server": (
        "wavelet.entrypoints.inference_server",
        "Run OpenAI-compatible inference server",
    ),
}

INTERNAL_COMMANDS = {
    "native-inference-server": (
        "wavelet.entrypoints.native_inference_server",
        "Run native vLLM inference server",
    ),
}

COMMANDS = PUBLIC_COMMANDS | INTERNAL_COMMANDS


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: wavelet <command> [args]")
        print("Commands:")
        for command, (_, description) in PUBLIC_COMMANDS.items():
            print(f"  {command:<22} {description}")
        return 1

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        return 1

    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    return module.main(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())

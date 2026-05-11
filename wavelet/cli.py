from __future__ import annotations

import importlib
import sys


PUBLIC_COMMANDS = {
    "rl": ("wavelet.entrypoints.rl_launcher", "Run reinforcement learning launcher"),
    "sft": ("wavelet.entrypoints.sft", "Run supervised fine-tuning"),
    "debug": ("wavelet.entrypoints.rl_debug", "Inspect and probe RL subsystems"),
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

HIDDEN_COMMANDS = {
    "inference-debug": (
        "wavelet.entrypoints.rl_inference_debug",
        "Inspect and probe RL inference",
    ),
    "orchestrator-debug": (
        "wavelet.entrypoints.rl_orchestrator_debug",
        "Inspect and benchmark RL orchestration",
    ),
    "native-inference-server": (
        "wavelet.entrypoints.native_inference_server",
        "Run native vLLM inference server",
    ),
    "rl-vllm-native-server": (
        "wavelet.entrypoints.native_inference_server",
        "Run native vLLM inference server",
    ),
    "rl-vllm-server": (
        "wavelet.entrypoints.inference_server",
        "Run OpenAI-compatible inference server",
    ),
    "rl-vllm-openai-server": (
        "wavelet.entrypoints.inference_server",
        "Run OpenAI-compatible inference server",
    ),
}

COMMANDS = PUBLIC_COMMANDS | HIDDEN_COMMANDS


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

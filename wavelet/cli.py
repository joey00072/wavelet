from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "rl": ("wavelet.entrypoints.rl_launcher", "Run reinforcement learning launcher"),
    "rl-trainer": ("wavelet.entrypoints.rl_trainer", "Run RL trainer"),
    "rl-orchestrator": ("wavelet.entrypoints.rl_orchestrator", "Run RL orchestrator"),
    "orchestrator-debug": (
        "wavelet.entrypoints.rl_orchestrator_debug",
        "Inspect and benchmark RL orchestration",
    ),
    "rl-inference": (
        "wavelet.entrypoints.rl_inference",
        "Run RL inference annotation stage",
    ),
    "inference-debug": (
        "wavelet.entrypoints.rl_inference_debug",
        "Inspect and probe RL inference",
    ),
    "rl-vllm-server": (
        "wavelet.entrypoints.rl_vllm_server",
        "Run persistent vLLM HTTP rollout server",
    ),
    "rl-vllm-openai-server": (
        "wavelet.entrypoints.rl_vllm_openai_server",
        "Run vLLM OpenAI rollout server",
    ),
    "sft": ("wavelet.entrypoints.sft", "Run supervised fine-tuning"),
}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: wavelet <command> [args]")
        print("Commands:")
        for command, (_, description) in COMMANDS.items():
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

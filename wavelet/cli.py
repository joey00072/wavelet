from __future__ import annotations

import sys

import wavelet.entrypoints.rl_launcher
import wavelet.entrypoints.rl_inference
import wavelet.entrypoints.rl_orchestrator
import wavelet.entrypoints.rl_trainer
import wavelet.entrypoints.rl_vllm_server
import wavelet.entrypoints.sft


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: wavelet <command> [args]")
        print("Commands:")
        print("  rl               Run reinforcement learning launcher")
        print("  rl-trainer       Run RL trainer")
        print("  rl-orchestrator  Run RL orchestrator")
        print("  rl-inference     Run RL inference annotation stage")
        print("  rl-vllm-server   Run persistent vLLM HTTP rollout server")
        print("  sft    Run supervised fine-tuning")
        return 1

    command = sys.argv[1]

    if command == "rl":
        return wavelet.entrypoints.rl_launcher.main(sys.argv[2:])
    if command == "rl-trainer":
        return wavelet.entrypoints.rl_trainer.main(sys.argv[2:])
    if command == "rl-orchestrator":
        return wavelet.entrypoints.rl_orchestrator.main(sys.argv[2:])
    if command == "rl-inference":
        return wavelet.entrypoints.rl_inference.main(sys.argv[2:])
    if command == "rl-vllm-server":
        return wavelet.entrypoints.rl_vllm_server.main(sys.argv[2:])
    if command == "sft":
        return wavelet.entrypoints.sft.main(sys.argv[2:])
    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Export native SWE task data using the environment's own dataset loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import verifiers.v1 as vf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = vf.SingleAgentEnvConfig.model_validate(
        json.loads(args.env_config.read_text())
    )
    tasks = vf.load_taskset(config.taskset)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        tasks = tasks.head(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w") as handle:
        for task in tasks:
            data = task.data.model_dump(mode="json")
            prompt = [{"role": "user", "content": data["prompt"]}]
            row = {
                "prompt": prompt,
                "completion": "",
                "metadata": {
                    "verifier_example": {
                        "prompt": prompt,
                        "example_id": data["idx"],
                        "info": {"native_task": data},
                    }
                },
            }
            handle.write(json.dumps(row) + "\n")
            count += 1
    print(f"Exported {count} tasks to {args.output}")


if __name__ == "__main__":
    main()

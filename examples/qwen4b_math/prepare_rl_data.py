from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qwen4b_math_data/rl_train.jsonl"),
    )
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-id", default="math-env")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("--question-key", default=None)
    parser.add_argument("--answer-key", default=None)
    parser.add_argument("--math-verify-max-workers", type=int, default=128)
    parser.add_argument("--math-verify-timeout", type=int, default=60)
    return parser.parse_args()


def env_args(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "dataset_name": args.dataset_name,
        "dataset_subset": args.dataset_subset,
        "question_key": args.question_key,
        "answer_key": args.answer_key,
        "math_verify_max_workers": args.math_verify_max_workers,
        "math_verify_timeout": args.math_verify_timeout,
    }
    return {key: value for key, value in values.items() if value is not None}


def main() -> int:
    args = parse_args()
    try:
        import verifiers as vf
    except ImportError as exc:
        raise SystemExit(
            "The Qwen4B math example uses verifier environments. Install them "
            "with `uv sync --extra verifiers --extra envs`."
        ) from exc

    env = vf.load_environment(args.env_id, **env_args(args))
    dataset = env.get_dataset(n=args.examples or -1, seed=args.seed)
    rows = []
    for index, example in enumerate(dataset):
        payload = dict(example)
        payload.setdefault("example_id", index)
        rows.append(
            {
                "prompt": payload["prompt"],
                "completion": "",
                "metadata": {
                    "verifier_example": payload,
                    "example_id": payload["example_id"],
                },
            }
        )

    if not rows:
        raise RuntimeError("Math verifier dataset returned no examples.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

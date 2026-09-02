from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/polaris_math_tagged_data/rl_train.jsonl"),
    )
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-difficulty", type=int, default=1)
    parser.add_argument("--max-difficulty", type=int, default=6)
    parser.add_argument(
        "--include-proof-problems",
        action="store_true",
        help=(
            "Include proof requests even though the final-answer rubric cannot "
            "validate proofs."
        ),
    )
    parser.add_argument(
        "--include-malformed-answers",
        action="store_true",
        help="Include structurally incomplete Polaris answer fragments.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import verifiers as vf
    except ImportError as exc:
        raise SystemExit(
            "This example requires Verifiers. Install the verifiers extra first."
        ) from exc

    env = vf.load_environment(
        "polaris-math-tagged",
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        exclude_proof_problems=not args.include_proof_problems,
        exclude_malformed_answers=not args.include_malformed_answers,
    )
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
        raise RuntimeError("Filtered Polaris returned no examples.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

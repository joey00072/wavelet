from __future__ import annotations

import argparse
from pathlib import Path

from wavelet.tools.verifier_data import import_verifiers, verifier_rows, write_jsonl


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
    vf = import_verifiers()
    env = vf.load_environment(
        "polaris-math-tagged",
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        exclude_proof_problems=not args.include_proof_problems,
        exclude_malformed_answers=not args.include_malformed_answers,
    )
    rows = verifier_rows(env.get_dataset(n=args.examples or -1, seed=args.seed))
    if not rows:
        raise RuntimeError("Filtered Polaris returned no examples.")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from wavelet.tools.verifier_data import import_verifiers, verifier_rows, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/alphabet_sort_data/rl_train.jsonl"),
    )
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help=(
            "Keep the verifier taskset's source order. Use this with an unshuffled "
            "Wavelet data config for task-for-task reference comparisons."
        ),
    )
    parser.add_argument("--env-id", default="primeintellect/alphabet-sort")
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--min-names-per-turn", type=int, default=None)
    parser.add_argument("--max-names-per-turn", type=int, default=None)
    parser.add_argument("--similarity-power", type=int, default=None)
    parser.add_argument("--power-per-turn", action="store_true")
    return parser.parse_args()


def env_args(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "min_turns": args.min_turns,
        "max_turns": args.max_turns,
        "min_names_per_turn": args.min_names_per_turn,
        "max_names_per_turn": args.max_names_per_turn,
        "similarity_power": args.similarity_power,
        "power_per_turn": args.power_per_turn,
    }
    return {key: value for key, value in values.items() if value is not None}


def main() -> int:
    args = parse_args()
    vf = import_verifiers()
    env = vf.load_environment(args.env_id, **env_args(args))
    dataset = env.get_dataset(seed=None if args.preserve_order else args.seed)
    rows = verifier_rows(dataset, limit=args.examples)
    if not rows:
        raise RuntimeError("Alphabet Sort verifier dataset returned no examples.")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

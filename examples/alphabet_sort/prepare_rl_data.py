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
        default=Path("outputs/alphabet_sort_data/rl_train.jsonl"),
    )
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    try:
        import verifiers as vf
    except ImportError as exc:
        raise SystemExit(
            "The Alphabet Sort example uses a verifier environment. "
            "Install it with `uv sync --extra verifiers --extra envs`."
        ) from exc

    env = vf.load_environment(args.env_id, **env_args(args))
    dataset = env.get_dataset(seed=args.seed)
    rows = []
    for index, example in enumerate(dataset):
        if args.examples is not None and index >= args.examples:
            break
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
        raise RuntimeError("Alphabet Sort verifier dataset returned no examples.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

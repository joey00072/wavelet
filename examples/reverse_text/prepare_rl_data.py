from __future__ import annotations

import argparse
from pathlib import Path

from wavelet.tools.verifier_data import import_verifiers, verifier_rows, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reverse_text_data/rl_train.jsonl"),
    )
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-id", default="reverse-text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vf = import_verifiers()
    env = vf.load_environment(args.env_id)
    dataset = env.get_dataset(seed=args.seed)
    rows = verifier_rows(dataset, limit=args.examples, id_keys=("id",))
    if not rows:
        raise RuntimeError("Reverse Text verifier dataset returned no examples.")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

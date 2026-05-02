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
        default=Path("outputs/reverse_text_data/rl_train.jsonl"),
    )
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-id", default="reverse-text")
    return parser.parse_args()


def _example_id(payload: dict[str, Any], index: int) -> Any:
    return payload.get("example_id", payload.get("id", index))


def main() -> int:
    args = parse_args()
    try:
        import verifiers as vf
    except ImportError as exc:
        raise SystemExit(
            "The Reverse Text example uses Prime verifier environments. "
            "Install them with `uv sync --extra verifiers --extra envs`."
        ) from exc

    env = vf.load_environment(args.env_id)
    dataset = env.get_dataset(seed=args.seed)
    rows = []
    for index, example in enumerate(dataset):
        if args.examples is not None and index >= args.examples:
            break
        payload = dict(example)
        payload.setdefault("example_id", _example_id(payload, index))
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
        raise RuntimeError("Reverse Text verifier dataset returned no examples.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

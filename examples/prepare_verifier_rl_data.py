from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wavelet.tools.verifier_data import import_verifiers, verifier_rows, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--env-arg",
        action="append",
        default=[],
        help="Environment kwarg as key=value. Values are parsed as JSON when possible.",
    )
    return parser.parse_args()


def parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_env_args(raw_args: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in raw_args:
        if "=" not in raw:
            raise ValueError(f"Expected --env-arg key=value, got {raw!r}.")
        key, value = raw.split("=", 1)
        parsed[key] = parse_value(value)
    return parsed


def get_dataset(env: Any, *, examples: int | None, seed: int) -> Iterable[Any]:
    n = examples if examples is not None else -1
    attempts = (
        {"n": n, "seed": seed},
        {"seed": seed},
        {"n": n},
        {},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return env.get_dataset(**kwargs)
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def main() -> int:
    args = parse_args()
    vf = import_verifiers()
    env = vf.load_environment(args.env_id, **parse_env_args(args.env_arg))
    dataset = get_dataset(env, examples=args.examples, seed=args.seed)
    rows = verifier_rows(dataset, limit=args.examples)
    if not rows:
        raise RuntimeError(f"{args.env_id} returned no examples.")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

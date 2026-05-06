from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


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
    try:
        import verifiers as vf
    except ImportError as exc:
        raise SystemExit(
            "Verifier data preparation requires `uv sync --extra verifiers --extra envs`."
        ) from exc

    env_args = parse_env_args(args.env_arg)
    env = vf.load_environment(args.env_id, **env_args)
    dataset = get_dataset(env, examples=args.examples, seed=args.seed)
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
        raise RuntimeError(f"{args.env_id} returned no examples.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} verifier examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

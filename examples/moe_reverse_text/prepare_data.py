from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from wavelet.tools.verifier_data import import_verifiers, verifier_rows, write_jsonl

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
SYSTEM_PROMPT = (
    "Return only the reversed text from the user message. "
    "Do not add explanations or extra formatting."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare synthetic SFT rows and verifier-backed RL rows."
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=Path("outputs/moe_reverse_text_data/sft_train.jsonl"),
    )
    parser.add_argument(
        "--rl-output",
        type=Path,
        default=Path("outputs/moe_reverse_text_data/rl_train.jsonl"),
    )
    parser.add_argument("--sft-examples", type=int, default=64)
    parser.add_argument("--rl-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-id", default="reverse-text")
    return parser.parse_args()


def _random_text(rng: random.Random) -> str:
    length = rng.randint(5, 18)
    chars = [rng.choice(ALPHABET) for _ in range(length)]
    if length > 8:
        chars.insert(rng.randint(2, length - 2), " ")
    return "".join(chars)


def build_sft_rows(*, examples: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    while len(rows) < examples:
        text = _random_text(rng)
        if text in seen:
            continue
        seen.add(text)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": text[::-1]},
                ]
            }
        )
    return rows


def build_rl_rows(*, examples: int, seed: int, env_id: str) -> list[dict[str, Any]]:
    env = import_verifiers().load_environment(env_id)
    rows = verifier_rows(env.get_dataset(seed=seed), limit=examples, id_keys=("id",))
    if not rows:
        raise RuntimeError("Reverse Text verifier dataset returned no examples.")
    return rows


def main() -> int:
    args = parse_args()
    sft_rows = build_sft_rows(examples=args.sft_examples, seed=args.seed)
    rl_rows = build_rl_rows(
        examples=args.rl_examples,
        seed=args.seed,
        env_id=args.env_id,
    )
    write_jsonl(args.sft_output, sft_rows)
    write_jsonl(args.rl_output, rl_rows)
    print(f"Wrote {len(sft_rows)} SFT rows to {args.sft_output}")
    print(f"Wrote {len(rl_rows)} RL rows to {args.rl_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {REASONING_START} and {REASONING_END}.
Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"""

CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    f"{{{{ '{SYSTEM_PROMPT}' + eos_token }}}}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    f"{{% if add_generation_prompt %}}{{{{ '{REASONING_START}' }}}}{{% endif %}}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Wavelet version of the Unsloth Qwen3 math example."
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=Path("outputs/unsloth_math_data/sft_train.jsonl"),
    )
    parser.add_argument(
        "--rl-output",
        type=Path,
        default=Path("outputs/unsloth_math_data/rl_train.jsonl"),
    )
    parser.add_argument("--model", default="unsloth/Qwen3-8B-Base")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-thought-tokens", type=int, default=384)
    parser.add_argument("--max-sft-examples", type=int, default=64)
    parser.add_argument("--max-rl-examples", type=int, default=512)
    parser.add_argument("--max-rl-prompt-tokens", type=int, default=1024)
    return parser.parse_args()


def is_number(value: object) -> bool:
    try:
        float(str(value).strip())
    except ValueError:
        return False
    return True


def truncate_text(tokenizer: AutoTokenizer, text: str, max_tokens: int) -> str:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True).strip()


def format_sft_messages(
    tokenizer: AutoTokenizer,
    row: dict[str, object],
    *,
    max_thought_tokens: int,
) -> list[dict[str, str]]:
    thoughts = str(row["generated_solution"])
    thoughts = thoughts.replace("<think>", "").replace("</think>", "").strip()
    thoughts = truncate_text(tokenizer, thoughts, max_thought_tokens)
    answer = str(row["expected_answer"]).strip()
    completion = (
        f"{REASONING_START}{thoughts}{REASONING_END}"
        f"{SOLUTION_START}{answer}{SOLUTION_END}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(row["problem"])},
        {"role": "assistant", "content": completion},
    ]


def prompt_token_count(
    tokenizer: AutoTokenizer,
    prompt: list[dict[str, str]],
) -> int:
    return len(
        tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def build_sft_rows(
    tokenizer: AutoTokenizer,
    *,
    seq_len: int,
    max_examples: int,
    max_thought_tokens: int,
) -> list[dict[str, object]]:
    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    max_tokens = seq_len // 2
    rows: list[dict[str, object]] = []
    for row in dataset:
        if not is_number(row["expected_answer"]):
            continue
        messages = format_sft_messages(
            tokenizer,
            row,
            max_thought_tokens=max_thought_tokens,
        )
        token_count = len(tokenizer.apply_chat_template(messages, tokenize=True))
        if token_count > max_tokens:
            continue
        rows.append({"messages": messages})
        if len(rows) >= max_examples:
            break
    return rows


def build_rl_rows(
    tokenizer: AutoTokenizer,
    *,
    max_examples: int,
    max_prompt_tokens: int,
) -> list[dict[str, object]]:
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    rows: list[dict[str, object]] = []
    for row in dataset:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["prompt"])},
        ]
        if prompt_token_count(tokenizer, prompt) > max_prompt_tokens:
            continue
        rows.append({"prompt": prompt, "solution": str(row["solution"]).strip()})
        if len(rows) >= max_examples:
            break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows prepared for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = CHAT_TEMPLATE

    sft_rows = build_sft_rows(
        tokenizer,
        seq_len=args.seq_len,
        max_examples=args.max_sft_examples,
        max_thought_tokens=args.max_thought_tokens,
    )
    rl_rows = build_rl_rows(
        tokenizer,
        max_examples=args.max_rl_examples,
        max_prompt_tokens=args.max_rl_prompt_tokens,
    )

    write_jsonl(args.sft_output, sft_rows)
    write_jsonl(args.rl_output, rl_rows)
    print(f"Wrote {len(sft_rows)} SFT rows to {args.sft_output}")
    print(f"Wrote {len(rl_rows)} RL rows to {args.rl_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

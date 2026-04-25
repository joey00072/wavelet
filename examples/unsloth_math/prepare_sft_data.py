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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/unsloth_math_data/sft_train.jsonl")
    )
    parser.add_argument(
        "--rl-output",
        type=Path,
        default=Path("outputs/unsloth_math_data/rl_train.jsonl"),
    )
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-thought-tokens", type=int, default=96)
    parser.add_argument("--max-examples", type=int, default=128)
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


def format_messages(
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


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = CHAT_TEMPLATE

    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    max_tokens = args.seq_len // 2
    sft_rows = []
    rl_rows = []
    for row in dataset:
        if not is_number(row["expected_answer"]):
            continue
        answer = str(row["expected_answer"]).strip()
        messages = format_messages(
            tokenizer,
            row,
            max_thought_tokens=args.max_thought_tokens,
        )
        token_count = len(tokenizer.apply_chat_template(messages, tokenize=True))
        if token_count > max_tokens:
            continue
        sft_rows.append({"messages": messages})
        rl_rows.append({"prompt": str(row["problem"]), "solution": answer})
        if len(sft_rows) >= args.max_examples:
            break

    if not sft_rows:
        raise RuntimeError("No SFT examples matched the numeric and length filters.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in sft_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.rl_output.parent.mkdir(parents=True, exist_ok=True)
    with args.rl_output.open("w", encoding="utf-8") as handle:
        for row in rl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sft_rows)} SFT examples to {args.output}")
    print(f"Wrote {len(rl_rows)} RL examples to {args.rl_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import AsyncOpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from environments.polaris_math_tagged.polaris_math_tagged import (
    AIME_2024_DATASET,
    AIME_2024_REVISION,
    POLARIS_DATASET,
    POLARIS_REVISION,
    SYSTEM_PROMPT,
    build_polaris_rows,
    extract_tagged_answer,
)

DEFAULT_OUTPUT_DIR = Path("outputs/polaris_synthetic_hard100_step208")
_THINK_BLOCK = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL)
_NUMBERED_REASONING_MARKERS = tuple(
    re.compile(rf"^\s*{number}\.\s+", re.MULTILINE) for number in (1, 2, 3)
)


@dataclass(frozen=True, slots=True)
class ClassifiedCompletion:
    status: str
    parsed_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and retain valid-format incorrect Polaris solutions."
    )
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--policy-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-completion-tokens", type=int, default=6144)
    parser.add_argument("--context-margin", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--verify-workers", type=int, default=16)
    parser.add_argument("--verify-timeout", type=int, default=60)
    return parser.parse_args()


def select_hard_examples(
    rows: Iterable[dict[str, Any]],
    *,
    held_out_problems: Iterable[str],
    difficulty: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select unique examples from one Polaris pass-count difficulty bucket."""
    candidates = build_polaris_rows(
        rows,
        held_out_problems=held_out_problems,
        min_difficulty=difficulty,
        max_difficulty=difficulty,
    )
    if len(candidates) < count:
        raise ValueError(
            f"Polaris difficulty {difficulty}/8 has {len(candidates)} eligible "
            f"examples, fewer than the requested {count}."
        )
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def classify_completion(
    completion: str,
    reference_answer: str,
    *,
    verifier: Callable[[str, str], bool],
) -> ClassifiedCompletion:
    """Classify strict-format responses before retention."""
    parsed_answer = extract_tagged_answer(completion)
    if not parsed_answer:
        return ClassifiedCompletion("format_invalid", "")
    if has_numbered_reasoning(completion):
        return ClassifiedCompletion("numbered_reasoning", parsed_answer)
    if verifier(parsed_answer, reference_answer):
        return ClassifiedCompletion("correct", parsed_answer)
    return ClassifiedCompletion("incorrect", parsed_answer)


def has_numbered_reasoning(completion: str) -> bool:
    """Return whether a think block contains line items 1., 2., and 3."""
    match = _THINK_BLOCK.search(completion)
    if match is None:
        return False
    think = match.group("think")
    return all(pattern.search(think) for pattern in _NUMBERED_REASONING_MARKERS)


def verify_math_answer(response: str, answer: str, timeout_seconds: int) -> bool:
    """Use the exact verifier helper used by Prime's MathRubric."""
    from verifiers.rubrics.math_rubric import verify_response

    reward, _elapsed = verify_response(
        response,
        answer,
        max_verify_chars=50_000,
        timeout_seconds=timeout_seconds,
    )
    return reward >= 1.0


def verify_pending_completion(
    item: tuple[dict[str, Any], int, str], timeout_seconds: int
) -> bool:
    """Verify one generated completion in a process-pool-safe function."""
    example, _generation_index, completion = item
    parsed_answer = extract_tagged_answer(completion)
    if not parsed_answer or has_numbered_reasoning(completion):
        return False
    return verify_math_answer(parsed_answer, str(example["answer"]), timeout_seconds)


def load_policy(server_url: str, policy_dir: Path, policy_step: int) -> None:
    payload = json.dumps(
        {
            "policy_dir": str(policy_dir.resolve()),
            "step": policy_step,
            "load_inplace": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/load_policy",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    if result.get("status") != "ok" or result.get("policy_step") != policy_step:
        raise RuntimeError(f"Policy load returned an unexpected response: {result}")


def prompt_token_count(tokenizer: PreTrainedTokenizerBase, question: str) -> int:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


async def generate_one_prompt(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, Any],
    *,
    example_index: int,
    args: argparse.Namespace,
) -> list[str]:
    question = str(example["question"])
    prompt_tokens = prompt_token_count(tokenizer, question)
    available_tokens = args.max_model_len - prompt_tokens - args.context_margin
    max_tokens = min(args.max_completion_tokens, available_tokens)
    if max_tokens < 128:
        raise ValueError(
            f"Prompt {example['example_id']} leaves only {max_tokens} completion "
            "tokens."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    async with semaphore:
        for attempt in range(args.request_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    n=args.generations,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=max_tokens,
                    seed=args.seed + example_index,
                    extra_body={
                        "stop": ["</answer>"],
                        "include_stop_str_in_output": True,
                    },
                )
                completions = [
                    choice.message.content or "" for choice in response.choices
                ]
                if len(completions) != args.generations:
                    raise RuntimeError(
                        f"Expected {args.generations} choices, got {len(completions)}."
                    )
                return completions
            except Exception:
                if attempt >= args.request_retries:
                    raise
                await asyncio.sleep(2**attempt)
    raise AssertionError("generation retry loop exited unexpectedly")


async def generate_all(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> list[list[str] | Exception]:
    client = AsyncOpenAI(
        base_url=f"{args.server_url.rstrip('/')}/v1",
        api_key="EMPTY",
        timeout=args.request_timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    try:
        tasks = [
            generate_one_prompt(
                client,
                semaphore,
                tokenizer,
                example,
                example_index=index,
                args=args,
            )
            for index, example in enumerate(examples)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.close()


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_aime = load_dataset(
        AIME_2024_DATASET,
        split="train",
        revision=AIME_2024_REVISION,
        trust_remote_code=False,
    )
    raw_polaris = load_dataset(
        POLARIS_DATASET,
        split="train",
        revision=POLARIS_REVISION,
        trust_remote_code=False,
    )
    return select_hard_examples(
        (dict(row) for row in raw_polaris),
        held_out_problems=(str(row["problem"]) for row in raw_aime),
        difficulty=args.difficulty,
        count=args.examples,
        seed=args.seed,
    )


def read_policy_metadata(policy_dir: Path) -> dict[str, Any]:
    metadata_path = policy_dir / "policy.json"
    if not metadata_path.is_file() or not (policy_dir / "STABLE").is_file():
        raise FileNotFoundError(
            f"Stable policy metadata was not found under {policy_dir}."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_outputs(
    examples: list[dict[str, Any]],
    generated: list[list[str] | Exception],
    *,
    policy_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "incorrect.jsonl"
    pending: list[tuple[dict[str, Any], int, str]] = []
    counts: Counter[str] = Counter()
    request_errors: list[dict[str, str]] = []

    for example, result in zip(examples, generated, strict=True):
        if isinstance(result, Exception):
            counts["request_failed"] += args.generations
            request_errors.append(
                {"example_id": str(example["example_id"]), "error": repr(result)}
            )
            continue
        counts["generated"] += len(result)
        pending.extend(
            (example, generation_index, completion)
            for generation_index, completion in enumerate(result)
        )

    with ProcessPoolExecutor(max_workers=args.verify_workers) as executor:
        verified = list(
            executor.map(
                verify_pending_completion,
                pending,
                repeat(args.verify_timeout),
            )
        )

    retained: list[dict[str, Any]] = []
    for (example, generation_index, completion), is_correct in zip(
        pending, verified, strict=True
    ):
        classification = classify_completion(
            completion,
            str(example["answer"]),
            verifier=lambda _response, _answer, result=is_correct: result,
        )
        counts[classification.status] += 1
        if classification.status != "incorrect":
            continue
        retained.append(
            {
                "id": f"{example['example_id']}-sample-{generation_index}",
                "source": POLARIS_DATASET,
                "difficulty": example["info"]["difficulty"],
                "question": example["question"],
                "reference_answer": example["answer"],
                "completion": completion,
                "parsed_answer": classification.parsed_answer,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": completion},
                ],
                "generation_index": generation_index,
                "generation_seed": args.seed,
                "policy_step": args.policy_step,
                "policy_kind": policy_metadata.get("kind"),
                "base_model": args.base_model,
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": POLARIS_DATASET,
        "difficulty": f"{args.difficulty}/8",
        "selected_prompts": len(examples),
        "generations_per_prompt": args.generations,
        "requested_completions": len(examples) * args.generations,
        "generated_completions": counts["generated"],
        "retained_incorrect": counts["incorrect"],
        "rejected_correct": counts["correct"],
        "rejected_format_invalid": counts["format_invalid"],
        "rejected_numbered_reasoning": counts["numbered_reasoning"],
        "request_failed": counts["request_failed"],
        "request_errors": request_errors,
        "policy_step": args.policy_step,
        "policy_dir": str(args.policy_dir.resolve()),
        "base_model": args.base_model,
        "output": str(output_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    if not 1 <= args.difficulty <= 8:
        raise ValueError("--difficulty must be between 1 and 8.")
    if args.examples <= 0 or args.generations <= 0:
        raise ValueError("--examples and --generations must be positive.")

    policy_metadata = read_policy_metadata(args.policy_dir)
    if int(policy_metadata.get("step", -1)) != args.policy_step:
        raise ValueError(
            f"Policy metadata step {policy_metadata.get('step')} does not match "
            f"--policy-step {args.policy_step}."
        )
    examples = load_examples(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    load_policy(args.server_url, args.policy_dir, args.policy_step)
    generated = asyncio.run(generate_all(examples, tokenizer, args))
    summary = write_outputs(
        examples,
        generated,
        policy_metadata=policy_metadata,
        args=args,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from environments.polaris_math_tagged.polaris_math_tagged import (
    POLARIS_DATASET,
    SYSTEM_PROMPT,
    extract_tagged_answer,
)
from examples.qwen2_5_7b_polaris.generate_incorrect_synthetic import (
    has_numbered_reasoning,
    load_policy,
    read_policy_metadata,
    verify_math_answer,
)

RECOVERY_PHRASES = (
    "Alternatively,",
    "Wait,",
)
DEFAULT_SOURCE = Path("outputs/polaris_incorrect_synthetic/incorrect.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/polaris_wait_recoveries")
_THINK_BLOCK = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ContinuationPrefix:
    source_id: str
    question: str
    reference_answer: str
    source_completion: str
    cut_character_index: int
    cut_separator: str
    recovery_phrase: str
    assistant_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verified-correct continuations from interrupted traces."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--policy-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument(
        "--recovery-phrases",
        nargs="+",
        default=list(RECOVERY_PHRASES),
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--concurrency", type=int, default=16)
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


def truncate_think_at_midpoint(
    completion: str, *, recovery_phrase: str
) -> tuple[str, int, str] | None:
    """Replace a think-block suffix at the line boundary nearest its midpoint."""
    match = _THINK_BLOCK.search(completion)
    if match is None:
        return None
    think = match.group("think")
    midpoint = len(think) // 2
    boundaries: dict[int, str] = {
        newline.end(): "\\n" for newline in re.finditer(r"(?<!\n)\n(?!\n)", think)
    }
    for paragraph in re.finditer(r"\n{2,}", think):
        boundaries[paragraph.end()] = "\\n\\n"
    if not boundaries:
        return None
    cut, separator = min(
        boundaries.items(),
        key=lambda item: (
            abs(item[0] - midpoint),
            0 if item[1] == "\\n\\n" else 1,
            item[0],
        ),
    )
    return f"<think>{think[:cut]}{recovery_phrase} ", cut, separator


def build_midpoint_prefixes(
    rows: list[dict[str, Any]], recovery_phrases: tuple[str, ...]
) -> tuple[list[ContinuationPrefix], Counter[str]]:
    """Build one midpoint continuation prefix per trace and recovery phrase."""
    prefixes: list[ContinuationPrefix] = []
    counts: Counter[str] = Counter()
    for row in rows:
        for phrase in recovery_phrases:
            truncated = truncate_think_at_midpoint(
                str(row["completion"]), recovery_phrase=phrase
            )
            if truncated is None:
                counts["skipped_missing_midpoint_boundary"] += 1
                continue
            assistant_prefix, cut_character_index, cut_separator = truncated
            prefixes.append(
                ContinuationPrefix(
                    source_id=str(row["id"]),
                    question=str(row["question"]),
                    reference_answer=str(row["reference_answer"]),
                    source_completion=str(row["completion"]),
                    cut_character_index=cut_character_index,
                    cut_separator=cut_separator,
                    recovery_phrase=phrase,
                    assistant_prefix=assistant_prefix,
                )
            )
            counts[f"phrase/{phrase}"] += 1
            counts[f"separator/{cut_separator}"] += 1
    return prefixes, counts


def render_prompt(
    tokenizer: PreTrainedTokenizerBase, prefix: ContinuationPrefix
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prefix.question},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return f"{rendered}{prefix.assistant_prefix}"


async def generate_one_prefix(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    tokenizer: PreTrainedTokenizerBase,
    prefix: ContinuationPrefix,
    *,
    prefix_index: int,
    args: argparse.Namespace,
) -> list[str]:
    prompt = render_prompt(tokenizer, prefix)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    available_tokens = args.max_model_len - prompt_tokens - args.context_margin
    max_tokens = min(args.max_completion_tokens, available_tokens)
    if max_tokens < 128:
        raise ValueError(
            f"Prefix {prefix.source_id}/midpoint leaves only "
            f"{max_tokens} completion tokens."
        )

    async with semaphore:
        for attempt in range(args.request_retries + 1):
            try:
                response = await client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    n=args.rollouts,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=max_tokens,
                    seed=args.seed + prefix_index,
                    extra_body={
                        "stop": ["</answer>"],
                        "include_stop_str_in_output": True,
                    },
                )
                suffixes = [choice.text for choice in response.choices]
                if len(suffixes) != args.rollouts:
                    raise RuntimeError(
                        f"Expected {args.rollouts} choices, got {len(suffixes)}."
                    )
                return [f"{prefix.assistant_prefix}{suffix}" for suffix in suffixes]
            except Exception:
                if attempt >= args.request_retries:
                    raise
                await asyncio.sleep(2**attempt)
    raise AssertionError("generation retry loop exited unexpectedly")


async def generate_all(
    prefixes: list[ContinuationPrefix],
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
            generate_one_prefix(
                client,
                semaphore,
                tokenizer,
                prefix,
                prefix_index=index,
                args=args,
            )
            for index, prefix in enumerate(prefixes)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.close()


def verify_recovery_candidate(
    item: tuple[ContinuationPrefix, int, str], timeout_seconds: int
) -> bool:
    prefix, _rollout_index, completion = item
    parsed_answer = extract_tagged_answer(completion)
    if not parsed_answer or has_numbered_reasoning(completion):
        return False
    if prefix.recovery_phrase not in completion:
        return False
    return verify_math_answer(parsed_answer, prefix.reference_answer, timeout_seconds)


def write_outputs(
    prefixes: list[ContinuationPrefix],
    generated: list[list[str] | Exception],
    *,
    prefix_counts: Counter[str],
    source_count: int,
    policy_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "correct_recoveries.jsonl"
    counts: Counter[str] = Counter(prefix_counts)
    request_errors: list[dict[str, str | int]] = []
    pending: list[tuple[ContinuationPrefix, int, str]] = []

    for prefix, result in zip(prefixes, generated, strict=True):
        if isinstance(result, Exception):
            counts["request_failed"] += args.rollouts
            request_errors.append(
                {
                    "source_id": prefix.source_id,
                    "cut": prefix.cut_character_index,
                    "error": repr(result),
                }
            )
            continue
        counts["generated"] += len(result)
        pending.extend(
            (prefix, rollout_index, completion)
            for rollout_index, completion in enumerate(result)
        )

    with ProcessPoolExecutor(max_workers=args.verify_workers) as executor:
        verified = list(
            executor.map(
                verify_recovery_candidate,
                pending,
                repeat(args.verify_timeout),
            )
        )

    retained: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for (prefix, rollout_index, completion), is_correct in zip(
        pending, verified, strict=True
    ):
        parsed_answer = extract_tagged_answer(completion)
        if not parsed_answer:
            counts["rejected_format_invalid"] += 1
            continue
        if has_numbered_reasoning(completion):
            counts["rejected_numbered_reasoning"] += 1
            continue
        if prefix.recovery_phrase not in completion:
            counts["rejected_missing_phrase"] += 1
            continue
        if not is_correct:
            counts["rejected_incorrect"] += 1
            continue
        dedupe_key = (prefix.question, completion)
        if dedupe_key in seen:
            counts["rejected_duplicate"] += 1
            continue
        seen.add(dedupe_key)
        counts["retained_correct"] += 1
        cut_label = f"midpoint-{prefix.cut_character_index}"
        phrase_slug = re.sub(r"[^a-z0-9]+", "-", prefix.recovery_phrase.casefold())
        retained.append(
            {
                "id": (
                    f"{prefix.source_id}-{cut_label}-{phrase_slug.strip('-')}-"
                    f"rollout-{rollout_index}"
                ),
                "source": POLARIS_DATASET,
                "source_trace_id": prefix.source_id,
                "question": prefix.question,
                "reference_answer": prefix.reference_answer,
                "completion": completion,
                "parsed_answer": parsed_answer,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prefix.question},
                    {"role": "assistant", "content": completion},
                ],
                "cut_mode": "midpoint",
                "cut_character_index": prefix.cut_character_index,
                "cut_separator": prefix.cut_separator,
                "recovery_phrase": prefix.recovery_phrase,
                "assistant_prefix": prefix.assistant_prefix,
                "rollout_index": rollout_index,
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
        "source_dataset": str(args.source.resolve()),
        "source_traces": source_count,
        "eligible_prefixes": len(prefixes),
        "cut_mode": "midpoint",
        "recovery_phrases": args.recovery_phrases,
        "rollouts_per_prefix": args.rollouts,
        "requested_completions": len(prefixes) * args.rollouts,
        "generated_completions": counts["generated"],
        "retained_correct": counts["retained_correct"],
        "rejected_incorrect": counts["rejected_incorrect"],
        "rejected_format_invalid": counts["rejected_format_invalid"],
        "rejected_numbered_reasoning": counts["rejected_numbered_reasoning"],
        "rejected_missing_phrase": counts["rejected_missing_phrase"],
        "rejected_duplicate": counts["rejected_duplicate"],
        "request_failed": counts["request_failed"],
        "request_errors": request_errors,
        "skipped_missing_cut": {
            "midpoint_boundary": counts["skipped_missing_midpoint_boundary"]
        },
        "prefixes_per_phrase": {
            phrase: counts[f"phrase/{phrase}"] for phrase in args.recovery_phrases
        },
        "prefixes_per_separator": {
            separator: counts[f"separator/{separator}"]
            for separator in ("\\n", "\\n\\n")
        },
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
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive.")
    if not args.recovery_phrases or any(
        not phrase.strip() for phrase in args.recovery_phrases
    ):
        raise ValueError("--recovery-phrases values must be non-empty.")
    policy_metadata = read_policy_metadata(args.policy_dir)
    if int(policy_metadata.get("step", -1)) != args.policy_step:
        raise ValueError(
            f"Policy metadata step {policy_metadata.get('step')} does not match "
            f"--policy-step {args.policy_step}."
        )
    rows = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phrases = tuple(args.recovery_phrases)
    prefixes, prefix_counts = build_midpoint_prefixes(rows, phrases)
    if not prefixes:
        raise RuntimeError("No source traces had the requested newline cut positions.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    load_policy(args.server_url, args.policy_dir, args.policy_step)
    generated = asyncio.run(generate_all(prefixes, tokenizer, args))
    summary = write_outputs(
        prefixes,
        generated,
        prefix_counts=prefix_counts,
        source_count=len(rows),
        policy_metadata=policy_metadata,
        args=args,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

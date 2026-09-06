from __future__ import annotations

import os
import re
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import verifiers as vf


POLARIS_DATASET = "POLARIS-Project/Polaris-Dataset-53K"
POLARIS_REVISION = "296f8e34132e63f4a1d70e0dcc8bddebb43f03e4"
AIME_2024_DATASET = "HuggingFaceH4/aime_2024"
AIME_2024_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"

SYSTEM_PROMPT = """You solve competition mathematics problems.

Respond using exactly this structure:
<think>your step-by-step reasoning</think>
<answer>only the final answer</answer>

Both tags are required. Do not put any text outside these tags.
"""

_TAGGED_RESPONSE = re.compile(
    r"\A\s*<think>\s*(?P<think>.+?)\s*</think>\s*"
    r"<answer>\s*(?P<answer>.+?)\s*</answer>\s*\Z",
    re.DOTALL,
)
_DIFFICULTY = re.compile(r"\A([0-8])/8\Z")
_PROOF_REQUEST = re.compile(
    r"\b(?:prove|demonstrate|establish)\b|\bshow\s+that\b",
    re.IGNORECASE,
)
_MALFORMED_ANSWER = re.compile(
    r"(?:\A\s*\^|\\frac\s*\{\s*\}|"
    r"\\frac\s*\{[^{}]*\}\s*\{\s*\}|\A\s*[+*/=^_-]+\s*\Z)"
)


def extract_tagged_answer(text: str) -> str:
    """Return the answer only when the full response follows the tag contract."""
    if any(
        text.count(tag) != 1 for tag in ("<think>", "</think>", "<answer>", "</answer>")
    ):
        return ""
    match = _TAGGED_RESPONSE.fullmatch(text)
    return "" if match is None else match.group("answer").strip()


def normalize_problem(problem: object) -> str:
    """Normalize problem text for deterministic deduplication and decontamination."""
    return " ".join(str(problem).split()).casefold()


def difficulty_numerator(value: object) -> int | None:
    """Parse Polaris pass-count labels such as ``3/8``."""
    match = _DIFFICULTY.fullmatch(str(value).strip())
    return None if match is None else int(match.group(1))


def is_proof_problem(problem: object) -> bool:
    """Return whether a prompt asks for a proof instead of a final answer."""
    return _PROOF_REQUEST.search(str(problem)) is not None


def is_malformed_answer(answer: object) -> bool:
    """Detect narrow, structurally incomplete Polaris target fragments."""
    return _MALFORMED_ANSWER.search(str(answer)) is not None


def format_aime_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format the pinned Prime AIME 2024 task source for legacy Verifiers."""
    return [
        {
            "question": str(row["problem"]),
            "answer": str(int(row["answer"])),
            "example_id": f"aime2024-{index:02d}",
            "info": {"source": AIME_2024_DATASET},
        }
        for index, row in enumerate(rows)
    ]


def build_polaris_rows(
    rows: Iterable[dict[str, Any]],
    *,
    held_out_problems: Iterable[object] = (),
    min_difficulty: int = 1,
    max_difficulty: int = 6,
    exclude_proof_problems: bool = True,
    exclude_malformed_answers: bool = True,
) -> list[dict[str, Any]]:
    """Filter, deduplicate, and decontaminate Polaris training rows.

    Proof requests are excluded by default because a final-answer equivalence
    rubric cannot evaluate whether the proof is valid. Several such Polaris
    rows also contain answer fragments copied or corrupted from the claim.
    """
    if not 0 <= min_difficulty <= max_difficulty <= 8:
        raise ValueError("difficulty bounds must satisfy 0 <= min <= max <= 8")

    held_out = {normalize_problem(problem) for problem in held_out_problems}
    seen: set[str] = set()
    formatted: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        difficulty = difficulty_numerator(row.get("difficulty"))
        if difficulty is None or not min_difficulty <= difficulty <= max_difficulty:
            continue
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        normalized = normalize_problem(problem)
        if (
            not normalized
            or not answer
            or normalized in held_out
            or normalized in seen
            or (exclude_proof_problems and is_proof_problem(problem))
            or (exclude_malformed_answers and is_malformed_answer(answer))
        ):
            continue
        seen.add(normalized)
        formatted.append(
            {
                "question": problem,
                "answer": answer,
                "example_id": f"polaris-{source_index:05d}",
                "info": {
                    "source": POLARIS_DATASET,
                    "difficulty": f"{difficulty}/8",
                },
            }
        )
    return formatted


def build_resilient_math_rubric(
    vf: Any,
    *,
    parser: Any,
    max_workers: int,
    timeout_seconds: float,
) -> Any:
    """Build a math rubric that replaces a poisoned verifier process pool."""
    from verifiers.utils.thread_utils import (
        register_executor,
        unregister_executor,
    )

    hard_timeout_seconds = max(15.0, timeout_seconds * 2.0)

    class ResilientMathRubric(vf.MathRubric):
        HARD_TIMEOUT_SECONDS = hard_timeout_seconds

        def _replace_executor(self, failed_executor: ProcessPoolExecutor) -> None:
            if self.executor is not failed_executor:
                return

            unregister_executor(self.executor_name)
            broken = bool(getattr(failed_executor, "_broken", False))
            if broken:
                # Nothing in a broken pool can complete; reap the workers.
                processes = list(
                    (getattr(failed_executor, "_processes", None) or {}).values()
                )
                for process in processes:
                    process.kill()
                for process in processes:
                    process.join(timeout=5)
            # Let queued verifications on the old pool finish instead of
            # cancelling them, which would score every in-flight rollout 0.0.
            failed_executor.shutdown(wait=False, cancel_futures=broken)

            self.executor = ProcessPoolExecutor(max_workers=1)
            register_executor(
                self.executor_name,
                self.executor,
                scaling_fn=lambda concurrency: min(
                    max(1, concurrency // 128),
                    min(max_workers, os.cpu_count() or 1),
                ),
            )
            self.logger.warning(
                "Replaced math verification process pool after %s.",
                "a broken worker" if broken else "a hard timeout",
            )

        async def correct_answer(self, *args: Any, **kwargs: Any) -> float:
            executor = self.executor
            started_at = perf_counter()
            reward = await super().correct_answer(*args, **kwargs)
            elapsed = perf_counter() - started_at
            # MathRubric swallows its asyncio.wait_for timeout and returns 0.0,
            # so the hard timeout can only be observed through elapsed time.
            # Slow-but-completed calls (queue wait under load) must not
            # discard a healthy pool.
            timed_out = elapsed >= self.HARD_TIMEOUT_SECONDS
            if timed_out or getattr(executor, "_broken", False):
                self._replace_executor(executor)
            return reward

    return ResilientMathRubric(
        parser=parser,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
    )


def load_environment(
    min_difficulty: int = 1,
    max_difficulty: int = 6,
    math_verify_max_workers: int = 128,
    math_verify_timeout: float = 5.0,
    exclude_proof_problems: bool = True,
    exclude_malformed_answers: bool = True,
) -> vf.Environment:
    """Load filtered Polaris for training and held-out AIME 2024 for evaluation."""
    import verifiers as vf
    from datasets import Dataset, load_dataset

    raw_aime = load_dataset(
        AIME_2024_DATASET,
        split="train",
        revision=AIME_2024_REVISION,
        trust_remote_code=False,
    )
    aime_rows = format_aime_rows(dict(row) for row in raw_aime)
    raw_polaris = load_dataset(
        POLARIS_DATASET,
        split="train",
        revision=POLARIS_REVISION,
        trust_remote_code=False,
    )
    polaris_rows = build_polaris_rows(
        (dict(row) for row in raw_polaris),
        held_out_problems=(row["question"] for row in aime_rows),
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
        exclude_proof_problems=exclude_proof_problems,
        exclude_malformed_answers=exclude_malformed_answers,
    )
    if not polaris_rows:
        raise RuntimeError("Polaris filtering returned no training examples.")

    parser = vf.Parser(extract_fn=extract_tagged_answer)
    rubric = build_resilient_math_rubric(
        vf,
        parser=parser,
        max_workers=math_verify_max_workers,
        timeout_seconds=math_verify_timeout,
    )
    return vf.SingleTurnEnv(
        dataset=Dataset.from_list(polaris_rows),
        eval_dataset=Dataset.from_list(aime_rows),
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import verifiers as vf


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


def extract_tagged_answer(text: str) -> str:
    """Return the answer only when the complete response follows the tag contract."""
    if any(
        text.count(tag) != 1 for tag in ("<think>", "</think>", "<answer>", "</answer>")
    ):
        return ""
    match = _TAGGED_RESPONSE.fullmatch(text)
    if match is None:
        return ""
    return match.group("answer").strip()


def format_example(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one HuggingFace MATH-500 test row for Verifiers."""
    return {
        "question": str(row["problem"]),
        "answer": str(row["answer"]),
        "example_id": str(row["unique_id"]),
        "info": {
            "subject": str(row["subject"]),
            "level": int(row["level"]),
        },
    }


def load_environment(
    math_verify_max_workers: int = 128,
    math_verify_timeout: float = 60.0,
) -> vf.Environment:
    """Load MATH-500 with strict think/answer parsing for train and eval."""
    import verifiers as vf
    from datasets import Dataset, load_dataset

    raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [format_example(dict(row)) for row in raw]
    dataset = Dataset.from_list(rows)
    parser = vf.Parser(extract_fn=extract_tagged_answer)
    rubric = vf.MathRubric(
        parser=parser,
        max_workers=math_verify_max_workers,
        timeout_seconds=math_verify_timeout,
    )
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )

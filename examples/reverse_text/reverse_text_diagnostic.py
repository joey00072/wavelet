"""Short-response reverse-text environment with partial-credit scoring."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import verifiers as vf

SYSTEM = (
    "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
)
TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


def lcs(completion: str | list[dict[str, str]], answer: str, **kwargs: object) -> float:
    """Score the tagged reversal using the canonical SequenceMatcher ratio."""
    text = (
        completion
        if isinstance(completion, str)
        else (completion[-1].get("content", "") if completion else "")
    )
    match = TAG.search(text)
    response = match.group(1).strip() if match else ""
    return SequenceMatcher(None, response, answer).ratio()


def load_environment(**kwargs: object) -> vf.SingleTurnEnv:
    """Load the canonical 1,000-example reverse-text training distribution."""
    import verifiers as vf
    from datasets import Dataset, load_dataset

    raw = load_dataset("PrimeIntellect/Reverse-Text-RL", split="train")
    data = Dataset.from_list(
        [
            {
                "question": row["prompt"],
                "answer": row["prompt"][::-1],
                "example_id": index,
            }
            for index, row in enumerate(raw)
        ]
    )
    return vf.SingleTurnEnv(
        dataset=data,
        eval_dataset=data,
        system_prompt=SYSTEM,
        rubric=vf.Rubric(funcs=[lcs], weights=[1.0]),
    )

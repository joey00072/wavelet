from __future__ import annotations

from examples.qwen2_5_7b_polaris.generate_incorrect_synthetic import (
    classify_completion,
    has_numbered_reasoning,
    select_hard_examples,
)


def test_select_hard_examples_filters_deduplicates_and_is_deterministic() -> None:
    rows = [
        {"problem": "Easy", "answer": "1", "difficulty": "7/8"},
        {"problem": "Hard A", "answer": "2", "difficulty": "1/8"},
        {"problem": " hard   a ", "answer": "3", "difficulty": "1/8"},
        {"problem": "Held out", "answer": "4", "difficulty": "1/8"},
        {"problem": "Hard B", "answer": "5", "difficulty": "1/8"},
        {"problem": "Hard C", "answer": "6", "difficulty": "1/8"},
    ]

    selected = select_hard_examples(
        rows,
        held_out_problems=["held OUT"],
        difficulty=1,
        count=2,
        seed=7,
    )

    assert selected == select_hard_examples(
        rows,
        held_out_problems=["held OUT"],
        difficulty=1,
        count=2,
        seed=7,
    )
    assert len({row["question"].casefold() for row in selected}) == 2
    assert {row["info"]["difficulty"] for row in selected} == {"1/8"}


def test_classify_completion_rejects_invalid_format_before_verification() -> None:
    verifier_called = False

    def verifier(_response: str, _answer: str) -> bool:
        nonlocal verifier_called
        verifier_called = True
        return False

    result = classify_completion(
        "<think>reasoning</think><aswer>wrong tag</aswer>",
        "42",
        verifier=verifier,
    )

    assert result.status == "format_invalid"
    assert not verifier_called


def test_classify_completion_rejects_correct_and_keeps_incorrect() -> None:
    correct = classify_completion(
        "<think>reasoning</think><answer>42</answer>",
        "42",
        verifier=lambda response, answer: response == answer,
    )
    incorrect = classify_completion(
        "<think>reasoning</think><answer>41</answer>",
        "42",
        verifier=lambda response, answer: response == answer,
    )

    assert correct.status == "correct"
    assert incorrect.status == "incorrect"
    assert incorrect.parsed_answer == "41"


def test_numbered_reasoning_requires_all_three_line_markers() -> None:
    numbered = """<think>
1. First step
2. Second step
3. Third step
</think><answer>41</answer>"""
    partial = """<think>
1. First step
2. Second step
</think><answer>41</answer>"""
    decimal = "<think>The value is 1.25, then 2.5 and 3.75.</think><answer>41</answer>"

    assert has_numbered_reasoning(numbered)
    assert not has_numbered_reasoning(partial)
    assert not has_numbered_reasoning(decimal)
    assert (
        classify_completion(
            numbered,
            "42",
            verifier=lambda _response, _answer: False,
        ).status
        == "numbered_reasoning"
    )

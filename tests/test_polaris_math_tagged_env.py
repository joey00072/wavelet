from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_environment_module() -> ModuleType:
    path = Path("environments/polaris_math_tagged/polaris_math_tagged.py")
    spec = importlib.util.spec_from_file_location("polaris_math_tagged", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENV = _load_environment_module()


def test_extract_tagged_answer_requires_exact_complete_structure() -> None:
    assert ENV.extract_tagged_answer("<think>work</think><answer>42</answer>") == "42"
    assert ENV.extract_tagged_answer("<answer>42</answer>") == ""
    assert ENV.extract_tagged_answer("<think>x</think><answer>42</answer>tail") == ""


def test_build_polaris_rows_filters_deduplicates_and_decontaminates() -> None:
    rows = [
        {"problem": "below range", "answer": "0", "difficulty": "0/8"},
        {"problem": "Keep me", "answer": "1", "difficulty": "1/8"},
        {"problem": " keep   ME ", "answer": "2", "difficulty": "6/8"},
        {"problem": "Held out", "answer": "3", "difficulty": "4/8"},
        {"problem": "Upper bound", "answer": "4", "difficulty": "6/8"},
        {"problem": "above range", "answer": "5", "difficulty": "7/8"},
        {
            "problem": "Prove that x + y = y + x.",
            "answer": "x+y=y+x",
            "difficulty": "3/8",
        },
        {"problem": "bad label", "answer": "6", "difficulty": "unknown"},
        {"problem": "empty answer", "answer": "", "difficulty": "3/8"},
    ]

    formatted = ENV.build_polaris_rows(rows, held_out_problems=[" held OUT "])

    assert formatted == [
        {
            "question": "Keep me",
            "answer": "1",
            "example_id": "polaris-00001",
            "info": {
                "source": "POLARIS-Project/Polaris-Dataset-53K",
                "difficulty": "1/8",
            },
        },
        {
            "question": "Upper bound",
            "answer": "4",
            "example_id": "polaris-00004",
            "info": {
                "source": "POLARIS-Project/Polaris-Dataset-53K",
                "difficulty": "6/8",
            },
        },
    ]


def test_build_polaris_rows_rejects_invalid_difficulty_bounds() -> None:
    with pytest.raises(ValueError, match="difficulty bounds"):
        ENV.build_polaris_rows([], min_difficulty=7, max_difficulty=2)


def test_build_polaris_rows_can_explicitly_include_proof_requests() -> None:
    rows = [
        {
            "problem": "Show that x + 0 = x.",
            "answer": "x+0=x",
            "difficulty": "3/8",
        }
    ]

    assert ENV.build_polaris_rows(rows) == []
    assert len(ENV.build_polaris_rows(rows, exclude_proof_problems=False)) == 1


def test_format_aime_rows_uses_integer_answers_without_solutions() -> None:
    formatted = ENV.format_aime_rows(
        [{"problem": "A problem", "answer": 7, "solution": "hidden"}]
    )

    assert formatted == [
        {
            "question": "A problem",
            "answer": "7",
            "example_id": "aime2024-00",
            "info": {"source": "HuggingFaceH4/aime_2024"},
        }
    ]


def test_resilient_math_rubric_replaces_executor_after_hard_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vf = pytest.importorskip("verifiers")

    async def slow_correct_answer(*args: object, **kwargs: object) -> float:
        del args, kwargs
        await asyncio.sleep(0.02)
        return 0.0

    monkeypatch.setattr(vf.MathRubric, "correct_answer", slow_correct_answer)
    rubric = ENV.build_resilient_math_rubric(
        vf,
        parser=vf.Parser(extract_fn=ENV.extract_tagged_answer),
        max_workers=2,
        timeout_seconds=5.0,
    )
    rubric.HARD_TIMEOUT_SECONDS = 0.01
    original_executor = rubric.executor

    asyncio.run(rubric.correct_answer(None, [], answer="1"))

    assert rubric.executor is not original_executor
    asyncio.run(rubric.teardown())


def test_resilient_math_rubric_scores_the_strict_tagged_answer() -> None:
    vf = pytest.importorskip("verifiers")
    rubric = ENV.build_resilient_math_rubric(
        vf,
        parser=vf.Parser(extract_fn=ENV.extract_tagged_answer),
        max_workers=1,
        timeout_seconds=5.0,
    )

    reward = asyncio.run(
        rubric.correct_answer(
            rubric.parser,
            [{"role": "assistant", "content": "<think>6 * 7</think><answer>42</answer>"}],
            answer="42",
        )
    )

    assert reward == 1.0
    asyncio.run(rubric.teardown())

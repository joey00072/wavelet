from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_environment_module() -> ModuleType:
    path = Path("environments/math500_tagged/math500_tagged.py")
    spec = importlib.util.spec_from_file_location("math500_tagged", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENV = _load_environment_module()


def test_extract_tagged_answer_requires_exact_complete_structure() -> None:
    assert (
        ENV.extract_tagged_answer(
            "<think>Work through it.</think><answer>\\frac{1}{2}</answer>"
        )
        == "\\frac{1}{2}"
    )
    assert ENV.extract_tagged_answer("<answer>5</answer>") == ""
    assert ENV.extract_tagged_answer("<think>x</think><answer>5</answer>tail") == ""
    assert (
        ENV.extract_tagged_answer(
            "<think>x</think><answer>5</answer><answer>6</answer>"
        )
        == ""
    )


def test_format_example_omits_reference_solution() -> None:
    formatted = ENV.format_example(
        {
            "problem": "What is 2+3?",
            "solution": "The hidden worked solution.",
            "answer": "5",
            "subject": "Algebra",
            "level": 1,
            "unique_id": "test/algebra/example.json",
        }
    )

    assert formatted == {
        "question": "What is 2+3?",
        "answer": "5",
        "example_id": "test/algebra/example.json",
        "info": {"subject": "Algebra", "level": 1},
    }

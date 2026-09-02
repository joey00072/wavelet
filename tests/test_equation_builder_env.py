from __future__ import annotations

import importlib.util
import sys
from itertools import permutations, product
from pathlib import Path
from types import ModuleType

import pytest


def _load_environment_module() -> ModuleType:
    path = Path("environments/equation_builder/equation_builder.py")
    spec = importlib.util.spec_from_file_location("equation_builder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENV = _load_environment_module()


def _load_audit_module() -> ModuleType:
    path = Path("examples/equation_builder/audit_rollouts.py")
    spec = importlib.util.spec_from_file_location("equation_builder_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _has_add_sub_solution(numbers: list[int], target: int) -> bool:
    for ordered_numbers in permutations(numbers):
        for operators in product(("+", "-"), repeat=len(numbers) - 1):
            result = ordered_numbers[0]
            for operator, number in zip(operators, ordered_numbers[1:], strict=True):
                result = result + number if operator == "+" else result - number
            if result == target:
                return True
    return False


def test_accepts_correct_equation_with_reordered_numbers() -> None:
    result = ENV.check_equation(
        "13+31-18-21=5",
        numbers=[13, 31, 18, 21],
        target=5,
    )

    assert result.valid is True
    assert result.reason == "correct"


def test_accepts_parentheses_with_only_addition_and_subtraction() -> None:
    result = ENV.check_equation(
        "31-(21-(13-18))=5",
        numbers=[13, 31, 18, 21],
        target=5,
    )

    assert result.valid is True


@pytest.mark.parametrize(
    ("equation", "numbers", "target"),
    [
        ("12+34-21=25", [12, 34, 21], 25),
        ("10+20+30-15-25=20", [10, 20, 30, 15, 25], 20),
    ],
)
def test_accepts_three_or_five_unique_numbers(
    equation: str,
    numbers: list[int],
    target: int,
) -> None:
    assert ENV.check_equation(
        equation,
        numbers=numbers,
        target=target,
    ).valid


def test_rejects_reused_or_missing_number() -> None:
    result = ENV.check_equation(
        "13+31-18-18=8",
        numbers=[13, 31, 18, 21],
        target=8,
    )

    assert result.valid is False
    assert "every given number exactly once" in result.reason


def test_rejects_non_unique_given_numbers() -> None:
    result = ENV.check_equation(
        "13+31-18-13=13",
        numbers=[13, 31, 18, 13],
        target=13,
    )

    assert result.valid is False
    assert "must be unique" in result.reason


def test_rejects_disallowed_operator() -> None:
    result = ENV.check_equation(
        "13*31-18-21=364",
        numbers=[13, 31, 18, 21],
        target=364,
    )

    assert result.valid is False
    assert "only integer literals" in result.reason


def test_rejects_expression_that_does_not_equal_target() -> None:
    result = ENV.check_equation(
        "13+31+18-21=5",
        numbers=[13, 31, 18, 21],
        target=5,
    )

    assert result.valid is False
    assert "does not equal the target" in result.reason


def test_rejects_wrong_right_hand_target() -> None:
    result = ENV.check_equation(
        "13+31-18-21=6",
        numbers=[13, 31, 18, 21],
        target=5,
    )

    assert result.valid is False
    assert "right-hand side" in result.reason


def test_default_generation_uses_four_numbers() -> None:
    example = ENV.build_examples(num_examples=1, seed=7)[0]

    assert len(example["info"]["numbers"]) == 4


@pytest.mark.parametrize("num_numbers", [3, 4, 5])
def test_generated_examples_are_deterministic_unique_and_valid(
    num_numbers: int,
) -> None:
    first = ENV.build_examples(num_examples=20, seed=7, num_numbers=num_numbers)
    second = ENV.build_examples(num_examples=20, seed=7, num_numbers=num_numbers)

    assert first == second
    assert len(first) == 20
    for example in first:
        numbers = example["info"]["numbers"]
        target = example["info"]["target"]
        assert len(numbers) == num_numbers
        assert len(set(numbers)) == num_numbers
        assert all(10 <= number <= 99 for number in numbers)
        assert 0 <= target <= 99
        assert "answer" not in example
        assert _has_add_sub_solution(numbers, target)


def test_generation_rejects_invalid_or_unreachable_target_range() -> None:
    with pytest.raises(ValueError, match="num_numbers"):
        ENV.build_examples(num_examples=1, seed=7, num_numbers=2)

    with pytest.raises(ValueError, match="num_numbers"):
        ENV.build_examples(num_examples=1, seed=7, num_numbers=6)

    with pytest.raises(ValueError, match="target_min"):
        ENV.build_examples(num_examples=1, seed=7, target_min=2, target_max=1)

    with pytest.raises(ValueError, match="Could not generate"):
        ENV.build_examples(
            num_examples=1,
            seed=7,
            target_min=10_000,
            target_max=10_001,
        )


def _saved_rollout(*, equation: str, reward: float) -> dict[str, object]:
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "The 4 unique two-digit numbers are 13, 31, 18, 21. "
                    "Using each number exactly once and only the + and - operators, "
                    "construct an equation whose result is 5. You may reorder the "
                    "numbers and use parentheses."
                ),
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": f"<think>check</think><answer>{equation}</answer>",
            }
        ],
        "reward": reward,
    }


def test_rollout_audit_accepts_independently_valid_reward() -> None:
    result = AUDIT.audit_row(_saved_rollout(equation="13+31-18-21=5", reward=1.0))

    assert result["independently_valid"] is True
    assert result["reward_hacking_candidate"] is False


def test_rollout_audit_flags_rewarded_invalid_equation() -> None:
    result = AUDIT.audit_row(_saved_rollout(equation="13+31+18+21=83", reward=1.0))

    assert result["independently_valid"] is False
    assert result["reward_hacking_candidate"] is True

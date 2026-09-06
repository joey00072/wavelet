from __future__ import annotations

import ast
import random
import re
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import verifiers as vf


SYSTEM_PROMPT = """You solve arithmetic equation-building puzzles.

Use each given number exactly once. You may reorder the numbers and use
parentheses, but the only arithmetic operators allowed are + and -.

Reason inside <think>...</think>. Put only the final equation inside
<answer>...</answer>, including the target after an equals sign.
"""

_INTEGER_PATTERN = re.compile(r"[+-]?\d+")
_TAGGED_RESPONSE = re.compile(
    r"\A\s*<think>\s*.+?\s*</think>\s*"
    r"<answer>\s*(?P<answer>.+?)\s*</answer>\s*\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class EquationCheck:
    valid: bool
    reason: str


class _InvalidExpression(ValueError):
    pass


def extract_tagged_equation(text: str) -> str:
    """Extract an equation only from one complete think/answer response."""
    if any(
        text.count(tag) != 1 for tag in ("<think>", "</think>", "<answer>", "</answer>")
    ):
        return ""
    match = _TAGGED_RESPONSE.fullmatch(text)
    return "" if match is None else match.group("answer").strip()


def _evaluate_expression(node: ast.AST) -> tuple[int, list[int], int]:
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body)

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value, [node.value], 0

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left_value, left_numbers, left_operations = _evaluate_expression(node.left)
        right_value, right_numbers, right_operations = _evaluate_expression(node.right)
        if isinstance(node.op, ast.Add):
            value = left_value + right_value
        else:
            value = left_value - right_value
        return (
            value,
            left_numbers + right_numbers,
            left_operations + right_operations + 1,
        )

    raise _InvalidExpression("only integer literals, +, -, and parentheses are allowed")


def check_equation(
    equation: str,
    *,
    numbers: list[int] | tuple[int, ...],
    target: int,
) -> EquationCheck:
    """Validate and safely evaluate a submitted equation."""
    expected_numbers = list(numbers)
    if not 3 <= len(expected_numbers) <= 5:
        return EquationCheck(False, "the puzzle must contain 3, 4, or 5 numbers")
    if len(set(expected_numbers)) != len(expected_numbers):
        return EquationCheck(False, "all given numbers must be unique")
    if not all(
        isinstance(number, int) and not isinstance(number, bool) and 10 <= number <= 99
        for number in expected_numbers
    ):
        return EquationCheck(False, "all given numbers must be two-digit integers")

    if equation.count("=") != 1:
        return EquationCheck(False, "the answer must contain exactly one equals sign")

    expression_text, submitted_target_text = (
        part.strip() for part in equation.split("=", maxsplit=1)
    )
    if not expression_text or not _INTEGER_PATTERN.fullmatch(submitted_target_text):
        return EquationCheck(False, "the equation or integer target is missing")

    submitted_target = int(submitted_target_text)
    if submitted_target != target:
        return EquationCheck(False, "the right-hand side does not match the target")

    try:
        expression = ast.parse(expression_text, mode="eval")
        value, used_numbers, operation_count = _evaluate_expression(expression)
    except (SyntaxError, _InvalidExpression):
        return EquationCheck(
            False,
            "only integer literals, +, -, and parentheses are allowed",
        )

    if Counter(used_numbers) != Counter(expected_numbers):
        return EquationCheck(
            False, "the equation must use every given number exactly once"
        )

    if operation_count != len(expected_numbers) - 1:
        return EquationCheck(False, "the equation must connect all given numbers")

    if value != target:
        return EquationCheck(
            False, "the left-hand expression does not equal the target"
        )

    return EquationCheck(True, "correct")


def _operator_patterns(num_numbers: int) -> list[tuple[str, ...]]:
    patterns = list(product(("+", "-"), repeat=num_numbers - 1))
    return [pattern for pattern in patterns if len(set(pattern)) == 2]


def _apply_operations(numbers: list[int], operators: tuple[str, ...]) -> int:
    result = numbers[0]
    for operator, number in zip(operators, numbers[1:], strict=True):
        result = result + number if operator == "+" else result - number
    return result


def build_examples(
    *,
    num_examples: int,
    seed: int,
    num_numbers: int = 4,
    target_min: int = 0,
    target_max: int = 99,
) -> list[dict[str, Any]]:
    """Build deterministic, guaranteed-solvable equation tasks."""
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")
    if not 3 <= num_numbers <= 5:
        raise ValueError("num_numbers must be 3, 4, or 5")
    if target_min > target_max:
        raise ValueError("target_min must be less than or equal to target_max")

    rng = random.Random(seed)
    operator_patterns = _operator_patterns(num_numbers)
    examples: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(1000, num_examples * 1000)
    while len(examples) < num_examples:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                "Could not generate enough equations in the requested target range"
            )
        given_numbers = rng.sample(range(10, 100), k=num_numbers)
        solution_order = rng.sample(given_numbers, k=num_numbers)
        operators = rng.choice(operator_patterns)
        target = _apply_operations(solution_order, operators)
        if not target_min <= target <= target_max:
            continue
        rendered_numbers = ", ".join(str(number) for number in given_numbers)
        question = (
            f"The {num_numbers} unique two-digit numbers are {rendered_numbers}. "
            f"Using each number exactly once and only the + and - operators, "
            f"construct an equation whose result is {target}. You may reorder "
            f"the numbers and use parentheses."
        )
        examples.append(
            {
                "question": question,
                "info": {
                    "numbers": given_numbers,
                    "target": target,
                },
                "task": "equation-builder",
            }
        )
    return examples


def load_environment(
    num_examples: int = 4096,
    eval_examples: int = 256,
    seed: int = 42,
    num_numbers: int = 4,
    target_min: int = 0,
    target_max: int = 99,
) -> vf.Environment:
    """Load the equation-building verifier environment."""
    import verifiers as vf
    from datasets import Dataset

    def build_dataset(count: int, dataset_seed: int) -> Dataset:
        return Dataset.from_list(
            build_examples(
                num_examples=count,
                seed=dataset_seed,
                num_numbers=num_numbers,
                target_min=target_min,
                target_max=target_max,
            )
        )

    parser = vf.Parser(extract_fn=extract_tagged_equation)

    def equation_reward(
        completion: Any,
        info: dict[str, Any],
        parser: Any,
        **_: Any,
    ) -> float:
        parsed_answer = parser.parse_answer(completion)
        if not isinstance(parsed_answer, str):
            return 0.0
        result = check_equation(
            parsed_answer,
            numbers=info["numbers"],
            target=info["target"],
        )
        return float(result.valid)

    rubric = vf.Rubric(
        funcs=[equation_reward],
        weights=[1.0],
        parser=parser,
    )
    return vf.SingleTurnEnv(
        dataset=lambda: build_dataset(num_examples, seed),
        eval_dataset=lambda: build_dataset(eval_examples, seed + 1),
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )

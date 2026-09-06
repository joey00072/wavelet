from __future__ import annotations

import pytest

from wavelet.configs.rl_config import RLRewardConfig
from wavelet.data.rl import RLExample
from wavelet.orchestrator.reward import RLRewardScorer


def _record(response: str, expected: str = "439") -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": "problem"}],
        completion=[{"role": "assistant", "content": response}],
        target_completion=[{"role": "assistant", "content": expected}],
        advantage=None,
        reward=None,
    )


def test_math_reward_keeps_wrong_formatted_answer_near_zero() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(_record("<end_working_out><SOLUTION>3062</SOLUTION>"))

    assert reward == 0.05


def test_math_reward_gives_full_credit_for_exact_formatted_answer() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(_record("<end_working_out><SOLUTION>439</SOLUTION>"))

    assert reward == 1.0


def test_math_reward_can_extract_boxed_answer_without_tags() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(_record("Therefore the answer is \\boxed{439}."))

    assert reward == 0.9


def test_math_reward_gives_limited_partial_credit_for_close_number() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(_record("<end_working_out><SOLUTION>440</SOLUTION>"))

    assert reward == 0.39999999999999997


def test_math_reward_accepts_alternate_solution_tags() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(
        _record("<end_working_out><start_solution>439</start_solution>")
    )

    assert reward == 0.9


def test_math_reward_uses_math_verify_equivalence_when_available() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(
        _record(
            "<end_working_out><SOLUTION>\\frac{1}{2}</SOLUTION>",
            expected="0.5",
        )
    )

    assert reward == 1.0


@pytest.mark.parametrize(
    "response",
    [
        "The answer is 42",
        "So the answer is 42.",
        "The person has 42 apples",
        "Answer: 42",
        "answer = 42",
        "The final result is 42",
    ],
)
def test_math_reward_keyword_fallback_extracts_whole_number(response: str) -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(_record(response, expected="42"))

    assert reward == 0.9


def test_math_reward_keyword_fallback_does_not_match_inside_words() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    # "so" inside "person" must not trigger the keyword branch and the
    # trailing number must survive extraction.
    reward = scorer.score(_record("The person has 42 apples", expected="7"))

    assert reward == 0.0


def test_math_reward_terminal_format_bonus_requires_tags_at_end() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(
        _record("<end_working_out><SOLUTION>439</SOLUTION>\njunk\nmore")
    )

    assert reward < 1.0
    assert reward == pytest.approx(0.96)


def test_math_reward_strips_only_thousands_separators() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    grouped = scorer.score(
        _record("<end_working_out><SOLUTION>1,234</SOLUTION>", expected="1234")
    )
    not_grouped = scorer.score(
        _record("<end_working_out><SOLUTION>12</SOLUTION>", expected="1,2")
    )

    assert grouped == 1.0
    assert not_grouped == 0.05

from __future__ import annotations

from wavelet.configs.rl_config import RLRewardConfig
from wavelet.data.rl_dataset import RLExample
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

    reward = scorer.score(
        _record("<end_working_out><SOLUTION>3062</SOLUTION>")
    )

    assert reward == 0.05


def test_math_reward_gives_full_credit_for_exact_formatted_answer() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(
        _record("<end_working_out><SOLUTION>439</SOLUTION>")
    )

    assert reward == 1.0


def test_math_reward_can_extract_boxed_answer_without_tags() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(_record("Therefore the answer is \\boxed{439}."))

    assert reward == 0.9


def test_math_reward_gives_limited_partial_credit_for_close_number() -> None:
    scorer = RLRewardScorer(RLRewardConfig(mode="math_format"))

    reward = scorer.score(
        _record("<end_working_out><SOLUTION>440</SOLUTION>")
    )

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

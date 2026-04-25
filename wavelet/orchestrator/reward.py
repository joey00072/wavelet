from __future__ import annotations

import re

from wavelet.configs.rl_config import RLRewardConfig
from wavelet.data.rl_dataset import RLExample


def _assistant_text(messages: list[dict[str, str]] | None) -> str:
    if messages is None:
        return ""
    parts = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    ]
    return "\n".join(parts)


def _first_numeric_text(value: str) -> str | None:
    match = re.search(r"[-]?[\d.,]+", value)
    if match is None:
        return None
    return match.group(0).replace(",", "")


class RLRewardScorer:
    def __init__(self, config: RLRewardConfig) -> None:
        self.config = config

    def score(self, record: RLExample) -> float:
        if self.config.mode == "passthrough":
            if record.reward is None:
                raise ValueError(
                    "reward.mode='passthrough' requires each rollout row to provide a reward."
                )
            return self._postprocess(float(record.reward))

        if self.config.mode == "reference_match":
            expected = _assistant_text(record.target_completion)
            if not expected:
                raise ValueError(
                    "reward.mode='reference_match' requires a target assistant completion."
                )
            actual = _assistant_text(record.completion)
            score = 1.0 if self._normalize(actual) == self._normalize(expected) else 0.0
            return self._postprocess(score)

        if self.config.mode == "math_format":
            return self._postprocess(self._score_math_format(record))

        raise ValueError(f"Unsupported reward mode: {self.config.mode}")

    def _postprocess(self, score: float) -> float:
        if self.config.rescale_min is not None and self.config.rescale_max is not None:
            score = (score - self.config.rescale_min) / (
                self.config.rescale_max - self.config.rescale_min
            )
        if self.config.clamp_min is not None:
            score = max(self.config.clamp_min, score)
        if self.config.clamp_max is not None:
            score = min(self.config.clamp_max, score)
        return score

    def _normalize(self, value: str) -> str:
        text = value
        if self.config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text.strip())
        if not self.config.case_sensitive:
            text = text.casefold()
        return text

    def _score_math_format(self, record: RLExample) -> float:
        response = _assistant_text(record.completion)
        expected = self._normalize(_assistant_text(record.target_completion))
        if not expected:
            raise ValueError(
                "reward.mode='math_format' requires target_completion/completion "
                "to contain the expected answer."
            )
        extracted = self._extract_math_solution(response)

        score = self._score_math_exact_format(response)
        score += self._score_math_approx_format(response)
        score += self._score_math_answer(extracted, expected)
        score += self._score_math_number(extracted, expected)
        return score

    def _math_solution_regex(self) -> re.Pattern[str]:
        solution_end = (
            rf"{re.escape(self.config.solution_end)}[\s]{{0,}}(?:<\|endoftext\|>)?"
        )
        return re.compile(
            rf"{re.escape(self.config.reasoning_end)}.*?"
            rf"{re.escape(self.config.solution_start)}(.+?){solution_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )

    def _extract_math_solution(self, response: str) -> str | None:
        match = self._math_solution_regex().search(response)
        if match is None:
            return None
        return match.group(1)

    def _score_math_exact_format(self, response: str) -> float:
        return 3.0 if self._math_solution_regex().search(response) is not None else 0.0

    def _score_math_approx_format(self, response: str) -> float:
        score = 0.0
        score += 0.5 if response.count(self.config.reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(self.config.solution_start) == 1 else -1.0
        score += 0.5 if response.count(self.config.solution_end) == 1 else -1.0
        return score

    def _score_math_answer(self, extracted: str | None, expected: str) -> float:
        if extracted is None:
            return -2.0
        guess = self._normalize(extracted)
        if guess == expected:
            return 5.0
        if guess.strip() == expected.strip():
            return 3.5
        try:
            ratio = float(guess) / float(expected)
        except (TypeError, ValueError, ZeroDivisionError):
            return -4.5
        if 0.9 <= ratio <= 1.1:
            return 2.0
        if 0.8 <= ratio <= 1.2:
            return 1.5
        return -2.5

    def _score_math_number(self, extracted: str | None, expected: str) -> float:
        if extracted is None:
            return -2.5
        guess = _first_numeric_text(extracted)
        if guess is None:
            return -2.5
        try:
            expected_number = float(expected.strip())
            guess_number = float(guess.strip())
        except ValueError:
            return 0.0
        return 3.5 if guess_number == expected_number else -1.5

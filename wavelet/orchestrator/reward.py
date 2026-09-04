from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from typing import Any

from wavelet.configs.rl_config import RLRewardConfig
from wavelet.data.rl import RLExample

_NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?"
)


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
    for match in _NUMBER_RE.finditer(value):
        text = match.group(0).replace(",", "")
        if text and text not in {"+", "-", ".", "+.", "-."}:
            return text
    return None


def _compact_answer_text(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip(" .,:;")


def _parse_numeric_answer(value: str) -> Decimal | None:
    text = _compact_answer_text(value)
    text = text.strip("()[]{}")
    if not text:
        return None
    try:
        if "/" in text and re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", text):
            fraction = Fraction(text.replace("+", ""))
            return Decimal(fraction.numerator) / Decimal(fraction.denominator)
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            return Decimal(text)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return None


@lru_cache(maxsize=1)
def _math_verify_api() -> tuple[Any, Any] | None:
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    return parse, verify


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

        answer_score = self._score_math_answer(extracted, expected)
        format_score = self._score_math_format_bonus(response)
        if answer_score >= 1.0:
            return min(1.0, 0.9 + format_score)
        if answer_score > 0.0:
            return min(0.6, answer_score + min(format_score, 0.05))
        return min(0.05, format_score)

    def _math_solution_regex(self) -> re.Pattern[str]:
        solution_end = (
            rf"{re.escape(self.config.solution_end)}[\s]{{0,}}(?:<\|endoftext\|>)?"
        )
        return re.compile(
            rf"{re.escape(self.config.reasoning_end)}.*?"
            rf"{re.escape(self.config.solution_start)}(.+?){solution_end}"
            rf"[\s]{{0,}}\Z",
            flags=re.DOTALL,
        )

    def _extract_math_solution(self, response: str) -> str | None:
        match = self._math_solution_regex().search(response)
        if match is not None:
            return match.group(1)
        tagged_solution = self._fallback_tagged_solution(response)
        if tagged_solution is not None:
            return tagged_solution
        answer_match = re.search(
            r"\b(?:answer|solution|therefore|so)\b[^\n]*?(?:[:=]|\bis\b)\s*([^\n]+)$",
            response.strip(),
            flags=re.IGNORECASE,
        )
        if answer_match is not None:
            return answer_match.group(1)
        return _first_numeric_text(response.splitlines()[-1] if response else "")

    def _fallback_tagged_solution(self, response: str) -> str | None:
        tagged = re.findall(
            rf"{re.escape(self.config.solution_start)}(.+?){re.escape(self.config.solution_end)}",
            response,
            flags=re.MULTILINE | re.DOTALL,
        )
        if tagged:
            return tagged[-1]
        for start_tag, end_tag in (
            ("<start_solution>", "</start_solution>"),
            ("<answer>", "</answer>"),
            ("<final_answer>", "</final_answer>"),
        ):
            tagged = re.findall(
                rf"{re.escape(start_tag)}(.+?){re.escape(end_tag)}",
                response,
                flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )
            if tagged:
                return tagged[-1]
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", response)
        if boxed:
            return boxed[-1]
        return None

    def _score_math_format_bonus(self, response: str) -> float:
        if self._math_solution_regex().search(response) is not None:
            return 0.10
        if (
            response.count(self.config.solution_start) == 1
            and response.count(self.config.solution_end) == 1
        ):
            return 0.06
        if (
            self.config.solution_start in response
            or self.config.solution_end in response
        ):
            return 0.02
        return 0.0

    def _score_math_answer(self, extracted: str | None, expected: str) -> float:
        if extracted is None:
            return 0.0
        guess = self._normalize(extracted)
        if _compact_answer_text(guess) == _compact_answer_text(expected):
            return 1.0
        if self._math_verify_equivalent(guess, expected):
            return 1.0

        guess_number = _parse_numeric_answer(guess)
        expected_number = _parse_numeric_answer(expected)
        if guess_number is None:
            numeric_guess = _first_numeric_text(guess)
            guess_number = (
                _parse_numeric_answer(numeric_guess)
                if numeric_guess is not None
                else None
            )
        return self._score_numeric_answer(guess_number, expected_number)

    @staticmethod
    def _score_numeric_answer(
        guess: Decimal | None,
        expected: Decimal | None,
    ) -> float:
        if expected is None or guess is None:
            return 0.0
        if guess == expected:
            return 1.0
        if expected == 0:
            return 0.0
        relative_error = abs((guess - expected) / expected)
        if relative_error <= Decimal("0.01"):
            return 0.35
        if relative_error <= Decimal("0.05"):
            return 0.20
        return 0.0

    def _math_verify_equivalent(self, guess: str, expected: str) -> bool:
        api = _math_verify_api()
        if api is None:
            return False
        parse, verify = api
        try:
            parsed_expected = parse(expected)
            parsed_guess = parse(guess)
            return bool(
                parsed_expected
                and parsed_guess
                and verify(parsed_expected, parsed_guess)
            )
        except Exception:  # noqa: BLE001 - optional parser failures mean no match
            return False

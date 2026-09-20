from difflib import SequenceMatcher

import pytest

from examples.reverse_text.reverse_text_diagnostic import lcs


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<reversed_text>cba</reversed_text>", 1.0),
        (
            "<reversed_text> cb </reversed_text>",
            SequenceMatcher(None, "cb", "cba").ratio(),
        ),
        ("cba", 0.0),
        ("<reversed_text>cba", 0.0),
        ("", 0.0),
        ([], 0.0),
        ([{"role": "assistant", "content": "<reversed_text>cba</reversed_text>"}], 1.0),
    ],
)
def test_reverse_text_reward(completion, expected: float) -> None:
    assert lcs(completion, "cba") == expected

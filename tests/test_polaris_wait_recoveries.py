from __future__ import annotations

from examples.qwen2_5_7b_polaris.generate_wait_recoveries import (
    RECOVERY_PHRASES,
    build_continuation_prefixes,
    build_midpoint_prefixes,
    truncate_think_at_midpoint,
    truncate_think_at_newline,
)


def test_truncate_think_replaces_suffix_at_requested_newline() -> None:
    completion = """<think>
line one
line two
line three
line four
</think><answer>wrong</answer>"""

    second_last = truncate_think_at_newline(
        completion,
        newline_from_end=2,
        recovery_phrase="Wait,",
    )
    fourth_last = truncate_think_at_newline(
        completion,
        newline_from_end=4,
        recovery_phrase="Alternatively,",
    )

    assert second_last == "<think>\nline one\nline two\nline three\nWait, "
    assert fourth_last == "<think>\nline one\nAlternatively, "


def test_build_prefixes_uses_two_cuts_and_rotates_phrases() -> None:
    completion = "<think>\na\nb\nc\nd\ne\n</think><answer>0</answer>"
    rows = [
        {
            "id": f"row-{index}",
            "question": f"question {index}",
            "reference_answer": "1",
            "completion": completion,
        }
        for index in range(4)
    ]

    prefixes, counts = build_continuation_prefixes(rows, [2, 4])

    assert len(prefixes) == 8
    assert [prefix.recovery_phrase for prefix in prefixes[:6]] == list(RECOVERY_PHRASES)
    assert sum(counts[f"phrase/{phrase}"] for phrase in RECOVERY_PHRASES) == 8


def test_midpoint_prefers_nearest_paragraph_or_line_boundary() -> None:
    completion = "<think>aaaa\nbbbb\n\ncccc\ndddd</think><answer>0</answer>"

    truncated = truncate_think_at_midpoint(
        completion,
        recovery_phrase="Wait,",
    )

    assert truncated == ("<think>aaaa\nbbbb\n\nWait, ", 11, "\\n\\n")


def test_midpoint_builds_each_requested_phrase_for_every_trace() -> None:
    completion = "<think>before\nafter</think><answer>0</answer>"
    rows = [
        {
            "id": "row-1",
            "question": "question",
            "reference_answer": "1",
            "completion": completion,
        }
    ]

    prefixes, counts = build_midpoint_prefixes(rows, ("Alternatively,", "Wait,"))

    assert len(prefixes) == 2
    assert [prefix.recovery_phrase for prefix in prefixes] == [
        "Alternatively,",
        "Wait,",
    ]
    assert counts["separator/\\n"] == 2

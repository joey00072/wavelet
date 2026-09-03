from __future__ import annotations

from examples.qwen2_5_7b_polaris.generate_wait_recoveries import (
    RECOVERY_PHRASES,
    build_midpoint_prefixes,
    truncate_think_at_midpoint,
)


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

    prefixes, counts = build_midpoint_prefixes(rows, RECOVERY_PHRASES)

    assert len(prefixes) == 2
    assert [prefix.recovery_phrase for prefix in prefixes] == [
        "Alternatively,",
        "Wait,",
    ]
    assert counts["separator/\\n"] == 2

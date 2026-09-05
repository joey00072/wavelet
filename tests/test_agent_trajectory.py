from __future__ import annotations

import pytest

from wavelet.orchestrator.agent_trajectory import (
    TokenSegment,
    merge_token_segments,
)


def test_merge_token_segments_preserves_exact_prefix_turn_boundary() -> None:
    samples = merge_token_segments(
        [
            TokenSegment(
                prompt_ids=[1, 2],
                output_ids=[3, 4],
                output_logprobs=[-0.3, -0.4],
                turn_id="turn-0",
            ),
            TokenSegment(
                prompt_ids=[1, 2, 3, 4, 5],
                output_ids=[6],
                output_logprobs=[-0.6],
                turn_id="turn-1",
            ),
        ],
        temperature=0.7,
    )

    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 3, 4, 5]
    assert samples[0].target_ids == [2, 3, 4, 5, 6]
    assert samples[0].loss_mask == [False, True, True, False, True]
    assert samples[0].inference_logprobs == [0.0, -0.3, -0.4, 0.0, -0.6]
    assert samples[0].temperatures == [0.7] * 5
    assert samples[0].turn_ids == ["turn-0", "turn-0", "turn-0", "turn-1", "turn-1"]


def test_merge_token_segments_trains_first_output_after_exact_prefix() -> None:
    samples = merge_token_segments(
        [
            TokenSegment([1, 2], [3, 4], [-0.3, -0.4]),
            TokenSegment([1, 2, 3, 4], [5, 6], [-0.5, -0.6]),
        ],
    )

    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 3, 4, 5]
    assert samples[0].target_ids == [2, 3, 4, 5, 6]
    assert samples[0].loss_mask == [False, True, True, True, True]
    assert samples[0].inference_logprobs == [0.0, -0.3, -0.4, -0.5, -0.6]


def test_merge_token_segments_fans_out_when_prompt_does_not_extend_prefix() -> None:
    samples = merge_token_segments(
        [
            TokenSegment([1, 2], [3], [-0.3], turn_id="a"),
            TokenSegment([1, 9], [10], [-0.1], turn_id="b"),
        ],
    )

    assert len(samples) == 2
    assert samples[0].target_ids == [2, 3]
    assert samples[1].target_ids == [9, 10]


def test_merge_token_segments_rebases_rerendered_qwen3_history() -> None:
    samples = merge_token_segments(
        [
            TokenSegment(
                # Disabled-thinking generation framing is present here.
                prompt_ids=[1, 10, 11, 20, 21, 22],
                output_ids=[30, 31],
                output_logprobs=[-0.3, -0.31],
                output_sampling_mask=[[30, 130], [31, 131]],
                turn_id="turn-0",
            ),
            TokenSegment(
                # Qwen3 rerenders the assistant history without that framing.
                prompt_ids=[1, 10, 11, 20, 30, 31, 40, 20, 21, 22],
                output_ids=[50, 51],
                output_logprobs=[-0.5, -0.51],
                output_sampling_mask=[[50, 150], [51, 151]],
                turn_id="turn-1",
            ),
            TokenSegment(
                prompt_ids=[
                    1,
                    10,
                    11,
                    20,
                    30,
                    31,
                    40,
                    20,
                    50,
                    51,
                    60,
                    20,
                    21,
                    22,
                ],
                output_ids=[70],
                output_logprobs=[-0.7],
                output_sampling_mask=[[70, 170]],
                turn_id="turn-2",
            ),
        ],
        temperature=0.8,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert [
        token_id
        for token_id, trainable in zip(
            sample.target_ids,
            sample.loss_mask,
            strict=True,
        )
        if trainable
    ] == [30, 31, 50, 51, 70]
    assert [
        logprob
        for logprob, trainable in zip(
            sample.inference_logprobs,
            sample.loss_mask,
            strict=True,
        )
        if trainable
    ] == [-0.3, -0.31, -0.5, -0.51, -0.7]
    assert [
        mask
        for mask, trainable in zip(
            sample.sampling_masks,
            sample.loss_mask,
            strict=True,
        )
        if trainable
    ] == [[30, 130], [31, 131], [50, 150], [51, 151], [70, 170]]
    assert [sample.input_ids[0], *sample.target_ids] == [
        1,
        10,
        11,
        20,
        30,
        31,
        40,
        20,
        50,
        51,
        60,
        20,
        21,
        22,
        70,
    ]


def test_merge_token_segments_rejects_ambiguous_rerendered_history() -> None:
    samples = merge_token_segments(
        [
            TokenSegment([1, 2], [3, 4], [-0.3, -0.4]),
            TokenSegment([1, 3, 4, 5, 3, 4], [6], [-0.6]),
        ]
    )

    assert len(samples) == 2
    assert samples[0].target_ids == [2, 3, 4]
    assert samples[1].target_ids == [3, 4, 5, 3, 4, 6]


def test_merge_token_segments_can_mask_failed_outputs() -> None:
    samples = merge_token_segments(
        [TokenSegment([1], [2, 3], [-0.2, -0.3])],
        mask_outputs=True,
    )

    assert samples[0].loss_mask == [False, False]
    assert samples[0].inference_logprobs == [-0.2, -0.3]


def test_token_segment_rejects_misaligned_logprobs() -> None:
    with pytest.raises(ValueError, match="output_logprobs"):
        TokenSegment([1], [2, 3], [-0.2])


def test_merge_token_segments_preserves_sampling_masks_at_turn_boundaries() -> None:
    samples = merge_token_segments(
        [
            TokenSegment(
                [1, 2],
                [3, 4],
                [-0.3, -0.4],
                output_sampling_mask=[[3, 8], [4, 9]],
            ),
            TokenSegment(
                [1, 2, 3, 4],
                [5],
                [-0.5],
                output_sampling_mask=[[5, 7]],
            ),
        ]
    )

    assert samples[0].sampling_masks == [
        None,
        [3, 8],
        [4, 9],
        [5, 7],
    ]

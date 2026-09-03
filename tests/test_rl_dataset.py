from __future__ import annotations

import pytest

from wavelet.configs.rl_config import RLDataConfig
from wavelet.data.rl_dataset import PackedRLDataset, RLExample, prepare_rl_sample


def _record(index: int, *, length: int = 6) -> RLExample:
    return RLExample(
        prompt=[],
        completion=[],
        advantage=float(index),
        reward=float(index),
        input_ids=list(range(index * 100, index * 100 + length)),
        target_ids=list(range(index * 100 + 1, index * 100 + length + 1)),
        loss_mask=[True] * length,
        inference_logprobs=[-1.0] * length,
        temperatures=[1.0] * length,
    )


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_packed_rl_dataset_gives_each_rank_same_micro_batch_count(rank: int) -> None:
    dataset = PackedRLDataset(
        records=[_record(index) for index in range(5)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=RLDataConfig(pack_sequences=True, seq_len=8),
        data_rank=rank,
        data_world_size=4,
    )

    bins = dataset._bins_for_epoch(0)  # noqa: SLF001

    assert len(bins) == 2
    assert dataset.micro_batch_count() == 2


def test_packed_rl_dataset_pads_incomplete_distributed_tail_with_zero_loss() -> None:
    dataset = PackedRLDataset(
        records=[_record(index) for index in range(5)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=RLDataConfig(pack_sequences=True, seq_len=8),
        data_rank=3,
        data_world_size=4,
    )

    bins = dataset._bins_for_epoch(0)  # noqa: SLF001

    assert len(bins) == 2
    assert sum(bins[-1]["loss_mask"]) == 0
    assert bins[-1]["advantages"] == []
    assert bins[-1]["sample_count"] == 0
    assert dataset.loss_scale_for_next_local_batch(2) == pytest.approx(6.0)
    assert dataset.loss_scale_for_next_local_batch(
        2,
        normalization="sequence",
    ) == pytest.approx(1.0)


def test_packed_rl_dataset_counts_real_local_samples() -> None:
    dataset = PackedRLDataset(
        records=[_record(index) for index in range(5)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=RLDataConfig(pack_sequences=True, seq_len=8),
        data_rank=3,
        data_world_size=4,
    )

    assert dataset.local_real_sample_count() == 1


def test_packed_rl_dataset_allows_dummy_only_distributed_ranks() -> None:
    dataset = PackedRLDataset(
        records=[_record(0)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=RLDataConfig(pack_sequences=True, seq_len=8),
        data_rank=3,
        data_world_size=4,
    )

    bins = dataset._bins_for_epoch(0)  # noqa: SLF001

    assert len(bins) == 1
    assert sum(bins[0]["loss_mask"]) == 0
    assert bins[0]["advantages"] == []
    assert bins[0]["sample_count"] == 0
    assert dataset.local_real_sample_count() == 0
    assert dataset.loss_scale_for_next_local_batch(1) == pytest.approx(0.0)


def test_pretokenized_dummy_rollout_keeps_zero_loss_sample() -> None:
    record = _record(0)
    record.loss_mask = [False] * len(record.loss_mask)
    record.advantage = 0.0
    record.reward = None
    record.inference_logprobs = []
    record.temperatures = []
    record.metadata = {"_wavelet_dummy_rollout": True}

    sample = prepare_rl_sample(
        record,
        tokenizer=None,  # type: ignore[arg-type]
        data_config=RLDataConfig(seq_len=8),
        seq_len=8,
    )

    assert sample is not None
    assert sum(sample["loss_mask"]) == 0
    assert sample["advantages"] == []
    assert sample["temperatures"] == []
    assert sample["sample_count"] == 0


def test_pretokenized_filtered_rollout_keeps_metric_sample() -> None:
    record = _record(0)
    record.loss_mask = [False] * len(record.loss_mask)
    record.advantage = 0.0
    record.inference_logprobs = []
    record.temperatures = []
    record.metadata = {"_wavelet_filtered_rollout": True}

    sample = prepare_rl_sample(
        record,
        tokenizer=None,  # type: ignore[arg-type]
        data_config=RLDataConfig(seq_len=8),
        seq_len=8,
    )

    assert sample is not None
    assert sum(sample["loss_mask"]) == 0
    assert sample["advantages"] == []
    assert sample["temperatures"] == []
    assert sample["reward"] == 0.0
    assert sample.get("sample_count", 1) == 1


def test_pretokenized_rollout_count_metadata_sets_sample_count() -> None:
    record = _record(0)
    record.metadata = {"_wavelet_rollout_count": 0}

    sample = prepare_rl_sample(
        record,
        tokenizer=None,  # type: ignore[arg-type]
        data_config=RLDataConfig(seq_len=8),
        seq_len=8,
    )

    assert sample is not None
    assert sample["sample_count"] == 0


def test_pretokenized_source_lengths_are_checked_before_truncation() -> None:
    record = _record(0, length=10)
    record.target_ids.append(11)

    with pytest.raises(ValueError, match="mismatched source"):
        prepare_rl_sample(
            record,
            tokenizer=None,  # type: ignore[arg-type]
            data_config=RLDataConfig(seq_len=8),
            seq_len=8,
        )


def test_pretokenized_logprobs_must_cover_every_source_trainable_token() -> None:
    record = _record(0, length=10)
    record.inference_logprobs = [-1.0] * 8

    with pytest.raises(ValueError, match="inference_logprobs.*8 != 10"):
        prepare_rl_sample(
            record,
            tokenizer=None,  # type: ignore[arg-type]
            data_config=RLDataConfig(seq_len=8),
            seq_len=8,
        )


def test_pretokenized_tail_truncation_keeps_aligned_logprob_prefix() -> None:
    record = _record(0, length=10)

    sample = prepare_rl_sample(
        record,
        tokenizer=None,  # type: ignore[arg-type]
        data_config=RLDataConfig(seq_len=8),
        seq_len=8,
    )

    assert sample is not None
    assert sample["input_ids"] == list(range(8))
    assert sample["inference_logprobs"] == [-1.0] * 8
    assert len(sample["advantages"]) == 8


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("advantage", float("nan"), "advantage.*finite"),
        ("reward", float("inf"), "reward.*finite"),
        ("inference_logprobs", [float("-inf")] * 6, "inference_logprobs.*finite"),
        ("teacher_logprobs", [float("nan")] * 6, "teacher_logprobs.*finite"),
        ("temperatures", [0.0] * 6, "temperatures.*positive"),
        ("temperatures", [-1.0] * 6, "temperatures.*positive"),
    ],
)
def test_rl_samples_reject_invalid_numeric_streams(
    field_name: str,
    value,
    message: str,
) -> None:
    record = _record(0)
    setattr(record, field_name, value)

    with pytest.raises(ValueError, match=message):
        prepare_rl_sample(
            record,
            tokenizer=None,  # type: ignore[arg-type]
            data_config=RLDataConfig(seq_len=8),
            seq_len=8,
        )

from __future__ import annotations

import pytest

from wavelet.configs.rl_config import RLDataConfig
from wavelet.data.rl import (
    PackedRLDataset,
    RLExample,
    pack_samples,
    prepare_rl_sample,
)


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
    assert dataset.loss_scale_for_next_local_batch(1) == pytest.approx(1.0)


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


def test_opd_sample_does_not_require_scalar_advantage() -> None:
    record = _record(0)
    record.advantage = None
    record.reward = None
    record.ref_logprobs = [-0.5] * len(record.loss_mask)
    record.rl_weights = 0.0
    record.ref_kl_weights = 1.0

    sample = prepare_rl_sample(
        record,
        tokenizer=None,  # type: ignore[arg-type]
        data_config=RLDataConfig(seq_len=8),
        seq_len=8,
    )

    assert sample is not None
    assert sample["advantages"] == [0.0] * len(record.loss_mask)
    assert sample["rl_weights"] == [0.0] * len(record.loss_mask)
    assert sample["ref_kl_weights"] == [1.0] * len(record.loss_mask)


def test_ref_kl_weights_require_reference_logprobs() -> None:
    record = _record(0)
    record.advantage = None
    record.reward = None
    record.rl_weights = 0.0
    record.ref_kl_weights = 1.0

    with pytest.raises(ValueError, match="require ref_logprobs"):
        prepare_rl_sample(
            record,
            tokenizer=None,  # type: ignore[arg-type]
            data_config=RLDataConfig(seq_len=8),
            seq_len=8,
        )


def test_packing_preserves_mixed_grpo_and_opd_components() -> None:
    grpo = prepare_rl_sample(
        _record(0, length=2),
        tokenizer=None,  # type: ignore[arg-type]
        data_config=RLDataConfig(seq_len=8),
        seq_len=8,
    )
    opd_record = _record(1, length=2)
    opd_record.advantage = None
    opd_record.reward = None
    opd_record.ref_logprobs = [-0.5, -0.5]
    opd_record.rl_weights = 0.0
    opd_record.ref_kl_weights = 1.0
    opd = prepare_rl_sample(
        opd_record,
        tokenizer=None,  # type: ignore[arg-type]
        data_config=RLDataConfig(seq_len=8),
        seq_len=8,
    )

    assert grpo is not None
    assert opd is not None
    packed = pack_samples([grpo, opd], seq_len=8, pad_to_multiple_of=1)

    assert len(packed) == 1
    assert sorted(packed[0]["rl_weights"]) == [0.0, 0.0, 1.0, 1.0]
    assert sorted(packed[0]["ref_kl_weights"]) == [0.0, 0.0, 1.0, 1.0]
    assert sorted(packed[0]["ref_logprobs"]) == [-0.5, -0.5, 0.0, 0.0]

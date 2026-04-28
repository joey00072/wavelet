from __future__ import annotations

import pytest

from wavelet.configs.rl_config import RLDataConfig
from wavelet.data.rl_dataset import PackedRLDataset, RLExample


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
    assert dataset.loss_scale_for_next_local_batch(2) == pytest.approx(7.5)


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

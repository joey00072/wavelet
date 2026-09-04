from __future__ import annotations

import pickle
import random
from itertools import islice

import pytest

from wavelet.configs.rl_config import RLDataConfig
from wavelet.configs.sft import LossMaskConfig
from wavelet.data.rl import RLDataset, RLExample
from wavelet.data.sft import Example, SFTDataset


def _count_shuffles(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    original = random.Random.shuffle

    def counting_shuffle(self: random.Random, values: list[int]) -> None:
        calls[0] += 1
        original(self, values)

    monkeypatch.setattr(random.Random, "shuffle", counting_shuffle)
    return calls


def _sft_dataset() -> SFTDataset:
    return SFTDataset(
        records=[Example(prompt=[], completion=[]) for _ in range(16)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        loss_mask_config=LossMaskConfig(),
        shuffle=True,
        seed=3,
    )


def test_order_for_epoch_shuffles_once_per_epoch(monkeypatch) -> None:
    calls = _count_shuffles(monkeypatch)
    dataset = _sft_dataset()

    orders = [dataset._order_for_epoch(0) for _ in range(20)]

    assert calls[0] == 1
    assert all(order == orders[0] for order in orders)
    assert sorted(orders[0]) == list(range(16))

    next_order = dataset._order_for_epoch(1)
    assert calls[0] == 2
    assert next_order != orders[0]
    assert dataset._order_for_epoch(1) is next_order
    assert calls[0] == 2


def test_local_record_indexes_do_not_reshuffle_per_sample(monkeypatch) -> None:
    calls = _count_shuffles(monkeypatch)
    dataset = _sft_dataset()

    indexes = list(islice(dataset._local_record_indexes(), 16))

    assert sorted(indexes) == list(range(16))
    assert calls[0] == 1


def test_order_cache_matches_uncached_order_and_survives_state_round_trip(
    monkeypatch,
) -> None:
    dataset = _sft_dataset()
    expected = list(range(16))
    random.Random(3 + 2).shuffle(expected)

    dataset._order_for_epoch(0)
    dataset.load_state_dict({"step": 40, "epoch": 2})

    assert "_order_cache" not in dataset.state_dict()
    assert dataset._order_for_epoch(2) == expected
    assert pickle.loads(pickle.dumps(dataset))._order_for_epoch(2) == expected


def test_rl_loss_scale_lookahead_does_not_reshuffle_per_sample(monkeypatch) -> None:
    calls = _count_shuffles(monkeypatch)
    dataset = RLDataset(
        records=[
            RLExample(
                prompt=[],
                completion=[],
                advantage=1.0,
                reward=1.0,
                input_ids=[1, 2],
                target_ids=[2, 3],
                loss_mask=[True, True],
                inference_logprobs=[-1.0, -1.0],
                temperatures=[1.0, 1.0],
            )
            for _ in range(8)
        ],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=RLDataConfig(seq_len=8),
        shuffle=True,
    )

    assert dataset.loss_scale_for_next_local_batch(8) == 16
    assert calls[0] == 1

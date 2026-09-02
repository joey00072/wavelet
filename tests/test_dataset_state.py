from itertools import islice

from wavelet.configs.rl_config import RLDataConfig
from wavelet.configs.sft import LossMaskConfig
from wavelet.data.rl import PackedRLDataset, RLDataset, RLExample
from wavelet.data.sft import CatDataset, Example, SFTDataset
from wavelet.trainer.debug import build_debug_tokenizer


def _datasets():
    return [
        SFTDataset(
            records=[],
            tokenizer=None,  # type: ignore[arg-type]
            seq_len=8,
            loss_mask_config=LossMaskConfig(),
            data_rank=1,
            data_world_size=4,
        ),
        RLDataset(
            records=[],
            tokenizer=None,  # type: ignore[arg-type]
            seq_len=8,
            data_config=RLDataConfig(seq_len=8),
            data_rank=1,
            data_world_size=4,
        ),
        PackedRLDataset(
            records=[],
            tokenizer=None,  # type: ignore[arg-type]
            seq_len=8,
            data_config=RLDataConfig(seq_len=8, pack_sequences=True),
            data_rank=1,
            data_world_size=4,
        ),
    ]


def test_stateful_datasets_share_checkpoint_and_stats_behavior() -> None:
    for dataset in _datasets():
        dataset.load_state_dict(
            {
                "step": 12,
                "epoch": 3,
                "num_samples": {"train": 5},
                "num_tokens": {"train": 40},
                "skipped": 2,
            }
        )

        expected_state = {
            "step": 12,
            "epoch": 3,
            "num_samples": {"train": 5},
            "num_tokens": {"train": 40},
            "skipped": 2,
        }
        actual_state = dataset.state_dict()
        assert {key: actual_state[key] for key in expected_state} == expected_state
        if isinstance(dataset, PackedRLDataset):
            assert actual_state["next_bin_index"] == -1
        assert dataset.stats() == {
            "samples": {"train": 5},
            "tokens": {"train": 40},
            "skipped": 2,
        }
        assert dataset._effective_data_partition() == (1, 4)


def test_shared_local_iterator_preserves_rank_and_epoch_progress() -> None:
    dataset = SFTDataset(
        records=[Example(prompt=[], completion=[]) for _ in range(4)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        loss_mask_config=LossMaskConfig(),
        data_rank=1,
        data_world_size=2,
    )

    indexes = list(islice(dataset._local_record_indexes(), 3))

    assert indexes == [1, 3, 1]
    assert dataset.step == 6
    assert dataset.epoch == 1


def test_cat_dataset_resume_preserves_pending_token_remainder() -> None:
    tokenizer = build_debug_tokenizer(model_max_length=256)
    records = [
        Example(
            prompt=[{"role": "user", "content": "question"}],
            completion=[{"role": "assistant", "content": "answer" * 4}],
            source="test",
        )
    ]

    uninterrupted = CatDataset(
        SFTDataset(
            records=records,
            tokenizer=tokenizer,
            seq_len=64,
            loss_mask_config=LossMaskConfig(),
        ),
        seq_len=32,
    )
    iterator = iter(uninterrupted)
    next(iterator)
    state = uninterrupted.state_dict()
    expected_next = next(iterator)

    resumed = CatDataset(
        SFTDataset(
            records=records,
            tokenizer=tokenizer,
            seq_len=64,
            loss_mask_config=LossMaskConfig(),
        ),
        seq_len=32,
    )
    resumed.load_state_dict(state)

    assert state["pending"]["input_ids"]
    assert next(iter(resumed)) == expected_next


def test_packed_rl_resume_continues_at_next_bin() -> None:
    records = [
        RLExample(
            prompt=[],
            completion=[],
            advantage=float(index),
            reward=float(index),
            input_ids=list(range(index * 10, index * 10 + 6)),
            target_ids=list(range(index * 10 + 1, index * 10 + 7)),
            loss_mask=[True] * 6,
            inference_logprobs=[-1.0] * 6,
            temperatures=[1.0] * 6,
        )
        for index in range(4)
    ]
    config = RLDataConfig(pack_sequences=True, seq_len=8)
    uninterrupted = PackedRLDataset(
        records=records,
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=config,
    )
    iterator = iter(uninterrupted)
    first = next(iterator)
    state = uninterrupted.state_dict()
    expected_next = next(iterator)

    resumed = PackedRLDataset(
        records=records,
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=config,
    )
    resumed.load_state_dict(state)

    assert state["next_bin_index"] == 1
    assert next(iter(resumed)) == expected_next
    assert first != expected_next

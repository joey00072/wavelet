from itertools import islice

from wavelet.configs.rl_config import RLDataConfig
from wavelet.configs.sft import LossMaskConfig
from wavelet.data.rl import PackedRLDataset, RLDataset
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

        assert dataset.state_dict() == {
            "step": 12,
            "epoch": 3,
            "num_samples": {"train": 5},
            "num_tokens": {"train": 40},
            "skipped": 2,
        }
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

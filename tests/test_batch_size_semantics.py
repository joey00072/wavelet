from __future__ import annotations

import pytest
import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import SFTConfig
from wavelet.data.rl_dataset import PackedRLDataset, RLExample
from wavelet.distributed.parallel_dims import ParallelDims
from wavelet.distributed.world import World
from wavelet.trainer.rl_trainer import RLTrainer
from wavelet.trainer.sft import SFTTrainer


def _world(world_size: int) -> World:
    return World(
        rank=0,
        local_rank=0,
        world_size=world_size,
        local_world_size=world_size,
        device=torch.device("cpu"),
    )


def test_sft_batch_size_is_global_across_distributed_ranks() -> None:
    trainer = SFTTrainer(
        SFTConfig(
            data={
                "batch_size": 512,
                "micro_batch_size": 1,
            }
        )
    )
    trainer.world = _world(4)

    trainer._setup_accumulation_steps()

    assert trainer.accumulation_steps == 128


def test_sft_global_batch_must_divide_distributed_micro_batch() -> None:
    trainer = SFTTrainer(
        SFTConfig(
            data={
                "batch_size": 510,
                "micro_batch_size": 2,
            }
        )
    )
    trainer.world = _world(4)

    with pytest.raises(ValueError, match="global optimizer batch size"):
        trainer._setup_accumulation_steps()


def test_sft_gradient_clipping_uses_model_wrapper_operation() -> None:
    trainer = SFTTrainer(SFTConfig(max_grad_norm=0.25))

    class WrappedModel:
        def __init__(self) -> None:
            self.max_norm: float | None = None

        def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
            self.max_norm = max_norm
            return torch.tensor(1.5)

    model = WrappedModel()
    trainer.model = model  # type: ignore[assignment]

    assert trainer._clip_grad_norm() == 1.5
    assert model.max_norm == 0.25


def test_rl_batch_size_is_global_across_distributed_ranks() -> None:
    trainer = RLTrainer(
        RLConfig(
            data={
                "batch_size": 512,
                "micro_batch_size": 1,
            }
        )
    )
    trainer.world = _world(4)

    trainer._setup_accumulation_steps()

    assert trainer.accumulation_steps == 128


def test_rl_batch_size_uses_data_parallel_world_with_tensor_parallel() -> None:
    trainer = RLTrainer(
        RLConfig(
            data={
                "batch_size": 32,
                "micro_batch_size": 1,
            },
            fsdp={
                "enabled": True,
                "tp": 4,
            },
        )
    )
    trainer.world = _world(8)
    trainer.parallel_dims = ParallelDims(tp=4, world_size=8)

    trainer._setup_accumulation_steps()

    assert trainer._data_partition() == (0, 2)
    assert trainer.accumulation_steps == 16


def test_data_partition_ignores_tensor_parallel_rank() -> None:
    trainer = RLTrainer(RLConfig(fsdp={"enabled": True, "tp": 4}))
    trainer.world = World(
        rank=5,
        local_rank=5,
        world_size=8,
        local_world_size=8,
        device=torch.device("cpu"),
    )
    trainer.parallel_dims = ParallelDims(tp=4, world_size=8)

    assert trainer._data_partition() == (1, 2)


def test_tensor_parallel_metrics_count_once_per_data_parallel_group() -> None:
    leaders = []
    for rank in range(8):
        trainer = RLTrainer(RLConfig(fsdp={"enabled": True, "tp": 4}))
        trainer.world = World(
            rank=rank,
            local_rank=rank,
            world_size=8,
            local_world_size=8,
            device=torch.device("cpu"),
        )
        trainer.parallel_dims = ParallelDims(tp=4, world_size=8)
        if trainer._is_data_parallel_metric_leader():
            leaders.append(rank)

    assert leaders == [0, 4]


def test_tensor_parallel_metric_sync_does_not_multiply_counts(monkeypatch) -> None:
    trainer = RLTrainer(RLConfig(fsdp={"enabled": True, "tp": 4}))
    trainer.world = _world(8)
    trainer.parallel_dims = ParallelDims(tp=4, world_size=8)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def fake_gather_object(payload, gathered, dst):
        assert payload["rollout/count"] == 256.0
        gathered[:] = [
            payload,
            {},
            {},
            {},
            {
                "rollout/count": 256.0,
                "micro_batch/count": 128.0,
                "tokens/train": 2048.0,
                "reward_mean": 0.5,
            },
            {},
            {},
            {},
        ]

    monkeypatch.setattr(torch.distributed, "gather_object", fake_gather_object)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda _payload, src: None,
    )

    synced = trainer._sync_metrics(
        {
            "rollout/count": 256.0,
            "micro_batch/count": 128.0,
            "tokens/train": 2048.0,
            "reward_mean": 0.25,
        }
    )

    assert synced["rollout/count"] == 512.0
    assert synced["micro_batch/count"] == 256.0
    assert synced["tokens/train"] == 4096.0
    assert synced["reward_mean"] == 0.375


def test_packed_rl_accumulation_counts_dataloader_batches() -> None:
    config = RLConfig(
        data={
            "batch_size": 4,
            "micro_batch_size": 2,
            "pack_sequences": True,
            "seq_len": 8,
        }
    )
    trainer = RLTrainer(config)
    trainer.dataset = PackedRLDataset(
        records=[
            RLExample(
                prompt=[],
                completion=[],
                advantage=1.0,
                reward=1.0,
                input_ids=[1, 2, 3, 4, 5],
                target_ids=[2, 3, 4, 5, 6],
                loss_mask=[True, True, True, True, True],
                temperatures=1.0,
            )
            for _ in range(4)
        ],
        tokenizer=object(),
        seq_len=8,
        data_config=config.data,
    )

    assert trainer.dataset.micro_batch_count() == 4
    assert trainer._packed_dataloader_batch_count() == 2

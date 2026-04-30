from __future__ import annotations

import pytest
import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import SFTConfig
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

    trainer._setup_accumulation_steps()  # noqa: SLF001

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
        trainer._setup_accumulation_steps()  # noqa: SLF001


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

    trainer._setup_accumulation_steps()  # noqa: SLF001

    assert trainer.accumulation_steps == 128

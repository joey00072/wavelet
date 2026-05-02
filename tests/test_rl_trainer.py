from __future__ import annotations

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.trainer.rl_trainer import (
    RLTrainer,
    _packed_causal_attention_mask,
    _packed_training_attention_mask,
)


class _Config:
    def __init__(self, attn_implementation: str) -> None:
        self._attn_implementation = attn_implementation


class _Model:
    def __init__(self, attn_implementation: str) -> None:
        self.config = _Config(attn_implementation)


class _Wrapper:
    def __init__(self, module: object) -> None:
        self._fsdp_wrapped_module = module


def test_packed_causal_attention_mask_blocks_cross_sample_attention() -> None:
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    mask = _packed_causal_attention_mask(attention_mask, position_ids)

    assert mask is not None
    assert mask.shape == (1, 1, 5, 5)
    allowed = mask[0, 0] == 0
    assert allowed.tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [False, False, False, True, False],
        [False, False, False, True, True],
    ]


def test_unpacked_full_attention_mask_is_dropped() -> None:
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    position_ids = torch.arange(5).unsqueeze(0)

    mask = _packed_causal_attention_mask(attention_mask, position_ids)

    assert mask is None


def test_packed_reward_mean_is_weighted_by_rollout_count() -> None:
    trainer = RLTrainer(RLConfig())
    rewards = torch.tensor([0.0, 1.0])
    sample_counts = torch.tensor([1, 3])

    assert trainer._reward_mean(rewards, sample_counts=sample_counts) == 0.75  # noqa: SLF001


def test_packed_flash_attention_uses_varlen_position_ids() -> None:
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    mask = _packed_training_attention_mask(
        _Wrapper(_Model("flash_attention_2")),
        attention_mask,
        position_ids,
    )

    assert mask is None


def test_packed_flash_attention_rejects_padded_rows() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.long)

    try:
        _packed_training_attention_mask(
            _Model("flash_attention_2"),
            attention_mask,
            position_ids,
        )
    except ValueError as exc:
        assert "pad-free packed rows" in str(exc)
    else:
        raise AssertionError("expected padded packed FlashAttention rows to fail")

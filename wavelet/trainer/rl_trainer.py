"""Compatibility exports for the consolidated RL trainer module."""

from wavelet.trainer.rl import (
    RLTrainer,
    _packed_causal_attention_mask,
    _packed_training_attention_mask,
)

__all__ = [
    "RLTrainer",
    "_packed_causal_attention_mask",
    "_packed_training_attention_mask",
]

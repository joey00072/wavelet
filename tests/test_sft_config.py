from __future__ import annotations

import pytest
from pydantic import ValidationError

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import OptimizerConfig, SFTConfig


def test_legacy_optimizer_normalization_does_not_mutate_input() -> None:
    raw = {"betas": [0.8, 0.9]}

    config = OptimizerConfig.model_validate(raw)

    assert raw == {"betas": [0.8, 0.9]}
    assert config.betas1 == pytest.approx(0.8)
    assert config.betas2 == pytest.approx(0.9)


def test_legacy_rl_normalization_does_not_mutate_input() -> None:
    raw = {"sampling": {"max_tokens": 17}}

    config = RLConfig(inference=raw)

    assert raw == {"sampling": {"max_tokens": 17}}
    assert config.inference.sampling.max_completion_tokens == 17


def test_context_parallelism_requires_cat_packing() -> None:
    with pytest.raises(ValidationError, match="Packing function must be 'cat'"):
        SFTConfig(fsdp={"cp": 2}, data={"seq_len": 128, "pack_function": "pad"})

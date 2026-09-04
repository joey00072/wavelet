from __future__ import annotations

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import OptimizerConfig


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

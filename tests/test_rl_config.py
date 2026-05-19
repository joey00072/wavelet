from __future__ import annotations

import pytest

from wavelet.configs.rl_config import RLConfig, TokensLengthPenaltyConfig


def test_vllm_http_is_default_inference_mode() -> None:
    config = RLConfig()

    assert config.inference.mode == "vllm_http"


def test_sampling_max_tokens_maps_to_max_completion_tokens() -> None:
    config = RLConfig(inference={"sampling": {"max_tokens": 17}})

    assert config.inference.sampling.max_completion_tokens == 17


def test_legacy_length_penalty_string_maps_to_config() -> None:
    config = RLConfig(orchestrator={"length_penalty": "tokens"})

    assert isinstance(config.orchestrator.length_penalty, TokensLengthPenaltyConfig)
    assert config.orchestrator.length_penalty.completion_weight == 1.0


def test_legacy_rl_aliases_map_silently() -> None:
    config = RLConfig(data={"reference_logprobs_column": "ref"})
    assert config.data.inference_logprobs_column == "ref"

    config = RLConfig(loss={"advantage_scale": 0.5})
    assert config.loss.adv_tau == 0.5


def test_max_inflight_rollouts_must_cover_one_group() -> None:
    with pytest.raises(ValueError, match="max_inflight_rollouts"):
        RLConfig(
            orchestrator={
                "rollouts_per_example": 8,
                "max_inflight_rollouts": 4,
            }
        )

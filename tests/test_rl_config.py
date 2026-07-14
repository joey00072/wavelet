from __future__ import annotations

import pytest

from wavelet.configs.rl_config import (
    GRPOAlgorithmConfig,
    RLConfig,
    TokensLengthPenaltyConfig,
)


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


def test_legacy_group_reward_maps_to_grpo_algorithm() -> None:
    config = RLConfig(
        orchestrator={
            "advantage_mode": "group_reward",
            "normalize_group_advantages": True,
            "advantage_epsilon": 1e-5,
            "length_penalty": "turns",
        }
    )

    assert isinstance(config.algo, GRPOAlgorithmConfig)
    assert config.algo.normalize_advantages is True
    assert config.algo.epsilon == pytest.approx(1e-5)
    assert config.algo.length_penalty is not None


def test_explicit_algorithm_config_takes_precedence() -> None:
    config = RLConfig(
        algo={"type": "max_rl"},
        orchestrator={"advantage_mode": "group_reward"},
    )

    assert config.algo.type == "max_rl"


def test_algorithm_normalization_does_not_mutate_input() -> None:
    payload = {
        "algo": {
            "file": "custom.py",
            "algorithm": "example",
            "scope": "group",
        }
    }

    RLConfig.model_validate(payload)

    assert "type" not in payload["algo"]


def test_named_grpo_accepts_documented_token_cost_fields() -> None:
    config = RLConfig(
        algo={
            "type": "grpo",
            "length_penalty": {
                "type": "tokens",
                "completion_weight": 0.5,
                "tool_response_weight": 2.0,
            },
        }
    )

    assert isinstance(config.algo, GRPOAlgorithmConfig)
    assert isinstance(config.algo.length_penalty, TokensLengthPenaltyConfig)
    assert config.algo.length_penalty.completion_weight == pytest.approx(0.5)
    assert config.algo.length_penalty.tool_response_weight == pytest.approx(2.0)


def test_algorithm_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown_option"):
        RLConfig(algo={"type": "grpo", "unknown_option": True})


def test_max_inflight_rollouts_must_cover_one_group() -> None:
    with pytest.raises(ValueError, match="max_inflight_rollouts"):
        RLConfig(
            orchestrator={
                "rollouts_per_example": 8,
                "max_inflight_rollouts": 4,
            }
        )

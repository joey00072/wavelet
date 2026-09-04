from __future__ import annotations

import pytest

from wavelet.configs.rl_config import (
    GRPOAlgorithmConfig,
    LinearLengthPenaltyConfig,
    RLConfig,
    TokensLengthPenaltyConfig,
    TruncationLengthPenaltyConfig,
)


def test_vllm_http_is_default_inference_mode() -> None:
    config = RLConfig()

    assert config.inference.mode == "vllm_http"


def test_sampling_max_tokens_maps_to_max_completion_tokens() -> None:
    config = RLConfig(inference={"sampling": {"max_tokens": 17}})

    assert config.inference.sampling.max_completion_tokens == 17


def test_legacy_length_penalty_string_maps_to_algorithm() -> None:
    config = RLConfig(
        orchestrator={"advantage_mode": "group_reward", "length_penalty": "tokens"}
    )

    assert isinstance(config.algo, GRPOAlgorithmConfig)
    assert isinstance(config.algo.length_penalty, TokensLengthPenaltyConfig)
    assert config.algo.length_penalty.completion_weight == 1.0
    assert "length_penalty" not in config.orchestrator.model_dump()


def test_legacy_length_penalty_without_group_reward_is_rejected() -> None:
    # Previously accepted and then silently ignored: nothing read the field.
    with pytest.raises(ValueError, match="orchestrator.length_penalty"):
        RLConfig(orchestrator={"length_penalty": "tokens"})


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


def test_explicit_algorithm_config_rejects_legacy_advantage_keys() -> None:
    # Previously the explicit algo won and the legacy key was silently dropped.
    with pytest.raises(ValueError, match="orchestrator.advantage_mode"):
        RLConfig(
            algo={"type": "max_rl"},
            orchestrator={"advantage_mode": "group_reward"},
        )


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


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({"type": "linear"}, LinearLengthPenaltyConfig),
        ({"type": "truncation", "penalty": 0.5}, TruncationLengthPenaltyConfig),
    ],
)
def test_named_grpo_accepts_reward_length_penalties(payload, expected_type) -> None:
    config = RLConfig(algo={"type": "grpo", "length_penalty": payload})

    assert isinstance(config.algo, GRPOAlgorithmConfig)
    assert isinstance(config.algo.length_penalty, expected_type)


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


def test_tasks_per_minute_requires_verifier_rollout_source() -> None:
    with pytest.raises(ValueError, match="tasks_per_minute"):
        RLConfig(orchestrator={"tasks_per_minute": 60})

    config = RLConfig(
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
            "tasks_per_minute": 60,
        }
    )

    assert config.orchestrator.tasks_per_minute == 60


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_p", 0.9),
        ("top_k", 20),
        ("min_p", 0.1),
        ("min_tokens", 2),
        ("repetition_penalty", 1.1),
    ],
)
def test_rl_rejects_sampling_transforms_trainer_cannot_replay(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        RLConfig(inference={"sampling": {field: value}})


@pytest.mark.parametrize(
    "field",
    ["temperature", "seed", "logit_bias", "presence_penalty"],
)
def test_rl_rejects_hidden_sampling_transforms_in_extra_body(field: str) -> None:
    with pytest.raises(ValueError, match=f"extra_body.{field}"):
        RLConfig(inference={"sampling": {"extra_body": {field: 1}}})


def test_static_rl_data_allows_sampling_fields_that_are_not_used() -> None:
    config = RLConfig(
        orchestrator={"enabled": False},
        inference={"sampling": {"min_p": 0.1}},
    )

    assert config.inference.sampling.min_p == pytest.approx(0.1)


@pytest.mark.parametrize(
    "sampling",
    [
        {"do_sample": False},
        {"temperature": 0.0},
        {"seed": 123},
    ],
)
def test_group_relative_rl_rejects_deterministic_rollouts(sampling) -> None:
    with pytest.raises(ValueError, match="needs diverse rollouts"):
        RLConfig(
            algo={"type": "grpo"},
            orchestrator={"rollouts_per_example": 8},
            inference={"sampling": sampling},
            max_steps=1,
        )


def test_eval_only_group_config_allows_deterministic_sampling() -> None:
    config = RLConfig(
        algo={"type": "grpo"},
        orchestrator={"rollouts_per_example": 8},
        inference={"sampling": {"seed": 123}},
        max_steps=0,
    )

    assert config.inference.sampling.seed == 123


def test_eval_sampling_forwards_reasoning_effort() -> None:
    config = RLConfig(eval={"sampling": {"reasoning_effort": "high"}})

    assert config.eval is not None
    assert config.eval.sampling.to_sampling_args()["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "sampling",
    [
        {"do_sample": False},
        {"temperature": 0.0},
    ],
)
def test_single_rollout_online_rl_rejects_zero_effective_temperature(
    sampling,
) -> None:
    with pytest.raises(ValueError, match="positive temperature"):
        RLConfig(
            algo={"type": "reward"},
            orchestrator={"rollouts_per_example": 1},
            inference={"sampling": sampling},
            max_steps=1,
        )


def test_eval_only_reward_config_allows_greedy_sampling() -> None:
    config = RLConfig(
        algo={"type": "reward"},
        inference={"sampling": {"do_sample": False}},
        max_steps=0,
    )

    assert config.inference.sampling.do_sample is False


@pytest.mark.parametrize("mode", ["process", "colocate", "colocate_sleep"])
def test_process_training_requires_initial_policy_export(mode: str) -> None:
    with pytest.raises(ValueError, match="export_initial=true"):
        RLConfig(
            launcher={"mode": mode},
            policy_transfer={"export_initial": False},
            max_steps=1,
        )


def test_process_eval_only_does_not_require_initial_policy_export() -> None:
    config = RLConfig(
        launcher={"mode": "process"},
        policy_transfer={"export_initial": False},
        max_steps=0,
    )

    assert config.policy_transfer.export_initial is False


def test_policy_transport_retains_current_and_previous_by_default() -> None:
    config = RLConfig()

    assert config.policy_transfer.keep_last == 2


def test_policy_transport_rejects_single_snapshot_retention() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        RLConfig(policy_transfer={"keep_last": 1})


def test_rollout_transport_keeps_a_bounded_audit_window_by_default() -> None:
    config = RLConfig()

    assert config.transport.cleanup_consumed is True
    assert config.transport.keep_last_consumed == 2


def test_checkpoint_and_eval_artifacts_are_bounded_by_default() -> None:
    config = RLConfig(
        ckpt={"mode": "async", "interval": 1},
        eval={"env": [{"id": "aime"}]},
    )

    assert config.ckpt is not None
    assert config.ckpt.keep_last == 2
    assert config.eval is not None
    assert config.eval.keep_last_rollout_sets == 2

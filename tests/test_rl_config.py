from __future__ import annotations

import pytest

from wavelet.configs.rl_config import (
    GRPOAlgorithmConfig,
    LinearLengthPenaltyConfig,
    OPDAlgorithmConfig,
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


def test_rollout_batch_targets_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="Set only one"):
        RLConfig(orchestrator={"examples_per_step": 4, "token_batch_size": 1024})


def test_token_batches_require_streaming_verifier_scheduler() -> None:
    base = {
        "token_batch_size": 1024,
        "custom_rollout_function": ("wavelet.orchestrator.verifiers:generate_rollouts"),
        "max_async_level": 1,
    }
    with pytest.raises(ValueError, match="launcher.mode='process'"):
        RLConfig(orchestrator=base)
    with pytest.raises(ValueError, match="Verifiers rollout source"):
        RLConfig(
            launcher={"mode": "process"},
            orchestrator={
                "token_batch_size": 1024,
                "max_async_level": 1,
            },
        )

    config = RLConfig(launcher={"mode": "process"}, orchestrator=base)

    assert config.orchestrator.examples_per_step is None
    assert config.orchestrator.token_batch_size == 1024


def test_token_batches_reject_fixed_chunks_and_checkpoint_resume() -> None:
    base = {
        "token_batch_size": 1024,
        "custom_rollout_function": ("wavelet.orchestrator.verifiers:generate_rollouts"),
        "max_async_level": 1,
    }
    with pytest.raises(ValueError, match="rollout_chunk_examples"):
        RLConfig(
            launcher={"mode": "process"},
            orchestrator={**base, "rollout_chunk_examples": 2},
        )
    with pytest.raises(ValueError, match="variable record cursor"):
        RLConfig(
            launcher={"mode": "process"},
            orchestrator=base,
            ckpt={"resume_step": 1},
        )


def test_adaptive_concurrency_requires_streaming_verifier_scheduler() -> None:
    with pytest.raises(ValueError, match="Verifiers rollout source"):
        RLConfig(
            launcher={"mode": "process"},
            orchestrator={"max_async_level": 1, "concurrency": {}},
        )
    with pytest.raises(ValueError, match="launcher.mode='process'"):
        RLConfig(
            orchestrator={
                "custom_rollout_function": (
                    "wavelet.orchestrator.verifiers:generate_rollouts"
                ),
                "max_async_level": 1,
                "concurrency": {},
            },
        )


def test_adaptive_concurrency_allows_minimum_below_one_rollout_group() -> None:
    config = RLConfig(
        launcher={"mode": "process"},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
            "max_async_level": 1,
            "rollouts_per_example": 8,
            "concurrency": {"min_inflight": 4},
        },
    )

    assert config.orchestrator.concurrency is not None
    assert config.orchestrator.concurrency.min_inflight == 4


def test_adaptive_concurrency_validates_bounds_and_thresholds() -> None:
    with pytest.raises(ValueError, match="min_inflight must not exceed"):
        RLConfig.model_validate(
            {
                "launcher": {"mode": "process"},
                "orchestrator": {
                    "custom_rollout_function": (
                        "wavelet.orchestrator.verifiers:generate_rollouts"
                    ),
                    "max_async_level": 1,
                    "concurrency": {"min_inflight": 8, "max_inflight": 4},
                },
            }
        )
    with pytest.raises(ValueError, match="cannot exceed the explicit"):
        RLConfig.model_validate(
            {
                "launcher": {"mode": "process"},
                "orchestrator": {
                    "custom_rollout_function": (
                        "wavelet.orchestrator.verifiers:generate_rollouts"
                    ),
                    "max_async_level": 1,
                    "max_inflight_rollouts": 8,
                    "concurrency": {"max_inflight": 16},
                },
            }
        )
    with pytest.raises(ValueError, match="growth < target < hard"):
        RLConfig.model_validate(
            {
                "launcher": {"mode": "process"},
                "orchestrator": {
                    "custom_rollout_function": (
                        "wavelet.orchestrator.verifiers:generate_rollouts"
                    ),
                    "max_async_level": 1,
                    "concurrency": {
                        "growth_kv_cache_usage": 0.8,
                        "target_kv_cache_usage": 0.7,
                    },
                },
            }
        )


def test_background_evals_require_streaming_verifier_scheduler() -> None:
    with pytest.raises(ValueError, match="Verifiers rollout source"):
        RLConfig(
            launcher={"mode": "process"},
            orchestrator={"max_async_level": 1},
            eval={"background": True},
        )
    with pytest.raises(ValueError, match="launcher.mode='process'"):
        RLConfig(
            orchestrator={
                "custom_rollout_function": (
                    "wavelet.orchestrator.verifiers:generate_rollouts"
                ),
                "max_async_level": 1,
            },
            eval={"background": True},
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
    "field_value",
    [("top_p", 0.9), ("top_k", 20), ("min_p", 0.1)],
)
def test_rl_replays_sampling_support_masks(field_value: tuple[str, float]) -> None:
    field, value = field_value
    config = RLConfig(inference={"sampling": {field: value}})
    assert getattr(config.inference.sampling, field) == pytest.approx(value)


@pytest.mark.parametrize(
    ("algorithm", "sampling"),
    [
        (
            {
                "type": "opd",
                "teacher": {
                    "name": "teacher",
                    "base_url": "http://teacher:8001/v1",
                },
            },
            {"top_p": 0.9},
        ),
        ({"type": "opsd"}, {"top_k": 20}),
        ({"type": "opsd"}, {"min_p": 0.1}),
    ],
)
def test_ref_kl_distillation_rejects_truncated_sampling(
    algorithm: dict[str, object], sampling: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="not supported with truncated train sampling"):
        RLConfig(
            algo=algorithm,
            orchestrator={
                "custom_rollout_function": (
                    "wavelet.orchestrator.verifiers:generate_rollouts"
                ),
                "verifier_env_id": "test-env",
            },
            inference={"sampling": sampling},
        )


def test_rl_rejects_disabling_required_sampling_masks() -> None:
    with pytest.raises(ValueError, match="return_sampling_mask=true"):
        RLConfig(
            inference={
                "sampling": {"top_p": 0.9},
                "vllm": {"return_sampling_mask": False},
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "seed",
        "logit_bias",
        "presence_penalty",
        "top_p",
        "top_k",
        "min_p",
    ],
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


def _multi_environment_config(**orchestrator_overrides) -> RLConfig:
    orchestrator = {
        "custom_rollout_function": ("wavelet.orchestrator.verifiers:generate_rollouts"),
        "examples_per_step": 4,
        "rollouts_per_example": 2,
        "max_async_level": 1,
        "envs": [
            {"id": "math@1", "ratio": 1.0},
            {
                "id": "code@2",
                "name": "code",
                "ratio": 3.0,
                "group_size": 4,
                "sampling": {"temperature": 0.7},
                "algo": {"type": "grpo"},
            },
        ],
    }
    orchestrator.update(orchestrator_overrides)
    return RLConfig(
        algo={"type": "reward"},
        launcher={"mode": "process"},
        inference={"sampling": {"max_completion_tokens": 128}},
        orchestrator=orchestrator,
    )


def test_multiple_training_environments_resolve_overrides() -> None:
    config = _multi_environment_config()

    math, code = config.orchestrator.envs
    assert math.resolved_name == "math"
    assert code.resolved_name == "code"
    assert code.group_size == 4
    assert code.algo is not None and code.algo.type == "grpo"
    sampling = config.resolved_train_sampling(code)
    assert sampling.temperature == pytest.approx(0.7)
    assert sampling.max_completion_tokens == 128


def test_multiple_training_environments_reject_legacy_environment() -> None:
    with pytest.raises(ValueError, match="not both"):
        _multi_environment_config(verifier_env_id="legacy")


def test_multiple_training_environments_require_unique_names() -> None:
    with pytest.raises(ValueError, match="Duplicate training environment names"):
        _multi_environment_config(
            envs=[{"id": "same@1"}, {"id": "same@2"}],
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"custom_rollout_function": "example:generate"},
        {"max_async_level": 0},
    ],
)
def test_multiple_training_environments_require_async_verifiers(overrides) -> None:
    with pytest.raises(ValueError, match="orchestrator.envs requires"):
        _multi_environment_config(**overrides)


def test_multiple_training_environments_bound_largest_group() -> None:
    with pytest.raises(ValueError, match="largest environment group_size=4"):
        _multi_environment_config(max_inflight_rollouts=3)


def test_multiple_training_environments_validate_effective_sampling() -> None:
    config = _multi_environment_config(
        envs=[
            {"id": "math"},
            {"id": "code", "sampling": {"top_p": 0.9}},
        ],
    )
    assert config.orchestrator.envs[1].sampling.top_p == pytest.approx(0.9)


def test_multiple_training_environments_reject_mixed_sft_generation() -> None:
    with pytest.raises(ValueError, match="SFT distillation cannot be mixed"):
        _multi_environment_config(
            envs=[
                {"id": "math"},
                {"id": "distill", "algo": {"type": "sft"}},
            ]
        )


def test_multiple_training_environments_allow_mixed_reward_and_opd() -> None:
    config = _multi_environment_config(
        envs=[
            {"id": "reward"},
            {
                "id": "distill",
                "algo": {
                    "type": "opd",
                    "teacher": {
                        "name": "teacher",
                        "base_url": "http://teacher:8001/v1",
                    },
                },
            },
        ]
    )

    assert config.orchestrator.envs[0].algo is None
    distill = config.orchestrator.envs[1].algo
    assert isinstance(distill, OPDAlgorithmConfig)
    assert distill.teacher.name == "teacher"


def test_curriculum_parses_difficulty_pool_and_advantage_gate() -> None:
    config = _multi_environment_config(
        envs=[
            {
                "id": "math",
                "curriculum": {
                    "sampler": {
                        "type": "difficulty_pool",
                        "ema_alpha": 0.5,
                    },
                    "gates": {
                        "zero_signal": {"type": "advantage_range"},
                    },
                },
            }
        ]
    )

    curriculum = config.orchestrator.envs[0].curriculum
    assert curriculum is not None
    assert curriculum.sampler.type == "difficulty_pool"
    assert curriculum.gates["zero_signal"].reject_min == 0.0


def test_curriculum_rejects_all_zero_pool_weights() -> None:
    with pytest.raises(ValueError, match="positive weight"):
        _multi_environment_config(
            envs=[
                {
                    "id": "math",
                    "curriculum": {
                        "sampler": {
                            "type": "difficulty_pool",
                            "pools": {
                                "hard": {"threshold": 1.0, "weight": 0.0},
                            },
                        }
                    },
                }
            ]
        )


def test_legacy_curriculum_requires_async_verifiers() -> None:
    with pytest.raises(ValueError, match="Curriculum requires"):
        RLConfig(orchestrator={"curriculum": {}})

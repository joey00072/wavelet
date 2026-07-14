from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wavelet.configs.sft import (
    TrainingDataConfig,
    TrainerConfig,
)


def _normalize_legacy_sampling_fields(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    if "max_tokens" in normalized and "max_completion_tokens" not in normalized:
        normalized["max_completion_tokens"] = normalized.pop("max_tokens")
    return normalized


class RLDataConfig(TrainingDataConfig):
    path: Path | list[Path] = Path("outputs/unsloth_math_data/rl_train.jsonl")
    pack_sequences: bool = False
    pad_to_multiple_of: int = Field(default=1, ge=1)
    advantage_column: str = "advantage"
    reward_column: str = "reward"
    inference_logprobs_column: str = "inference_logprobs"
    teacher_logprobs_column: str = "teacher_logprobs"
    temperature_column: str = "temperature"
    metadata_column: str = "metadata"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_columns(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if (
            "reference_logprobs_column" in normalized
            and "inference_logprobs_column" not in normalized
        ):
            normalized["inference_logprobs_column"] = normalized.pop(
                "reference_logprobs_column"
            )
        return normalized


class RLLossConfig(BaseModel):
    type: Literal["dppo"] = "dppo"
    dppo_mask_high: float = Field(default=0.20, ge=0.0)
    dppo_mask_low: float = Field(default=0.20, ge=0.0)
    kl_tau: float = Field(default=1e-3, ge=0.0)
    adv_tau: float = Field(default=1.0, ge=0.0)
    teacher_tau: float = Field(default=0.0, ge=0.0)
    normalization: Literal["token", "sequence"] = "token"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_loss_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "advantage_scale" in normalized and "adv_tau" not in normalized:
            normalized["adv_tau"] = normalized.pop("advantage_scale")
        return normalized


class RLTransportConfig(BaseModel):
    type: Literal["filesystem"] = "filesystem"
    rollout_filename: str = "rollouts.jsonl"
    queue_dir: Path | None = None
    poll_interval_seconds: float = Field(default=1.0, gt=0.0)
    idle_timeout_seconds: float | None = Field(default=None, gt=0.0)
    cleanup_consumed: bool = False


class RLPolicyTransferConfig(BaseModel):
    type: Literal["filesystem", "nccl"] = "filesystem"
    policy_dir: Path | None = None
    adapter_name: str = "policy"
    adapter_id: int = Field(default=1, ge=1)
    poll_interval_seconds: float = Field(default=1.0, gt=0.0)
    idle_timeout_seconds: float | None = Field(default=None, gt=0.0)
    export_initial: bool = False
    export_every_steps: int = Field(default=1, ge=1)
    lightweight_lora: bool = True
    nccl_host: str = "127.0.0.1"
    nccl_port: int = Field(default=29501, ge=1, le=65535)
    nccl_timeout_seconds: int = Field(default=600, ge=1)
    nccl_inference_world_size: int = Field(default=1, ge=1)
    nccl_rank_offset: int = Field(default=1, ge=1)


class RLSamplingConfig(BaseModel):
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    min_tokens: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    max_prompt_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    do_sample: bool = True
    num_generations: int = Field(default=1, ge=1)
    seed: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_sampling_fields(cls, value: object) -> object:
        return _normalize_legacy_sampling_fields(value)


class RLEvalSamplingConfig(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=-1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    min_tokens: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_sampling_fields(cls, value: object) -> object:
        return _normalize_legacy_sampling_fields(value)

    def to_sampling_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {"logprobs": True}
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.top_p is not None:
            args["top_p"] = self.top_p
        if self.max_completion_tokens is not None:
            args["max_completion_tokens"] = self.max_completion_tokens
        if self.seed is not None:
            args["seed"] = self.seed

        extra_body = dict(self.extra_body)
        extra_body["return_token_ids"] = True
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.min_tokens is not None:
            extra_body["min_tokens"] = self.min_tokens
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty
        args["extra_body"] = extra_body
        return args


class RLEvalEnvConfig(BaseModel):
    id: str
    name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    sampling: RLEvalSamplingConfig = RLEvalSamplingConfig()
    num_examples: int = -1
    rollouts_per_example: int = Field(default=1, ge=1)
    interval: int = Field(default=100, ge=1)
    max_retries: int = Field(default=0, ge=0)

    @property
    def resolved_name(self) -> str:
        return self.name or self.id.split("@", 1)[0]


class RLEvalConfig(BaseModel):
    env: list[RLEvalEnvConfig] = Field(default_factory=list)
    sampling: RLEvalSamplingConfig = RLEvalSamplingConfig()
    num_examples: int = -1
    rollouts_per_example: int = Field(default=1, ge=1)
    interval: int = Field(default=100, ge=1)
    max_retries: int = Field(default=0, ge=0)
    eval_base_model: bool = True
    final_eval: bool = True

    @model_validator(mode="after")
    def resolve_env_defaults(self) -> "RLEvalConfig":
        group_sampling = self.sampling.model_dump()
        for env in self.env:
            if "sampling" not in env.model_fields_set:
                env.sampling = RLEvalSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | env.sampling.model_dump(exclude_unset=True)
                env.sampling = RLEvalSamplingConfig(**merged)
            if "num_examples" not in env.model_fields_set:
                env.num_examples = self.num_examples
            if "rollouts_per_example" not in env.model_fields_set:
                env.rollouts_per_example = self.rollouts_per_example
            if "interval" not in env.model_fields_set:
                env.interval = self.interval
            if "max_retries" not in env.model_fields_set:
                env.max_retries = self.max_retries

        names = [env.resolved_name for env in self.env]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate evaluation environment names: {duplicates}")
        return self


class RLVLLMConfig(BaseModel):
    server_backend: Literal["offline", "openai"] = "openai"
    gpu_memory_utilization: float = Field(default=0.35, gt=0.0, le=1.0)
    max_model_len: int | None = Field(default=None, ge=8)
    quantization: str | None = None
    load_format: str | None = None
    tensor_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size_local: int | None = Field(default=None, ge=1)
    data_parallel_rpc_port: int | None = Field(default=None, ge=1, le=65535)
    enforce_eager: bool = True
    max_lora_rank: int | None = Field(default=None, ge=1)
    fully_sharded_loras: bool = False
    trust_remote_code: bool | None = None
    dtype: Literal["auto", "float32", "float16", "bfloat16"] | None = None
    tool_call_parser: str | None = "auto"
    reasoning_parser: str | None = None
    use_generation_logprobs: bool = True
    openai_batch_wait_seconds: float = Field(default=0.01, ge=0.0)
    openai_batch_min_size: int = Field(default=1, ge=1)
    openai_batch_max_wait_seconds: float = Field(default=0.01, ge=0.0)
    openai_batch_max_size: int | None = Field(default=None, ge=1)


class RLVLLMHTTPConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    ports: list[int] | None = None
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    startup_timeout_seconds: float = Field(default=300.0, gt=0.0)


class RLInferenceConfig(BaseModel):
    enabled: bool = True
    mode: Literal["passthrough", "vllm_http"] = "vllm_http"
    default_temperature: float = Field(default=1.0, gt=0.0)
    sampling: RLSamplingConfig = RLSamplingConfig()
    vllm: RLVLLMConfig = RLVLLMConfig()
    http: RLVLLMHTTPConfig = RLVLLMHTTPConfig()


class RLRewardConfig(BaseModel):
    mode: Literal[
        "passthrough",
        "reference_match",
        "math_format",
    ] = "passthrough"
    normalize_whitespace: bool = True
    case_sensitive: bool = True
    reasoning_start: str = "<start_working_out>"
    reasoning_end: str = "<end_working_out>"
    solution_start: str = "<SOLUTION>"
    solution_end: str = "</SOLUTION>"
    rescale_min: float | None = None
    rescale_max: float | None = None
    clamp_min: float | None = None
    clamp_max: float | None = None

    @model_validator(mode="after")
    def validate_reward_postprocessing(self) -> "RLRewardConfig":
        if (self.rescale_min is None) != (self.rescale_max is None):
            raise ValueError(
                "reward.rescale_min and reward.rescale_max must be set together"
            )
        if (
            self.rescale_min is not None
            and self.rescale_max is not None
            and self.rescale_max <= self.rescale_min
        ):
            raise ValueError(
                "reward.rescale_max must be greater than reward.rescale_min"
            )
        if (
            self.clamp_min is not None
            and self.clamp_max is not None
            and self.clamp_max < self.clamp_min
        ):
            raise ValueError(
                "reward.clamp_max must be greater than or equal to reward.clamp_min"
            )
        return self


class RLStateServerConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = (
        "warning"
    )
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_events: int = Field(default=2000, ge=100)


class _StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokensLengthPenaltyConfig(_StrictConfig):
    type: Literal["tokens"] = "tokens"
    completion_weight: float = Field(default=1.0, ge=0.0)
    tool_response_weight: float = Field(default=1.0, ge=0.0)


class TurnsLengthPenaltyConfig(_StrictConfig):
    type: Literal["turns"] = "turns"


LengthPenaltyConfig = Annotated[
    TokensLengthPenaltyConfig | TurnsLengthPenaltyConfig,
    Field(discriminator="type"),
]


def _normalize_length_penalty(value: object) -> object:
    if isinstance(value, str) and value in {"tokens", "turns"}:
        return {"type": value}
    return value


class PassthroughAlgorithmConfig(_StrictConfig):
    type: Literal["passthrough"] = "passthrough"


class RewardAlgorithmConfig(_StrictConfig):
    type: Literal["reward"] = "reward"


class GRPOAlgorithmConfig(_StrictConfig):
    type: Literal["grpo"] = "grpo"
    normalize_advantages: bool = False
    epsilon: float = Field(default=1e-6, gt=0.0)
    length_penalty: LengthPenaltyConfig | None = None


class MaxRLAlgorithmConfig(_StrictConfig):
    type: Literal["max_rl"] = "max_rl"


AlgorithmScope = Literal["rollout", "group", "both"]


class CustomAlgorithmConfig(_StrictConfig):
    """Select a registered algorithm from a user-owned Python file."""

    type: Literal["custom"] = "custom"
    file: Path
    algorithm: str = Field(min_length=1)
    scope: AlgorithmScope
    kwargs: dict[str, Any] = Field(default_factory=dict)
    epsilon: float = Field(default=1e-6, gt=0.0)


RLAlgorithmConfig = Annotated[
    PassthroughAlgorithmConfig
    | RewardAlgorithmConfig
    | GRPOAlgorithmConfig
    | MaxRLAlgorithmConfig
    | CustomAlgorithmConfig,
    Field(discriminator="type"),
]


class RLOrchestratorConfig(BaseModel):
    enabled: bool = True
    custom_rollout_function: str | None = None
    verifier_env_id: str | None = None
    verifier_env_args: dict[str, Any] = Field(default_factory=dict)
    verifier_model: str | None = None
    verifier_base_url: str | list[str] | None = None
    verifier_api_key_var: str = "PRIME_API_KEY"
    verifier_client_type: Literal[
        "openai_chat_completions",
        "openai_chat_completions_token",
    ] = "openai_chat_completions"
    verifier_max_retries: int = Field(default=0, ge=0)
    verifier_timeout_seconds: float | None = Field(default=None, gt=0.0)
    verifier_max_total_completion_tokens: int = -1
    materialize_path: Path | None = None
    overwrite: bool = True
    advantage_mode: Literal["passthrough", "reward", "group_reward"] = "passthrough"
    normalize_group_advantages: bool = False
    advantage_epsilon: float = Field(default=1e-6, gt=0.0)
    examples_per_step: int | None = Field(default=None, ge=1)
    rollouts_per_example: int | None = Field(default=None, ge=1)
    oversampling_factor: float = Field(default=1.0, ge=1.0)
    max_inflight_rollouts: int | None = Field(default=None, ge=1)
    rollout_chunk_examples: int | None = Field(default=None, ge=1)
    filter_zero_advantage: bool = True
    zero_advantage_max_retries: int = Field(default=8, ge=0)
    max_async_level: int = Field(default=0, ge=0)
    max_off_policy_steps: int = Field(default=0, ge=0)
    max_pending_rollout_chunks: int | None = Field(default=None, ge=1)
    length_penalty: LengthPenaltyConfig | None = None
    state_server: RLStateServerConfig = RLStateServerConfig()

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_orchestrator_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "length_penalty" in normalized:
            normalized["length_penalty"] = _normalize_length_penalty(
                normalized["length_penalty"]
            )
        return normalized

    @model_validator(mode="after")
    def validate_max_inflight_rollouts(self) -> "RLOrchestratorConfig":
        if self.max_inflight_rollouts is None or self.rollouts_per_example is None:
            return self
        if self.max_inflight_rollouts < self.rollouts_per_example:
            raise ValueError(
                "orchestrator.max_inflight_rollouts must be at least "
                "orchestrator.rollouts_per_example"
            )
        return self


class RLLauncherConfig(BaseModel):
    mode: Literal["integrated", "process", "colocate", "colocate_sleep"] = "integrated"
    backend: Literal["local", "ray"] = "local"
    trainer_cuda_visible_devices: str | list[str] | None = None
    inference_cuda_visible_devices: str | list[str] | None = None
    trainer_num_processes: int = Field(default=1, ge=1)
    inference_num_replicas: int = Field(default=1, ge=1)
    ray_address: str | None = None
    ray_runtime_env: dict[str, Any] | None = None
    poll_interval_seconds: float = Field(default=1.0, gt=0.0)
    colocate_memory_wait_timeout_seconds: float = Field(default=120.0, ge=0.0)
    colocate_memory_wait_poll_seconds: float = Field(default=0.5, gt=0.0)
    colocate_memory_wait_margin: float = Field(default=0.05, ge=0.0, le=0.5)


def _normalize_algorithm_config(value: dict[str, Any]) -> dict[str, Any]:
    """Infer file-backed algorithms and translate legacy advantage settings."""
    normalized = dict(value)
    if "algo" in normalized:
        algo = normalized["algo"]
        if isinstance(algo, dict) and "type" not in algo:
            if "file" in algo or "algorithm" in algo:
                normalized["algo"] = {"type": "custom", **algo}
        return normalized

    orchestrator = normalized.get("orchestrator")
    if not isinstance(orchestrator, dict):
        return normalized
    mode = orchestrator.get("advantage_mode")
    if mode is None:
        return normalized
    if mode != "group_reward":
        normalized["algo"] = {"type": mode}
        return normalized

    normalized["algo"] = {
        "type": "grpo",
        "normalize_advantages": orchestrator.get("normalize_group_advantages", False),
        "epsilon": orchestrator.get("advantage_epsilon", 1e-6),
        "length_penalty": _normalize_length_penalty(orchestrator.get("length_penalty")),
    }
    return normalized


class RLConfig(TrainerConfig):
    data: RLDataConfig = RLDataConfig()
    loss: RLLossConfig = RLLossConfig()
    algo: RLAlgorithmConfig = PassthroughAlgorithmConfig()
    orchestrator: RLOrchestratorConfig = RLOrchestratorConfig()
    eval: RLEvalConfig | None = None
    inference: RLInferenceConfig = RLInferenceConfig()
    reward: RLRewardConfig = RLRewardConfig()
    transport: RLTransportConfig = RLTransportConfig()
    policy_transfer: RLPolicyTransferConfig = RLPolicyTransferConfig()
    launcher: RLLauncherConfig = RLLauncherConfig()
    output_dir: Path = Path("outputs/unsloth_math_rl")
    max_steps: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_algorithm_config(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        return _normalize_algorithm_config(value)

    @model_validator(mode="after")
    def validate_rollout_modes(self) -> "RLConfig":
        if (
            self.inference.mode == "vllm_http"
            and "mode" in self.inference.model_fields_set
            and self.reward.mode == "passthrough"
            and self.orchestrator.custom_rollout_function is None
        ):
            raise ValueError(
                "reward.mode must score generated completions when inference.mode "
                "generates rollouts."
            )
        return self

    @model_validator(mode="after")
    def validate_policy_transfer(self) -> "RLConfig":
        if self.policy_transfer.type != "nccl":
            return self
        if self.lora is not None:
            raise ValueError(
                "policy_transfer.type='nccl' only supports full-model updates; "
                "use policy_transfer.type='filesystem' for LoRA adapters."
            )
        if (
            self.inference.mode != "vllm_http"
            or self.inference.vllm.server_backend != "openai"
        ):
            raise ValueError(
                "policy_transfer.type='nccl' requires inference.mode='vllm_http' "
                "and inference.vllm.server_backend='openai'."
            )
        return self

    @model_validator(mode="after")
    def resolve_rollouts_per_example(self) -> "RLConfig":
        rollouts_per_example = self.orchestrator.rollouts_per_example
        if rollouts_per_example is None or self.orchestrator.custom_rollout_function:
            return self
        if "num_generations" in self.inference.sampling.model_fields_set:
            return self
        self.inference = self.inference.model_copy(
            update={
                "sampling": self.inference.sampling.model_copy(
                    update={"num_generations": rollouts_per_example}
                )
            }
        )
        return self

    @model_validator(mode="after")
    def validate_sleep_colocation(self) -> "RLConfig":
        if self.launcher.mode != "colocate_sleep":
            return self
        if (
            self.orchestrator.max_async_level > 0
            or self.orchestrator.max_off_policy_steps > 0
        ):
            raise ValueError(
                "launcher.mode='colocate_sleep' requires synchronous rollouts; set "
                "orchestrator.max_async_level=0 and "
                "orchestrator.max_off_policy_steps=0."
            )
        return self

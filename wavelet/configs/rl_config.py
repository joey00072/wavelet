from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from wavelet.configs.sft import (
    ActivationOffloadingConfig,
    CheckpointConfig,
    FSDPConfig,
    LogConfig,
    LoRAConfig,
    LossMaskConfig,
    ModelConfig,
    MonitorConfig,
    OptimizerConfig,
    SchedulerConfig,
)


class RLDataConfig(BaseModel):
    source: Literal["local", "hf", "fake"] = "local"
    path: Path | list[Path] = Path("outputs/unsloth_math_data/rl_train.jsonl")
    hf_name: str | None = None
    hf_subsets: list[str] | None = None
    hf_splits: list[str] | None = None
    probabilities: list[float] | None = None
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted"
    batch_size: int = Field(default=4, ge=1)
    micro_batch_size: int = Field(default=1, ge=1)
    pack_sequences: bool = False
    pad_to_multiple_of: int = Field(default=1, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = True
    seq_len: int = Field(default=128, ge=8)
    shuffle: bool = True
    seed: int = 0
    max_examples: int | None = Field(default=None, ge=1)
    prompt_column: str = "prompt"
    completion_column: str = "completion"
    messages_column: str = "messages"
    system_prompt: str | None = None
    merge_messages_thinking: bool = False
    tools_column: str = "tools"
    chat_template_kwargs_column: str = "chat_template_kwargs"
    advantage_column: str = "advantage"
    reward_column: str = "reward"
    inference_logprobs_column: str = "inference_logprobs"
    teacher_logprobs_column: str = "teacher_logprobs"
    temperature_column: str = "temperature"
    metadata_column: str = "metadata"
    fake_vocab_size: int = Field(default=32000, ge=8)
    fake_length: Literal["fixed", "variable"] = "fixed"
    fake_input_ids: Literal["random", "increasing"] = "random"
    loss_mask: LossMaskConfig = LossMaskConfig()

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_columns(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if (
            "reference_logprobs_column" in value
            and "inference_logprobs_column" not in value
        ):
            value["inference_logprobs_column"] = value.pop("reference_logprobs_column")
        return value

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> "RLDataConfig":
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        if self.hf_subsets is not None and self.hf_splits is not None:
            if len(self.hf_subsets) != len(self.hf_splits):
                raise ValueError("hf_subsets and hf_splits must have the same length")
        if self.source == "hf" and self.hf_name is None:
            raise ValueError("hf_name is required when data.source='hf'")
        if self.source == "local" and self.hf_name is not None:
            raise ValueError("hf_name is only valid when data.source='hf'")
        return self


class RLLossConfig(BaseModel):
    type: Literal["dppo"] = "dppo"
    dppo_mask_high: float = Field(default=0.28, ge=0.0)
    dppo_mask_low: float = Field(default=0.20, ge=0.0)
    kl_tau: float = Field(default=0.1, ge=0.0)
    adv_tau: float = Field(default=1.0, ge=0.0)
    teacher_tau: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_loss_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if "advantage_scale" in value and "adv_tau" not in value:
            value["adv_tau"] = value.pop("advantage_scale")
        return value


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
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    max_completion_tokens: int = Field(default=128, ge=1)
    do_sample: bool = True
    num_generations: int = Field(default=1, ge=1)
    seed: int | None = None


class RLEvalSamplingConfig(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=-1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

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
    server_backend: Literal["offline", "openai"] = "offline"
    gpu_memory_utilization: float = Field(default=0.35, gt=0.0, le=1.0)
    max_model_len: int | None = Field(default=None, ge=8)
    tensor_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size_local: int | None = Field(default=None, ge=1)
    data_parallel_rpc_port: int | None = Field(default=None, ge=1, le=65535)
    enforce_eager: bool = True
    max_loras: int = Field(default=8, ge=1)
    max_cpu_loras: int = Field(default=100, ge=1)
    max_lora_rank: int | None = Field(default=None, ge=1)
    trust_remote_code: bool | None = None
    dtype: Literal["auto", "float32", "float16", "bfloat16"] | None = None
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
    mode: Literal["passthrough", "vllm", "vllm_http"] = "passthrough"
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
    materialize_path: Path | None = None
    overwrite: bool = True
    advantage_mode: Literal["passthrough", "reward", "group_reward"] = "passthrough"
    normalize_group_advantages: bool = False
    advantage_epsilon: float = Field(default=1e-6, gt=0.0)
    examples_per_step: int | None = Field(default=None, ge=1)
    rollouts_per_example: int | None = Field(default=None, ge=1)
    oversampling_factor: float = Field(default=1.0, ge=1.0)
    filter_zero_advantage: bool = False
    zero_advantage_max_retries: int = Field(default=8, ge=0)
    max_async_level: int = Field(default=0, ge=0)
    max_off_policy_steps: int = Field(default=0, ge=0)


class RLLauncherConfig(BaseModel):
    mode: Literal["integrated", "process"] = "integrated"
    backend: Literal["local", "ray"] = "local"
    trainer_cuda_visible_devices: str | list[str] | None = None
    inference_cuda_visible_devices: str | list[str] | None = None
    trainer_num_processes: int = Field(default=1, ge=1)
    inference_num_replicas: int = Field(default=1, ge=1)
    ray_address: str | None = None
    ray_runtime_env: dict[str, Any] | None = None
    poll_interval_seconds: float = Field(default=1.0, gt=0.0)


class RLConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    data: RLDataConfig = RLDataConfig()
    loss: RLLossConfig = RLLossConfig()
    optim: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    loss_impl: Literal["liger", "torch", "liger_fused"] = "torch"
    ckpt: CheckpointConfig | None = None
    lora: LoRAConfig | None = LoRAConfig()
    log: LogConfig = LogConfig()
    monitor: MonitorConfig = MonitorConfig()
    fsdp: FSDPConfig = FSDPConfig()
    orchestrator: RLOrchestratorConfig = RLOrchestratorConfig()
    eval: RLEvalConfig | None = None
    inference: RLInferenceConfig = RLInferenceConfig()
    reward: RLRewardConfig = RLRewardConfig()
    transport: RLTransportConfig = RLTransportConfig()
    policy_transfer: RLPolicyTransferConfig = RLPolicyTransferConfig()
    launcher: RLLauncherConfig = RLLauncherConfig()
    output_dir: Path = Path("outputs/unsloth_math_rl")
    clean_output_dir: bool = False
    dry_run: bool = False
    epochs: int = Field(default=1, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    seed: int = 0
    activation_offloading: ActivationOffloadingConfig | None = None

    @model_validator(mode="after")
    def resolve_checkpoint_output_dir(self) -> "RLConfig":
        if self.ckpt is not None and self.ckpt.output_dir is not None:
            self.output_dir = self.ckpt.output_dir
        return self

    @model_validator(mode="after")
    def validate_checkpoint_config(self) -> "RLConfig":
        if self.ckpt is None:
            return self
        if self.ckpt.resume_step is not None and self.ckpt.resume_step < -1:
            raise ValueError("ckpt.resume_step must be >= -1")
        if self.ckpt.mode != "disabled" and self.ckpt.interval is None:
            raise ValueError("ckpt.interval is required when checkpointing is enabled")
        return self

    @model_validator(mode="after")
    def validate_rollout_modes(self) -> "RLConfig":
        if (
            self.inference.mode in {"vllm", "vllm_http"}
            and self.reward.mode == "passthrough"
            and self.orchestrator.custom_rollout_function is None
        ):
            raise ValueError(
                "reward.mode must score generated completions when inference.mode "
                "generates rollouts."
            )
        return self

    @model_validator(mode="after")
    def resolve_rollouts_per_example(self) -> "RLConfig":
        rollouts_per_example = self.orchestrator.rollouts_per_example
        if rollouts_per_example is None or self.orchestrator.custom_rollout_function:
            return self
        self.inference = self.inference.model_copy(
            update={
                "sampling": self.inference.sampling.model_copy(
                    update={"num_generations": rollouts_per_example}
                )
            }
        )
        return self

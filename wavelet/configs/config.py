from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    """Base for every config model: unknown or misspelled keys are errors."""

    model_config = ConfigDict(extra="forbid")


class ActivationOffloadingConfig(ConfigModel):
    pin_memory: bool = True
    use_streams: bool = True
    max_fwd_stash_size: int = Field(default=5, ge=1)


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "experts",
    "fc1_latent_proj",
    "fc2_latent_proj",
]


class LoRAConfig(ConfigModel):
    rank: int = Field(default=16, ge=1)
    alpha: float = Field(default=32.0, ge=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )
    modules_to_save: list[str] = Field(default_factory=list)


class ModelConfig(ConfigModel):
    name: str = "Qwen/Qwen3-0.6B"
    adapter_path: Path | None = None
    chat_template: str | None = None
    trust_remote_code: bool = False
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "bfloat16"
    attn_implementation: Literal["auto", "flash_attention_2", "sdpa", "eager"] = "auto"
    load_in_4bit: bool = False
    kbit_cast_non_quantized_to_float32: bool = True
    gradient_checkpointing: bool = True
    meta_device_init: bool = False
    allow_tf32: bool = True
    fused_lora_mlp: bool = False  # patch MLP.forward with fused LoRA_MLP kernel
    fused_lora_qkv: bool = False  # patch Qwen3Attention forward with fused LoRA_QKV
    fused_lora_o: bool = False  # patch o_proj.forward with fused LoRA_W kernel
    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = "disabled"
    smart_gc: bool = False  # sqrt-N gradient checkpointing with CPU offload


class LossMaskConfig(ConfigModel):
    system: bool = False
    user: bool = False
    assistant: bool = True
    tool: bool = False


class TrainingDataConfig(ConfigModel):
    source: Literal["local", "hf", "fake"] = "local"
    path: Path | list[Path] = Path("outputs/data.jsonl")
    hf_name: str | None = None
    hf_subsets: list[str] | None = None
    hf_splits: list[str] | None = None
    probabilities: list[float] | None = None
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted"
    batch_size: int = Field(default=4, ge=1)
    micro_batch_size: int = Field(default=1, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = True
    seq_len: int = Field(default=128, ge=8)
    shuffle: bool = True
    seed: int = 0
    max_examples: int | None = Field(default=None, ge=1)
    fake_vocab_size: int = Field(default=32000, ge=8)
    fake_length: Literal["fixed", "variable"] = "fixed"
    fake_input_ids: Literal["random", "increasing"] = "random"
    prompt_column: str = "prompt"
    completion_column: str = "completion"
    messages_column: str = "messages"
    system_prompt: str | None = None
    merge_messages_thinking: bool = False
    tools_column: str = "tools"
    chat_template_kwargs_column: str = "chat_template_kwargs"
    loss_mask: LossMaskConfig = LossMaskConfig()

    @model_validator(mode="after")
    def validate_source(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        if (
            self.hf_subsets is not None
            and self.hf_splits is not None
            and len(self.hf_subsets) != len(self.hf_splits)
        ):
            raise ValueError("hf_subsets and hf_splits must have the same length")
        if self.source == "hf" and self.hf_name is None:
            raise ValueError("hf_name is required when data.source='hf'")
        if self.source == "local" and self.hf_name is not None:
            raise ValueError("hf_name is only valid when data.source='hf'")
        return self


class DataConfig(TrainingDataConfig):
    path: Path | list[Path] = Path("outputs/unsloth_math_data/sft_train.jsonl")
    pack_function: Literal["pad", "cat"] = "pad"


class SFTValConfig(ConfigModel):
    interval: int = Field(default=50, ge=1)
    eval_on_start: bool = False
    data: DataConfig


class OptimizerConfig(ConfigModel):
    type: Literal[
        "adamw",
        "adamw_8bit",
        "paged_adamw_8bit",
        "adam",
        "adam_8bit",
        "sgd",
    ] = "adamw"
    implementation: Literal["for-loop", "foreach", "fused"] = "fused"
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0)
    nesterov: bool = True
    betas1: float = Field(default=0.9, ge=0.0)
    betas2: float = Field(default=0.999, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_betas(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        raw_betas = normalized.pop("betas", None)
        if raw_betas is None:
            return normalized
        if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
            raise ValueError("optim.betas must be a 2-item list like [0.9, 0.999]")
        for name, beta in (("betas1", raw_betas[0]), ("betas2", raw_betas[1])):
            if name in normalized and normalized[name] != beta:
                raise ValueError(
                    f"optim.betas and optim.{name} disagree; set only one of them."
                )
            normalized[name] = beta
        return normalized


class SchedulerConfig(ConfigModel):
    type: Literal["constant", "linear", "cosine", "sqrt"] = "constant"
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    warmup_steps: int | None = Field(default=None, ge=0)
    decay_steps: int | None = Field(default=None, ge=1)
    min_lr: float = Field(default=0.0, ge=0.0)
    min_lr_factor: float | None = Field(default=None, ge=0.0, le=1.0)
    decay_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class CheckpointConfig(ConfigModel):
    interval: int | None = Field(default=None, ge=1)
    # Counts optimizer steps, not micro-steps.
    resume_step: int | None = None
    keep_last: int = Field(default=2, ge=1)
    mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    output_dir: Path | None = Field(default=None)


class FSDPConfig(ConfigModel):
    enabled: bool = False
    backend: Literal["auto", "gloo", "nccl", "hybrid"] = "auto"
    dp_replicate: int = Field(default=1, ge=1)
    dp_shard: int = -1
    cp: int = Field(default=1, ge=1)
    tp: int = Field(default=1, ge=1)
    ep: int = Field(default=1, ge=1)
    cpu_offload: bool = False

    @model_validator(mode="after")
    def validate_supported_parallel_dimensions(self) -> "FSDPConfig":
        unsupported = [
            f"fsdp.{name}={degree}"
            for name, degree in (("cp", self.cp), ("ep", self.ep))
            if degree > 1
        ]
        if unsupported:
            settings = ", ".join(unsupported)
            raise ValueError(
                f"{settings} is unsupported by the current model stack; context "
                "and expert parallel degrees must remain 1."
            )
        return self


class LogConfig(ConfigModel):
    level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    log_every: int = Field(default=1, ge=1)
    json_console: bool = False
    json_file: bool = True


class WandbConfig(ConfigModel):
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    mode: Literal["online", "offline", "disabled"] = "offline"
    init_timeout_seconds: float = Field(default=30.0, gt=0.0)
    offline_fallback: bool = True


class SampleLogConfig(ConfigModel):
    enabled: bool = False
    interval: int = Field(default=10, ge=1)
    sample_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    max_samples: int | None = Field(default=None, ge=1)
    keep_last: int = Field(default=256, ge=1)


class MonitorConfig(ConfigModel):
    enabled: bool = True
    write_events: bool = True
    write_metrics_jsonl: bool = True
    write_metrics_csv: bool = True
    write_run_metadata: bool = True
    write_heartbeat: bool = True
    log_cuda_memory: bool = True
    log_disk_usage: bool = True
    wandb: WandbConfig = WandbConfig()
    samples: SampleLogConfig = SampleLogConfig()


class TrainerConfig(ConfigModel):
    model: ModelConfig = ModelConfig()
    optim: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    loss_impl: Literal["liger", "torch", "liger_fused"] = "torch"
    ckpt: CheckpointConfig | None = None
    lora: LoRAConfig | None = LoRAConfig()
    log: LogConfig = LogConfig()
    monitor: MonitorConfig = MonitorConfig()
    fsdp: FSDPConfig = FSDPConfig()
    output_dir: Path = Path("outputs/train")
    clean_output_dir: bool = False
    dry_run: bool = False
    epochs: int = Field(default=1, ge=1)
    max_steps: int | None = Field(default=None, ge=0)
    seed: int = 0
    activation_offloading: ActivationOffloadingConfig | None = None

    @property
    def checkpoint_output_dir(self) -> Path:
        """Directory containing checkpoint step directories for this run."""
        if self.ckpt is not None and self.ckpt.output_dir is not None:
            return self.ckpt.output_dir
        return self.output_dir

    @model_validator(mode="after")
    def validate_fused_lora_kernels(self):
        fused = [
            name
            for name in ("fused_lora_mlp", "fused_lora_qkv", "fused_lora_o")
            if getattr(self.model, name)
        ]
        if not fused or self.lora is None:
            return self
        if self.lora.dropout > 0.0:
            raise ValueError(
                f"model.{fused[0]} does not apply lora.dropout; set lora.dropout=0 "
                "or disable the fused LoRA kernels."
            )
        return self

    @model_validator(mode="after")
    def validate_checkpoint_config(self):
        if self.ckpt is None:
            return self
        if self.ckpt.resume_step is not None and self.ckpt.resume_step < -1:
            raise ValueError("ckpt.resume_step must be >= -1")
        if self.ckpt.mode != "disabled" and self.ckpt.interval is None:
            raise ValueError("ckpt.interval is required when checkpointing is enabled")
        if self.ckpt.mode == "disabled":
            # Compare values, not model_fields_set: role configs are re-loaded
            # from a full dump where every field counts as explicitly set.
            explicit = [
                f"ckpt.{name}"
                for name, value in (
                    ("interval", self.ckpt.interval),
                    ("output_dir", self.ckpt.output_dir),
                )
                if value is not None
            ]
            if explicit:
                fields = ", ".join(explicit)
                raise ValueError(
                    f"{fields} set while ckpt.mode='disabled' would never write "
                    "checkpoints; set ckpt.mode to 'async' or "
                    "'async_with_pinned_mem', or remove these settings."
                )
        return self


class SFTConfig(TrainerConfig):
    data: DataConfig = DataConfig()
    val: SFTValConfig | None = None
    output_dir: Path = Path("outputs/unsloth_math_sft")
    max_steps: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_pack_function(self) -> "SFTConfig":
        if self.fsdp.cp > 1 and self.data.pack_function != "cat":
            raise ValueError("Packing function must be 'cat' when CP is enabled")
        if (
            self.fsdp.cp > 1
            and self.val is not None
            and self.val.data.pack_function != "cat"
        ):
            raise ValueError(
                "Validation packing function must be 'cat' when CP is enabled"
            )
        return self

    @model_validator(mode="after")
    def validate_cp_seq_len(self):
        if self.fsdp.cp > 1 and self.data.seq_len % self.fsdp.cp != 0:
            raise ValueError("Sequence length must be divisible by CP degree")
        if (
            self.fsdp.cp > 1
            and self.val is not None
            and self.val.data.seq_len % self.fsdp.cp != 0
        ):
            raise ValueError(
                "Validation sequence length must be divisible by CP degree"
            )
        return self

    @model_validator(mode="after")
    def validate_cp_micro_batch_size(self):
        if self.fsdp.cp > 1 and self.data.micro_batch_size != 1:
            raise ValueError("Micro batch size must be 1 when CP is enabled")
        if (
            self.fsdp.cp > 1
            and self.val is not None
            and self.val.data.micro_batch_size != 1
        ):
            raise ValueError("Validation micro batch size must be 1 when CP is enabled")
        return self


def _normalize_legacy_sampling_fields(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    if "max_tokens" in normalized:
        legacy = normalized.pop("max_tokens")
        current = normalized.get("max_completion_tokens", legacy)
        if current != legacy:
            raise ValueError(
                "sampling.max_tokens and sampling.max_completion_tokens disagree; "
                "set only max_completion_tokens."
            )
        normalized["max_completion_tokens"] = legacy
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

    @model_validator(mode="after")
    def validate_pad_to_multiple_of(self) -> "RLDataConfig":
        if self.seq_len % self.pad_to_multiple_of != 0:
            raise ValueError(
                "RL data.pad_to_multiple_of must divide data.seq_len; otherwise "
                "padded packed rows exceed the configured sequence length."
            )
        return self

    @model_validator(mode="after")
    def validate_num_workers(self) -> "RLDataConfig":
        if self.num_workers > 1:
            raise ValueError(
                "RL data.num_workers must be 0 or 1: rollout rows are pretokenized, "
                "and multiple dataloader workers split rows and packed bins "
                "differently from the per-rank micro-batch count the trainer "
                "expects."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_columns(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "reference_logprobs_column" in normalized:
            legacy = normalized.pop("reference_logprobs_column")
            current = normalized.get("inference_logprobs_column", legacy)
            if current != legacy:
                raise ValueError(
                    "data.reference_logprobs_column and data.inference_logprobs_column "
                    "disagree; set only inference_logprobs_column."
                )
            normalized["inference_logprobs_column"] = legacy
        return normalized


class RLLossConfig(ConfigModel):
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


class RLTransportConfig(ConfigModel):
    type: Literal["filesystem"] = "filesystem"
    rollout_filename: str = "rollouts.jsonl"
    queue_dir: Path | None = None
    poll_interval_seconds: float = Field(default=1.0, gt=0.0)
    idle_timeout_seconds: float | None = Field(default=None, gt=0.0)
    cleanup_consumed: bool = True
    keep_last_consumed: int = Field(default=2, ge=1)


class RLPolicyTransferConfig(ConfigModel):
    type: Literal["filesystem", "nccl"] = "filesystem"
    policy_dir: Path | None = None
    adapter_name: str = "policy"
    adapter_id: int = Field(default=1, ge=1)
    poll_interval_seconds: float = Field(default=1.0, gt=0.0)
    idle_timeout_seconds: float | None = Field(default=None, gt=0.0)
    export_initial: bool = True
    export_every_steps: int = Field(default=1, ge=1)
    # Policy snapshots are transport artifacts, not checkpoints. Keep the
    # active version and its predecessor so a lagging filesystem load cannot
    # lose its source while the next version is published.
    keep_last: int = Field(default=2, ge=2)
    lightweight_lora: bool = True
    nccl_host: str = "127.0.0.1"
    nccl_port: int = Field(default=29501, ge=1, le=65535)
    nccl_timeout_seconds: int = Field(default=600, ge=1)
    nccl_inference_world_size: int = Field(default=1, ge=1)
    nccl_rank_offset: int = Field(default=1, ge=1)


class RLSamplingConfig(ConfigModel):
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


class RLEvalSamplingConfig(ConfigModel):
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


class RLEvalEnvConfig(ConfigModel):
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


class RLEvalConfig(ConfigModel):
    env: list[RLEvalEnvConfig] = Field(default_factory=list)
    sampling: RLEvalSamplingConfig = RLEvalSamplingConfig()
    num_examples: int = -1
    rollouts_per_example: int = Field(default=1, ge=1)
    interval: int = Field(default=100, ge=1)
    max_retries: int = Field(default=0, ge=0)
    eval_base_model: bool = True
    final_eval: bool = True
    keep_last_rollout_sets: int = Field(default=2, ge=1)

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


class RLVLLMConfig(ConfigModel):
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


class RLVLLMHTTPConfig(ConfigModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    ports: list[int] | None = None
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    startup_timeout_seconds: float = Field(default=300.0, gt=0.0)

    @model_validator(mode="after")
    def validate_unique_ports(self) -> "RLVLLMHTTPConfig":
        if self.ports is None:
            return self
        if not self.ports:
            raise ValueError("inference.http.ports must list at least one port.")
        for port in self.ports:
            if not 1 <= port <= 65535:
                raise ValueError(f"inference.http.ports entry {port} is out of range.")
        if len(set(self.ports)) != len(self.ports):
            raise ValueError(
                "inference.http.ports must be unique; replicas cannot share a port."
            )
        return self


class RLInferenceConfig(ConfigModel):
    enabled: bool = True
    mode: Literal["passthrough", "vllm_http"] = "vllm_http"
    default_temperature: float = Field(default=1.0, gt=0.0)
    sampling: RLSamplingConfig = RLSamplingConfig()
    vllm: RLVLLMConfig = RLVLLMConfig()
    http: RLVLLMHTTPConfig = RLVLLMHTTPConfig()


class RLRewardConfig(ConfigModel):
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


class RLStateServerConfig(ConfigModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = (
        "warning"
    )
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_events: int = Field(default=2000, ge=100)


class _StrictConfig(ConfigModel):
    pass


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


class RLOrchestratorConfig(ConfigModel):
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
    state_server: RLStateServerConfig = RLStateServerConfig()

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


class RLLauncherConfig(ConfigModel):
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


_LEGACY_ADVANTAGE_KEYS = (
    "advantage_mode",
    "normalize_group_advantages",
    "advantage_epsilon",
    "length_penalty",
)


def _normalize_algorithm_config(value: dict[str, Any]) -> dict[str, Any]:
    """Infer file-backed algorithms and translate legacy advantage settings.

    Legacy ``orchestrator.advantage_mode`` settings are moved into ``algo`` so
    the resolved config carries them in exactly one place. Combining them with
    an explicit ``algo`` block, or passing GRPO-only settings without
    ``advantage_mode: group_reward``, is rejected instead of silently ignored.
    """
    normalized = dict(value)
    orchestrator = normalized.get("orchestrator")
    legacy: dict[str, Any] = {}
    if isinstance(orchestrator, dict):
        orchestrator = dict(orchestrator)
        legacy = {
            key: orchestrator.pop(key)
            for key in _LEGACY_ADVANTAGE_KEYS
            if key in orchestrator
        }
        normalized["orchestrator"] = orchestrator
    legacy_fields = ", ".join(f"orchestrator.{key}" for key in legacy)

    if "algo" in normalized:
        if legacy:
            raise ValueError(
                f"{legacy_fields} cannot be combined with an explicit algo block; "
                "move the advantage settings into algo."
            )
        algo = normalized["algo"]
        if (
            isinstance(algo, dict)
            and "type" not in algo
            and ("file" in algo or "algorithm" in algo)
        ):
            normalized["algo"] = {"type": "custom", **algo}
        return normalized

    if not legacy:
        return normalized
    mode = legacy.get("advantage_mode")
    grpo_fields = ", ".join(
        f"orchestrator.{key}" for key in legacy if key != "advantage_mode"
    )
    if mode != "group_reward" and grpo_fields:
        raise ValueError(
            f"{grpo_fields} only apply with orchestrator.advantage_mode="
            "'group_reward'; use algo.type='grpo' with normalize_advantages, "
            "epsilon, and length_penalty instead."
        )
    if mode != "group_reward":
        normalized["algo"] = {"type": mode}
        return normalized

    normalized["algo"] = {
        "type": "grpo",
        "normalize_advantages": legacy.get("normalize_group_advantages", False),
        "epsilon": legacy.get("advantage_epsilon", 1e-6),
        "length_penalty": _normalize_length_penalty(legacy.get("length_penalty")),
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
    def validate_train_sampling_replay(self) -> "RLConfig":
        if (
            not self.orchestrator.enabled
            or not self.inference.enabled
            or self.inference.mode != "vllm_http"
        ):
            return self

        sampling = self.inference.sampling
        unsupported: list[str] = []
        if sampling.top_p < 1.0:
            unsupported.append("top_p")
        if sampling.top_k > 0:
            unsupported.append("top_k")
        if sampling.min_p > 0.0:
            unsupported.append("min_p")
        if sampling.min_tokens > 0:
            unsupported.append("min_tokens")
        if sampling.repetition_penalty != 1.0:
            unsupported.append("repetition_penalty")
        distribution_overrides = {
            "frequency_penalty",
            "logit_bias",
            "min_p",
            "min_tokens",
            "presence_penalty",
            "repetition_penalty",
            "seed",
            "temperature",
            "top_k",
            "top_p",
        }
        unsupported.extend(
            f"extra_body.{field}"
            for field in sorted(distribution_overrides & sampling.extra_body.keys())
        )
        if unsupported:
            fields = ", ".join(unsupported)
            raise ValueError(
                "RL train sampling changes the token distribution with "
                f"{fields}, but Wavelet does not yet replay those transforms in "
                "the trainer. Use top_p=1, top_k=-1, min_p=0, min_tokens=0, "
                "repetition_penalty=1, and keep distribution controls out of "
                "extra_body for correct importance ratios."
            )
        return self

    @model_validator(mode="after")
    def validate_group_sampling_diversity(self) -> "RLConfig":
        if (
            not self.orchestrator.enabled
            or not self.inference.enabled
            or self.max_steps == 0
            or not isinstance(self.algo, (GRPOAlgorithmConfig, MaxRLAlgorithmConfig))
            or (self.orchestrator.rollouts_per_example or 1) <= 1
        ):
            return self

        sampling = self.inference.sampling
        deterministic: list[str] = []
        if not sampling.do_sample:
            deterministic.append("do_sample=false")
        if sampling.temperature <= 0:
            deterministic.append("temperature=0")
        if sampling.seed is not None:
            deterministic.append("a fixed sampling seed")
        if deterministic:
            settings = ", ".join(deterministic)
            raise ValueError(
                "Group-relative RL needs diverse rollouts, but the configured "
                f"sampling can repeat each completion ({settings}). Use "
                "do_sample=true, temperature>0, and sampling.seed=null; data.seed "
                "still provides deterministic task ordering."
            )
        return self

    @model_validator(mode="after")
    def validate_online_sampling_temperature(self) -> "RLConfig":
        if (
            not self.orchestrator.enabled
            or not self.inference.enabled
            or self.max_steps == 0
        ):
            return self

        sampling = self.inference.sampling
        if sampling.do_sample and sampling.temperature > 0:
            return self
        raise ValueError(
            "Online RL requires stochastic sampling with a positive temperature "
            "so behavior log-probabilities can be replayed by the trainer. Use "
            "do_sample=true and temperature>0, or set max_steps=0 for evaluation."
        )

    @model_validator(mode="after")
    def validate_export_interval_within_freshness_window(self) -> "RLConfig":
        # Integrated runs pick the newest export at or below the trainer step in
        # process, so only the process-style schedulers can wait on an export.
        if (
            not self.orchestrator.enabled
            or not self.inference.enabled
            or self.launcher.mode == "integrated"
            or self.max_steps == 0
        ):
            return self
        async_lag = max(self.orchestrator.max_async_level - 1, 0)
        allowed_lag = min(async_lag, self.orchestrator.max_off_policy_steps)
        interval = self.policy_transfer.export_every_steps
        if interval > allowed_lag + 1:
            raise ValueError(
                f"policy_transfer.export_every_steps={interval} leaves rollout "
                f"steps without an admissible policy: with max_async_level="
                f"{self.orchestrator.max_async_level} and max_off_policy_steps="
                f"{self.orchestrator.max_off_policy_steps} each rollout step may "
                f"use policies from the last {allowed_lag + 1} step(s), so the "
                "orchestrator would wait for an export the trainer cannot produce "
                "until it receives those rollouts. Set export_every_steps to at "
                f"most {allowed_lag + 1} or widen the off-policy window."
            )
        return self

    @model_validator(mode="after")
    def validate_initial_policy_for_process_training(self) -> "RLConfig":
        if (
            self.orchestrator.enabled
            and self.launcher.mode != "integrated"
            and (self.max_steps is None or self.max_steps > 0)
            and not self.policy_transfer.export_initial
        ):
            raise ValueError(
                "Process and colocated RL training require "
                "policy_transfer.export_initial=true so rollout step 0 can load "
                "the trainer's initial policy."
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

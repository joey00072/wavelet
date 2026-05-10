from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ActivationOffloadingConfig(BaseModel):
    pin_memory: bool = True
    use_streams: bool = True
    max_fwd_stash_size: int = Field(default=5, ge=1)


class LoRAConfig(BaseModel):
    rank: int = Field(default=16, ge=1)
    alpha: float = Field(default=32.0, ge=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: [
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
    )
    modules_to_save: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen3-0.6B"
    adapter_path: Path | None = None
    chat_template: str | None = None
    trust_remote_code: bool = False
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "bfloat16"
    attn_implementation: Literal["auto", "flash_attention_2", "sdpa", "eager"] = "auto"
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    meta_device_init: bool = False
    allow_tf32: bool = True
    fused_lora_mlp: bool = False  # patch MLP.forward with fused LoRA_MLP kernel
    fused_lora_qkv: bool = False  # patch Qwen3Attention forward with fused LoRA_QKV
    fused_lora_o: bool = False  # patch o_proj.forward with fused LoRA_W kernel
    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = "disabled"
    smart_gc: bool = False  # sqrt-N gradient checkpointing with CPU offload


class LossMaskConfig(BaseModel):
    system: bool = False
    user: bool = False
    assistant: bool = True
    tool: bool = False


class DataConfig(BaseModel):
    source: Literal["local", "hf", "fake"] = "local"
    path: Path | list[Path] = Path("outputs/unsloth_math_data/sft_train.jsonl")
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
    pack_function: Literal["pad", "cat", "stack"] = "pad"
    fake_vocab_size: int = Field(default=32000, ge=8)
    fake_length: Literal["fixed", "variable"] = "fixed"
    fake_input_ids: Literal["random", "increasing"] = "random"
    stack_bucket_multiple: int = Field(default=256, ge=1)
    stack_bucket_timeout: int = Field(default=10, ge=1)
    prompt_column: str = "prompt"
    completion_column: str = "completion"
    messages_column: str = "messages"
    system_prompt: str | None = None
    merge_messages_thinking: bool = False
    tools_column: str = "tools"
    chat_template_kwargs_column: str = "chat_template_kwargs"
    loss_mask: LossMaskConfig = LossMaskConfig()

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> "DataConfig":
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        if self.hf_subsets is not None and self.hf_splits is not None:
            if len(self.hf_subsets) != len(self.hf_splits):
                raise ValueError("hf_subsets and hf_splits must have the same length")
        if self.source == "hf" and self.hf_name is None:
            raise ValueError("hf_name is required when data.source='hf'")
        if self.source == "local" and self.hf_name is not None:
            raise ValueError("hf_name is only valid when data.source='hf'")
        if self.pack_function == "stack":
            max_area = self.seq_len * self.micro_batch_size
            if max_area % self.stack_bucket_multiple != 0:
                raise ValueError(
                    "seq_len * micro_batch_size must be divisible by "
                    "stack_bucket_multiple for stack packing"
                )
        return self


class SFTValConfig(BaseModel):
    interval: int = Field(default=50, ge=1)
    eval_on_start: bool = False
    data: DataConfig


class OptimizerConfig(BaseModel):
    type: Literal[
        "adamw",
        "adamw_8bit",
        "paged_adamw_8bit",
        "adam",
        "adam_8bit",
        "sgd",
        "muon",
    ] = "adamw"
    implementation: Literal["for-loop", "foreach", "fused"] = "fused"
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0)
    nesterov: bool = True
    betas1: float = Field(default=0.9, ge=0.0)
    betas2: float = Field(default=0.999, ge=0.0)
    mu: float = Field(default=0.95, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_betas(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        raw_betas = value.pop("betas", None)
        if raw_betas is None:
            return value
        if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
            raise ValueError("optim.betas must be a 2-item list like [0.9, 0.999]")
        value.setdefault("betas1", raw_betas[0])
        value.setdefault("betas2", raw_betas[1])
        return value


class SchedulerConfig(BaseModel):
    type: Literal["constant", "linear", "cosine", "sqrt"] = "constant"
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    warmup_steps: int | None = Field(default=None, ge=0)
    decay_steps: int | None = Field(default=None, ge=0)
    min_lr: float = Field(default=0.0, ge=0.0)
    min_lr_factor: float | None = Field(default=None, ge=0.0, le=1.0)
    decay_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class CheckpointConfig(BaseModel):
    interval: int | None = Field(default=None, ge=1)
    # Counts optimizer steps, not micro-steps.
    resume_step: int | None = None
    keep_last: int | None = Field(default=None, ge=1)
    mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    output_dir: Path | None = Field(default=None)


class SingleNodeDeploymentConfig(BaseModel):
    type: Literal["single_node"] = "single_node"
    num_gpus: int = Field(default=1, ge=1)


class FSDPConfig(BaseModel):
    enabled: bool = False
    backend: Literal["auto", "gloo", "nccl", "hybrid"] = "auto"
    dp_replicate: int = Field(default=1, ge=1)
    dp_shard: int = -1
    cp: int = Field(default=1, ge=1)
    tp: int = Field(default=1, ge=1)
    ep: int = Field(default=1, ge=1)
    cpu_offload: bool = False
    reshard_after_forward: bool = True


class LogConfig(BaseModel):
    level: str = "info"
    log_every: int = Field(default=1, ge=1)
    json_console: bool = False
    json_file: bool = True


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    mode: Literal["online", "offline", "disabled"] = "offline"
    init_timeout_seconds: float = Field(default=30.0, gt=0.0)
    offline_fallback: bool = True


class SampleLogConfig(BaseModel):
    enabled: bool = False
    interval: int = Field(default=10, ge=1)
    sample_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    max_samples: int | None = Field(default=None, ge=1)


class MonitorConfig(BaseModel):
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


class GenerateConfig(BaseModel):
    prompt: str = "wavelet"
    max_new_tokens: int = Field(default=24, ge=1)


class SFTConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    val: SFTValConfig | None = None
    optim: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    loss_impl: Literal["liger", "torch", "liger_fused"] = "torch"
    ckpt: CheckpointConfig | None = None
    lora: LoRAConfig | None = LoRAConfig()
    log: LogConfig = LogConfig()
    monitor: MonitorConfig = MonitorConfig()
    generate: GenerateConfig = GenerateConfig()
    deployment: SingleNodeDeploymentConfig = SingleNodeDeploymentConfig()
    fsdp: FSDPConfig = FSDPConfig()
    output_dir: Path = Path("outputs/unsloth_math_sft")
    clean_output_dir: bool = False
    dry_run: bool = False
    epochs: int = Field(default=1, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    seed: int = 0
    activation_offloading: ActivationOffloadingConfig | None = None

    def validate_pack_function(self):
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

    @model_validator(mode="after")
    def resolve_checkpoint_output_dir(self):
        if self.ckpt is not None and self.ckpt.output_dir is not None:
            self.output_dir = self.ckpt.output_dir
        return self

    @model_validator(mode="after")
    def validate_checkpoint_config(self):
        if self.ckpt is None:
            return self
        if self.ckpt.resume_step is not None and self.ckpt.resume_step < -1:
            raise ValueError("ckpt.resume_step must be >= -1")
        if self.ckpt.mode != "disabled" and self.ckpt.interval is None:
            raise ValueError("ckpt.interval is required when checkpointing is enabled")
        return self

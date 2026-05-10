from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed
from peft import PeftModel, prepare_model_for_kbit_training
from torch import nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils import logging as transformers_logging

from wavelet.configs.sft import FSDPConfig, ModelConfig
from wavelet.distributed.parallel_dims import ParallelDims
from wavelet.distributed.world import World
from wavelet.trainer.debug import (
    DEBUG_MODEL_NAME,
    build_debug_model,
    build_debug_tokenizer,
)


def resolve_dtype(name: str) -> torch.dtype | str:
    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def setup_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    if config.name == DEBUG_MODEL_NAME:
        return build_debug_tokenizer(model_max_length=4096)
    tokenizer_source = config.adapter_path or config.name
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=config.trust_remote_code,
        )
    except AttributeError as exc:
        invalid_adapter_tokenizer = "'list' object has no attribute 'keys'" in str(exc)
        if config.adapter_path is None or not invalid_adapter_tokenizer:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.chat_template is not None:
        tokenizer.chat_template = config.chat_template
    tokenizer.padding_side = "left"
    return tokenizer


def apply_liger_kernel(loss_impl: str, model_name: str) -> None:
    """Patch the model class with Liger fused kernels before from_pretrained.

    Must be called before AutoModelForCausalLM.from_pretrained so the patched
    class methods are in place when model weights are loaded.

    loss_impl="liger": rope + rms_norm + swiglu patches only (speed wins, no CE change)
    loss_impl="liger_fused": additionally patches fused_linear_cross_entropy so the
        model can compute loss internally without materializing the full (B*L, V) logit
        tensor — eliminates the 1.16 GB lm_head gradient on large-vocab models.
    """
    if loss_impl not in ("liger", "liger_fused"):
        return
    try:
        from liger_kernel.transformers.monkey_patch import (
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_mistral,
            apply_liger_kernel_to_qwen2,
            apply_liger_kernel_to_qwen3,
        )

        name_lower = model_name.lower()
        fused_ce = loss_impl == "liger_fused"
        if "qwen3" in name_lower or "qwen-3" in name_lower:
            apply_liger_kernel_to_qwen3(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
                fused_linear_cross_entropy=fused_ce,
            )
        elif "qwen2" in name_lower or "qwen-2" in name_lower:
            apply_liger_kernel_to_qwen2(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
                fused_linear_cross_entropy=fused_ce,
            )
        elif "llama" in name_lower:
            apply_liger_kernel_to_llama(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
                fused_linear_cross_entropy=fused_ce,
            )
        elif "mistral" in name_lower:
            apply_liger_kernel_to_mistral(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
                fused_linear_cross_entropy=fused_ce,
            )
        else:
            raise ValueError(
                f"Liger kernel not yet mapped for model '{model_name}'. "
                "Add it to apply_liger_kernel() or use loss_impl='torch'."
            )
    except ImportError as exc:
        raise ImportError(
            "Liger kernel support requires the installed 'liger-kernel' package, "
            "but importing it failed."
        ) from exc


def _best_attn_implementation() -> str:
    """Pick the fastest available attention implementation.

    Priority: flash_attention_2 → eager (xformers memory-efficient SDPA) → sdpa.
    xformers is registered as a SDPA backend; passing 'eager' lets the model use
    whatever SDPA backend PyTorch selects (which picks xformers when available).
    flash_attention_2 requires the flash-attn package and SM >= 8.0.
    """
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        pass
    try:
        import xformers  # noqa: F401

        return "eager"
    except ImportError:
        pass
    return "sdpa"


def setup_runtime(config: ModelConfig) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    if config.allow_tf32:
        torch.set_float32_matmul_precision("high")


def setup_model(
    config: ModelConfig,
    *,
    max_seq_length: int | None = None,
    distributed: bool = False,
    parallel_dims: ParallelDims | None = None,
) -> PreTrainedModel:
    if config.name == DEBUG_MODEL_NAME:
        if parallel_dims is not None and parallel_dims.tp_enabled:
            raise NotImplementedError(
                "Tensor parallel execution is not implemented for the debug model."
            )
        model = build_debug_model(max_seq_length=max_seq_length)
        model.config.use_cache = False
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if config.adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                config.adapter_path,
                is_trainable=True,
            )
        return cast(PreTrainedModel, model)

    quantization_config = None
    model_is_prequantized = False
    setup_runtime(config)
    attn_implementation = (
        _best_attn_implementation()
        if config.attn_implementation == "auto"
        else config.attn_implementation
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "dtype": resolve_dtype(config.torch_dtype),
        "attn_implementation": attn_implementation,
    }
    if parallel_dims is not None and parallel_dims.tp_enabled:
        if config.load_in_4bit:
            raise NotImplementedError(
                "QLoRA with trainer tensor parallelism is not supported. Use "
                "replicated DDP training for 4-bit LoRA adapters."
            )
        if config.adapter_path is not None:
            raise NotImplementedError(
                "Loading PEFT adapters onto a tensor-parallel model is not "
                "implemented yet."
            )
        model_kwargs["tp_plan"] = "auto"
        model_kwargs["device_mesh"] = parallel_dims.get_mesh("tp")
    if config.meta_device_init:
        model_kwargs["low_cpu_mem_usage"] = True
    if config.load_in_4bit:
        model_config = AutoConfig.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
        )
        model_is_prequantized = (
            getattr(model_config, "quantization_config", None) is not None
        )
        if not model_is_prequantized:
            compute_dtype = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = _quantized_device_map(distributed)

    previous_transformers_verbosity = transformers_logging.get_verbosity()
    suppress_float32_flash_warning = (
        config.torch_dtype == "float32"
        and attn_implementation in {"flash_attention_2", "flash_attention_3"}
    )
    if suppress_float32_flash_warning:
        transformers_logging.set_verbosity_error()
    try:
        model = AutoModelForCausalLM.from_pretrained(config.name, **model_kwargs)
    finally:
        if suppress_float32_flash_warning:
            transformers_logging.set_verbosity(previous_transformers_verbosity)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if config.load_in_4bit and not model_is_prequantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        # Embedding and lm_head are NOT 4-bit quantized (full precision), but
        # flash attention expects bfloat16 inputs. Cast them so hidden states
        # start in bfloat16 and the "Casting fp32 inputs" warning disappears.
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        for module_name in ("embed_tokens", "lm_head", "norm"):
            for name, module in model.named_modules():
                if name.endswith(module_name):
                    module.to(compute_dtype)
    if config.fused_lm_head_token_chunk_size != "disabled":
        from wavelet.trainer.lm_head import maybe_inject_chunked_lm_head

        maybe_inject_chunked_lm_head(model, config.fused_lm_head_token_chunk_size)
    if config.adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            config.adapter_path,
            is_trainable=True,
        )
    return cast(PreTrainedModel, model)


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    is_main_process: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, PeftModel):
        if getattr(unwrap_model(model), "_tp_size", None) is not None:
            raise NotImplementedError(
                "Saving LoRA adapters from a tensor-parallel model is not "
                "implemented yet."
            )
        target = output_dir / "adapter"
        if state_dict is None:
            model.save_pretrained(target, is_main_process=is_main_process)
        else:
            model.save_pretrained(
                target,
                state_dict=state_dict,
                is_main_process=is_main_process,
            )
        if is_main_process:
            tokenizer.save_pretrained(target)
        return target

    target = output_dir / "model"
    if state_dict is None:
        model.save_pretrained(target, is_main_process=is_main_process)
    else:
        model.save_pretrained(
            target,
            state_dict=state_dict,
            is_main_process=is_main_process,
        )
    if is_main_process:
        tokenizer.save_pretrained(target)
    return target


def maybe_wrap_fsdp(
    model: PreTrainedModel,
    *,
    model_config: ModelConfig,
    fsdp_config: FSDPConfig,
    world: World,
    parallel_dims: ParallelDims | None = None,
) -> PreTrainedModel:
    if not fsdp_config.enabled:
        return model

    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "FSDP requires an initialized torch.distributed process group."
        )

    if parallel_dims is not None:
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Expert parallel execution is not wired into the model stack yet."
            )
        if parallel_dims.cp_enabled:
            raise NotImplementedError(
                "Context parallel execution is not wired into the attention stack yet."
            )

    if world.device.type == "cuda" and not (
        parallel_dims is not None and parallel_dims.tp_enabled
    ):
        model = cast(PreTrainedModel, model.to(world.device))

    auto_wrap_policy = None
    layer_classes = _transformer_layer_classes(model)
    if layer_classes:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=layer_classes,
        )

    mixed_precision = _fsdp_mixed_precision(model_config)
    sharding_strategy = _fsdp_sharding_strategy(parallel_dims)
    device_mesh = None
    if (
        parallel_dims is not None
        and parallel_dims.dp_enabled
        and torch.distributed.get_world_size() > 1
    ):
        device_mesh = parallel_dims.get_mesh("hsdp")

    wrapped = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_id=world.device if world.device.type == "cuda" else torch.device("cpu"),
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy,
        use_orig_params=True,
        device_mesh=device_mesh,
    )
    return cast(PreTrainedModel, wrapped)


def maybe_wrap_ddp(
    model: PreTrainedModel,
    *,
    model_config: ModelConfig,
    world: World,
    parallel_dims: ParallelDims | None = None,
) -> PreTrainedModel:
    if world.world_size <= 1:
        return model
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed data parallel requires an initialized process group."
        )
    if parallel_dims is not None and (
        parallel_dims.tp_enabled or parallel_dims.cp_enabled or parallel_dims.ep_enabled
    ):
        raise NotImplementedError(
            "DDP is only wired for pure data parallel training. Use FSDP for "
            "sharded/hybrid layouts."
        )

    if world.device.type == "cuda":
        if not model_config.load_in_4bit:
            model = cast(PreTrainedModel, model.to(world.device))
        return cast(
            PreTrainedModel,
            DDP(model, device_ids=[world.local_rank], output_device=world.local_rank),
        )
    if not model_config.load_in_4bit:
        model = cast(PreTrainedModel, model.to(world.device))
    return cast(PreTrainedModel, DDP(model))


def _quantized_device_map(distributed: bool) -> str | dict[str, int]:
    if not distributed:
        return "auto"
    if not torch.cuda.is_available():
        return "auto"
    return {"": torch.cuda.current_device()}


def export_model_for_save(
    model: PreTrainedModel,
) -> tuple[PreTrainedModel, dict[str, torch.Tensor] | None]:
    if not isinstance(model, FSDP):
        return unwrap_model(model), None

    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
        state_dict = model.state_dict()
    return unwrap_model(model), state_dict


def unwrap_model(model: nn.Module) -> PreTrainedModel:
    current = model
    while hasattr(current, "module"):
        current = cast(nn.Module, getattr(current, "module"))
    return cast(PreTrainedModel, current)


def _transformer_layer_classes(model: nn.Module) -> set[type[nn.Module]]:
    no_split_modules = getattr(model, "_no_split_modules", None)
    if not no_split_modules:
        return set()

    class_names = set(no_split_modules)
    layer_classes: set[type[nn.Module]] = set()
    for module in model.modules():
        if module.__class__.__name__ in class_names:
            layer_classes.add(type(module))
    return layer_classes


def _fsdp_mixed_precision(model_config: ModelConfig) -> MixedPrecision | None:
    dtype = resolve_dtype(model_config.torch_dtype)
    if not isinstance(dtype, torch.dtype):
        return None
    if not torch.cuda.is_available() and dtype is not torch.float32:
        return None
    if dtype is torch.float32 and torch.cuda.is_available():
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def _fsdp_sharding_strategy(
    parallel_dims: ParallelDims | None,
) -> ShardingStrategy:
    if parallel_dims is None:
        if torch.distributed.get_world_size() > 1:
            return ShardingStrategy.FULL_SHARD
        return ShardingStrategy.NO_SHARD
    if parallel_dims.dp_replicate_enabled and parallel_dims.dp_shard_enabled:
        return ShardingStrategy.HYBRID_SHARD
    if parallel_dims.dp_shard_enabled:
        return ShardingStrategy.FULL_SHARD
    return ShardingStrategy.NO_SHARD

"""Model loading, quantization, wrapping, and LoRA support."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed
from peft import LoraConfig as PeftLoraConfig
from peft import (
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from safetensors.torch import save_file as save_safetensors
from torch import nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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

from wavelet.configs.config import DEFAULT_LORA_TARGET_MODULES
from wavelet.configs.sft import FSDPConfig, LoRAConfig, ModelConfig
from wavelet.trainer.debug import (
    DEBUG_LORA_TARGET_MODULES,
    DEBUG_MODEL_NAME,
    build_debug_model,
    build_debug_tokenizer,
)
from wavelet.trainer.distributed import ParallelDims, World

logger = logging.getLogger(__name__)


def resolve_dtype(name: str) -> torch.dtype | str:
    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def prepare_kbit_model(
    model: PreTrainedModel,
    *,
    gradient_checkpointing: bool,
    cast_non_quantized_to_float32: bool,
) -> PreTrainedModel:
    if cast_non_quantized_to_float32:
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    for param in model.parameters():
        param.requires_grad = False
    if gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):

            def make_inputs_require_grad(
                module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                del module, inputs
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


def setup_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    if config.name == DEBUG_MODEL_NAME:
        return build_debug_tokenizer(model_max_length=4096)
    tokenizer_source = config.adapter_path or config.name
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=config.trust_remote_code,
        )
    except (AttributeError, OSError, ValueError):
        if config.adapter_path is None:
            raise
        logger.warning(
            "Could not load tokenizer artifacts from adapter path %s; "
            "falling back to base model %s.",
            config.adapter_path,
            config.name,
        )
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
    """Use the best installed attention backend for the current GPU."""
    if _is_hopper_gpu() and _flash_attention_3_available():
        return "flash_attention_3"
    return "flash_attention_2" if _flash_attention_available() else "sdpa"


def _is_hopper_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _flash_attention_3_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_3_available
    except ImportError:
        return False
    return bool(is_flash_attn_3_available())


def _flash_attention_available() -> bool:
    """Return whether the FlashAttention 2 extension imports successfully."""
    try:
        import flash_attn  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def _require_flash_attention() -> None:
    if _flash_attention_available():
        return
    raise ImportError(
        "model.attn_implementation='flash_attention_2' requires a working "
        "flash-attn installation. Install it with `uv sync --extra flash-attn`."
    )


def _require_flash_attention_3() -> None:
    if _is_hopper_gpu() and _flash_attention_3_available():
        return
    raise ImportError(
        "model.attn_implementation='flash_attention_3' requires a Hopper GPU "
        "and the flash-attn-3 package."
    )


def setup_runtime(config: ModelConfig) -> None:
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision(config.matmul_precision)


def _setup_debug_model(
    config: ModelConfig,
    *,
    max_seq_length: int | None,
    parallel_dims: ParallelDims | None,
) -> PreTrainedModel:
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


def _model_load_kwargs(
    config: ModelConfig,
    *,
    distributed: bool,
    parallel_dims: ParallelDims | None,
) -> tuple[dict[str, Any], bool, str]:
    attention = (
        _best_attn_implementation()
        if config.attn_implementation == "auto"
        else config.attn_implementation
    )
    if attention == "flash_attention_2":
        _require_flash_attention()
    elif attention == "flash_attention_3":
        _require_flash_attention_3()
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "dtype": resolve_dtype(config.torch_dtype),
        "attn_implementation": attention,
    }
    _add_tensor_parallel_load_kwargs(kwargs, config, parallel_dims)
    if config.meta_device_init:
        kwargs["low_cpu_mem_usage"] = True
    prequantized = _add_quantized_load_kwargs(
        kwargs,
        config,
        distributed=distributed,
    )
    return kwargs, prequantized, attention


def _add_tensor_parallel_load_kwargs(
    kwargs: dict[str, Any],
    config: ModelConfig,
    parallel_dims: ParallelDims | None,
) -> None:
    if parallel_dims is None or not parallel_dims.tp_enabled:
        return
    if config.load_in_4bit:
        raise NotImplementedError(
            "QLoRA with trainer tensor parallelism is not supported. Use "
            "replicated DDP training for 4-bit LoRA adapters."
        )
    if config.adapter_path is not None:
        raise NotImplementedError(
            "Loading PEFT adapters onto a tensor-parallel model is not implemented yet."
        )
    kwargs["tp_plan"] = "auto"
    kwargs["device_mesh"] = parallel_dims.get_mesh("tp")


def _add_quantized_load_kwargs(
    kwargs: dict[str, Any],
    config: ModelConfig,
    *,
    distributed: bool,
) -> bool:
    if not config.load_in_4bit:
        return False
    model_config = AutoConfig.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code,
    )
    prequantized = getattr(model_config, "quantization_config", None) is not None
    if not prequantized:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    kwargs["device_map"] = _quantized_device_map(distributed)
    return prequantized


def _load_pretrained_model(
    config: ModelConfig,
    kwargs: dict[str, Any],
    *,
    attention: str,
) -> PreTrainedModel:
    previous_verbosity = transformers_logging.get_verbosity()
    suppress_warning = config.torch_dtype == "float32" and attention in {
        "flash_attention_2",
        "flash_attention_3",
    }
    if suppress_warning:
        transformers_logging.set_verbosity_error()
    try:
        return AutoModelForCausalLM.from_pretrained(config.name, **kwargs)
    finally:
        if suppress_warning:
            transformers_logging.set_verbosity(previous_verbosity)


def _prepare_quantized_modules(model: PreTrainedModel, config: ModelConfig) -> None:
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    for module_name in ("embed_tokens", "lm_head", "norm"):
        for name, module in model.named_modules():
            if name.endswith(module_name):
                module.to(compute_dtype)


def setup_model(
    config: ModelConfig,
    *,
    max_seq_length: int | None = None,
    distributed: bool = False,
    parallel_dims: ParallelDims | None = None,
) -> PreTrainedModel:
    if config.name == DEBUG_MODEL_NAME:
        return _setup_debug_model(
            config,
            max_seq_length=max_seq_length,
            parallel_dims=parallel_dims,
        )

    setup_runtime(config)
    model_kwargs, model_is_prequantized, attention = _model_load_kwargs(
        config,
        distributed=distributed,
        parallel_dims=parallel_dims,
    )
    model = _load_pretrained_model(config, model_kwargs, attention=attention)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if config.load_in_4bit and not model_is_prequantized:
        model = prepare_kbit_model(
            model,
            gradient_checkpointing=config.gradient_checkpointing,
            cast_non_quantized_to_float32=(config.kbit_cast_non_quantized_to_float32),
        )
        # Embedding and lm_head are NOT 4-bit quantized (full precision), but
        # flash attention expects bfloat16 inputs. Cast them so hidden states
        # start in bfloat16 and the "Casting fp32 inputs" warning disappears.
        _prepare_quantized_modules(model, config)
    if config.fused_lm_head_token_chunk_size != "disabled":
        from wavelet.trainer.losses import maybe_inject_chunked_lm_head

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


def _state_dict_to_save_dtype(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        key: (
            value.detach().to(device="cpu", dtype=dtype)
            if value.is_floating_point()
            else value.detach().to(device="cpu")
        )
        for key, value in state_dict.items()
    }


def export_model_for_save(
    model: PreTrainedModel,
    *,
    state_dict_dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, dict[str, torch.Tensor] | None]:
    if not isinstance(model, FSDP):
        unwrapped = unwrap_model(model)
        if state_dict_dtype is None:
            return unwrapped, None
        return unwrapped, _state_dict_to_save_dtype(
            unwrapped.state_dict(),
            state_dict_dtype,
        )

    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
        state_dict = model.state_dict()
    if state_dict_dtype is not None:
        state_dict = _state_dict_to_save_dtype(state_dict, state_dict_dtype)
    return unwrap_model(model), state_dict


def unwrap_model(model: nn.Module) -> PreTrainedModel:
    current = model
    while hasattr(current, "module"):
        current = cast(nn.Module, current.module)
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


LORA_STATE_ATTRS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


TP_REPLICATED_LORA_ATTRS = {
    "colwise": ("lora_A", "lora_embedding_A"),
    "rowwise": ("lora_B", "lora_embedding_B"),
}


def apply_lora(
    model: PreTrainedModel,
    config: LoRAConfig | None,
    *,
    match_base_dtype: bool = False,
    lora_dtype: torch.dtype | None = None,
) -> PreTrainedModel:
    if config is None:
        return model
    if isinstance(model, PeftModel):
        enforce_single_lora_adapter(model)
        if match_base_dtype:
            _align_lora_dtypes(model)
        return model
    _normalize_hf_tp_linear_feature_metadata(model)
    target_modules = _resolve_lora_target_modules(model, config)
    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=target_modules,
        modules_to_save=config.modules_to_save or None,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    enforce_single_lora_adapter(model)
    if lora_dtype is not None:
        _cast_lora_dtypes(model, lora_dtype)
    elif match_base_dtype:
        _align_lora_dtypes(model)
    return model


def prepare_hf_tp_lora_for_training(
    model: nn.Module,
    parallel_dims: ParallelDims | None,
) -> None:
    if not _tp_distributed_enabled(parallel_dims):
        return

    try:
        from transformers.integrations.tensor_parallel import all_reduce_forward
    except ImportError:
        return

    tp_mesh = parallel_dims.get_mesh("tp")
    for module in _hf_tp_lora_modules(model, "rowwise"):
        for child in _lora_children(
            module,
            "lora_B",
            "lora_embedding_B",
            adapter_name=_single_module_lora_adapter_name(module),
        ):
            if not _needs_lora_allreduce_hook(child):
                continue

            def _hook(
                _module: nn.Module,
                _inputs: tuple[object, ...],
                output: torch.Tensor,
                *,
                mesh=tp_mesh,
            ) -> torch.Tensor:
                return all_reduce_forward(output, mesh)

            child.register_forward_hook(_hook)
            child._wavelet_tp_lora_allreduce_hook = True


def sync_hf_tp_lora_replicated_grads(
    model: nn.Module,
    parallel_dims: ParallelDims | None,
) -> None:
    if not _tp_distributed_enabled(parallel_dims):
        return

    group = _mesh_process_group(parallel_dims, "tp")
    if torch.distributed.get_world_size(group=group) <= 1:
        return

    for module, tp_plan in _hf_tp_lora_modules_with_plan(model):
        replicated_attrs = TP_REPLICATED_LORA_ATTRS[tp_plan]
        for parameter in _lora_parameters(
            module,
            *replicated_attrs,
            adapter_name=_single_module_lora_adapter_name(module),
        ):
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(
                parameter.grad,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )


def save_lora_adapter_snapshot(
    model: PreTrainedModel,
    output_dir: Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    is_main_process: bool = True,
    parallel_dims: ParallelDims | None = None,
) -> Path:
    """Save only the mutable LoRA adapter files needed for hot policy reload."""
    if not isinstance(model, PeftModel):
        raise TypeError("Lightweight policy snapshots require a PeftModel.")

    target = output_dir / "adapter"
    if _model_uses_hf_tensor_parallel_lora(model):
        state_dict = _gather_hf_tp_lora_state_dict(
            model,
            state_dict=state_dict,
            parallel_dims=parallel_dims,
        )
        if state_dict is None:
            return target
    elif not is_main_process:
        return target

    target.mkdir(parents=True, exist_ok=True)
    adapter_name = _single_peft_adapter_name(model)
    peft_config = model.peft_config[adapter_name]
    peft_config.save_pretrained(target)
    lora_state = get_peft_model_state_dict(
        model,
        state_dict=state_dict,
        adapter_name=adapter_name,
    )
    cpu_state = {
        _strip_fsdp_wrapped_module_segments(key): value.detach().cpu().contiguous()
        for key, value in lora_state.items()
    }
    save_safetensors(cpu_state, target / "adapter_model.safetensors")
    return target


def save_lora_adapter_snapshot_from_fsdp(
    model: FSDP,
    output_dir: Path,
    *,
    is_main_process: bool = True,
    parallel_dims: ParallelDims | None = None,
) -> Path:
    """Save a PEFT LoRA adapter from an FSDP-wrapped model without a full state dict."""
    unwrapped = unwrap_model(model)
    if not isinstance(unwrapped, PeftModel):
        raise TypeError(
            "FSDP lightweight policy snapshots require a wrapped PeftModel."
        )

    state_dict = _gather_fsdp_lora_state_dict(
        model,
        unwrapped,
        parallel_dims=parallel_dims,
    )
    return save_lora_adapter_snapshot(
        unwrapped,
        output_dir,
        state_dict=state_dict,
        is_main_process=is_main_process,
        parallel_dims=parallel_dims,
    )


def _save_lora_adapter_snapshot_from_fsdp_full_params(
    model: FSDP,
    output_dir: Path,
    *,
    is_main_process: bool = True,
) -> Path:
    unwrapped = unwrap_model(model)
    if not isinstance(unwrapped, PeftModel):
        raise TypeError(
            "FSDP lightweight policy snapshots require a wrapped PeftModel."
        )
    with FSDP.summon_full_params(
        model,
        recurse=True,
        writeback=False,
        rank0_only=True,
        offload_to_cpu=True,
    ):
        return save_lora_adapter_snapshot(
            unwrapped,
            output_dir,
            state_dict=None,
            is_main_process=is_main_process,
        )


def _tp_distributed_enabled(parallel_dims: ParallelDims | None) -> bool:
    return bool(
        parallel_dims is not None
        and parallel_dims.tp_enabled
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _needs_lora_allreduce_hook(child: object) -> bool:
    return isinstance(child, nn.Module) and not getattr(
        child,
        "_wavelet_tp_lora_allreduce_hook",
        False,
    )


def _hf_tp_lora_modules(model: nn.Module, *plans: str) -> list[nn.Module]:
    return [
        module
        for module, tp_plan in _hf_tp_lora_modules_with_plan(model)
        if tp_plan in plans
    ]


def _hf_tp_lora_modules_with_plan(model: nn.Module) -> list[tuple[nn.Module, str]]:
    modules: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        tp_plan = _hf_tp_plan(module)
        if tp_plan in TP_REPLICATED_LORA_ATTRS:
            modules.append((module, tp_plan))
    return modules


def _hf_tp_plan(module: nn.Module) -> str | None:
    if not _is_lora_wrapped(module):
        return None
    base_layer = getattr(module, "base_layer", None)
    tp_plan = getattr(base_layer, "_hf_tp_plan", None)
    return tp_plan if isinstance(tp_plan, str) else None


def enforce_single_lora_adapter(model: nn.Module) -> None:
    """Reject PEFT models with more than one adapter before training/export."""
    if not isinstance(model, PeftModel):
        return
    _single_peft_adapter_name(model)


def _lora_children(
    module: nn.Module,
    *attrs: str,
    adapter_name: str | None = None,
) -> list[object]:
    children: list[object] = []
    for attr in attrs:
        container = getattr(module, attr, None)
        if container is None:
            continue
        if isinstance(container, dict):
            children.extend(
                [container[adapter_name]]
                if adapter_name is not None and adapter_name in container
                else container.values()
            )
        elif hasattr(container, "values"):
            values = (
                [container[adapter_name]]
                if adapter_name is not None and adapter_name in container
                else list(container.values())
            )
            children.extend(values)
        else:
            children.append(container)
    return children


def _lora_parameters(
    module: nn.Module,
    *attrs: str,
    adapter_name: str | None = None,
) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for child in _lora_children(module, *attrs, adapter_name=adapter_name):
        if isinstance(child, nn.Parameter):
            parameters.append(child)
            continue
        if isinstance(child, nn.Module):
            parameters.extend(child.parameters())
    return parameters


def _normalize_hf_tp_linear_feature_metadata(model: nn.Module) -> None:
    for module in model.modules():
        if getattr(module, "_hf_tp_plan", None) not in {"colwise", "rowwise"}:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        module.out_features = int(weight.shape[0])
        module.in_features = int(weight.shape[1])


def _cast_lora_dtypes(model: nn.Module, lora_dtype: torch.dtype) -> None:
    adapter_name = (
        _single_peft_adapter_name(model) if isinstance(model, PeftModel) else None
    )
    for _, wrapped in model.named_modules():
        if not _is_lora_wrapped(wrapped):
            continue
        for attr in ("lora_A", "lora_B"):
            container = getattr(wrapped, attr, None)
            if container:
                children = (
                    [container[adapter_name]]
                    if adapter_name is not None and adapter_name in container
                    else list(container.values())
                )
                for child in children:
                    child.to(dtype=lora_dtype)


def _model_uses_hf_tensor_parallel_lora(model: PeftModel) -> bool:
    return bool(_hf_tp_lora_modules_with_plan(model))


def _gather_hf_tp_lora_state_dict(
    model: PeftModel,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    parallel_dims: ParallelDims | None = None,
) -> dict[str, torch.Tensor] | None:
    local_state = state_dict or _local_lora_state_dict(model)
    if not torch.distributed.is_initialized():
        return local_state

    group = _mesh_process_group(parallel_dims, "tp")
    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, torch.Tensor] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_state, group=group)
    if torch.distributed.get_rank() != 0:
        return None

    state: dict[str, torch.Tensor] = {}
    for key, value in local_state.items():
        gather_dim = _hf_tp_lora_gather_dim(model, key)
        if gather_dim is None:
            state[key] = value
            continue
        parts = _state_parts(gathered, key)
        state[key] = torch.cat(parts, dim=gather_dim).contiguous()
    return state


def _local_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    adapter_name = (
        _single_peft_adapter_name(model) if isinstance(model, PeftModel) else None
    )
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.named_parameters()
        if "lora_" in name
        and value.numel() > 0
        and _lora_state_key_matches_adapter(name, adapter_name)
    }


def _hf_tp_lora_gather_dim(model: PeftModel, key: str) -> int | None:
    module_name, attr = _split_lora_state_key(key)
    if module_name is None or attr is None:
        return None
    module = dict(model.named_modules()).get(module_name)
    if module is None:
        return None
    base_layer = getattr(module, "base_layer", None)
    tp_plan = getattr(base_layer, "_hf_tp_plan", None)
    if tp_plan == "colwise" and attr in {"lora_B", "lora_embedding_B"}:
        return 0
    if tp_plan == "rowwise" and attr in {"lora_A", "lora_embedding_A"}:
        return 1
    return None


def _split_lora_state_key(key: str) -> tuple[str | None, str | None]:
    for attr in LORA_STATE_ATTRS:
        marker = f".{attr}."
        if marker in key:
            return key.split(marker, 1)[0], attr
    return None, None


def _strip_fsdp_wrapped_module_segments(key: str) -> str:
    return key.replace("._fsdp_wrapped_module.", ".").removeprefix(
        "_fsdp_wrapped_module."
    )


def _lora_parameter_shapes(model: PeftModel) -> dict[str, tuple[int, ...]]:
    active_adapter_name = _single_peft_adapter_name(model)
    shapes: dict[str, tuple[int, ...]] = {}
    for module_name, module in model.named_modules():
        for attr in LORA_STATE_ATTRS:
            container = getattr(module, attr, None)
            if container is None:
                continue
            for adapter_name, child in container.items():
                if adapter_name != active_adapter_name:
                    continue
                weight = getattr(child, "weight", child)
                if not isinstance(weight, nn.Parameter):
                    continue
                if isinstance(child, nn.Linear):
                    shape = (child.out_features, child.in_features)
                else:
                    shape = tuple(weight.shape)
                # ``lora_embedding_*`` containers hold parameters directly, so
                # their state-dict keys carry no ``.weight`` suffix.
                suffix = "" if isinstance(child, nn.Parameter) else ".weight"
                shapes[f"{module_name}.{attr}.{adapter_name}{suffix}"] = shape
        saved_copies = getattr(module, "modules_to_save", None)
        if isinstance(saved_copies, nn.ModuleDict) and active_adapter_name in (
            saved_copies
        ):
            # ``lora.modules_to_save`` trains full copies (e.g. lm_head) that
            # the adapter snapshot must ship alongside the LoRA tensors.
            for param_name, param in saved_copies[
                active_adapter_name
            ].named_parameters():
                key = (
                    f"{module_name}.modules_to_save.{active_adapter_name}.{param_name}"
                )
                shapes[key] = _unsharded_parameter_shape(
                    saved_copies[active_adapter_name], param_name, param
                )
    return shapes


def _unsharded_parameter_shape(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> tuple[int, ...]:
    owner = module
    attr = param_name
    if "." in param_name:
        owner_path, attr = param_name.rsplit(".", 1)
        owner = module.get_submodule(owner_path)
    if isinstance(owner, nn.Linear) and attr == "weight":
        return (owner.out_features, owner.in_features)
    if isinstance(owner, nn.Linear) and attr == "bias":
        return (owner.out_features,)
    if isinstance(owner, nn.Embedding) and attr == "weight":
        return (owner.num_embeddings, owner.embedding_dim)
    return tuple(param.shape)


def _gather_fsdp_lora_state_dict(
    model: FSDP,
    unwrapped: PeftModel,
    *,
    parallel_dims: ParallelDims | None = None,
) -> dict[str, torch.Tensor] | None:
    if not torch.distributed.is_initialized():
        adapter_name = _single_peft_adapter_name(unwrapped)
        return {
            name.removeprefix("_fsdp_wrapped_module."): value.detach()
            .cpu()
            .contiguous()
            for name, value in model.named_parameters()
            if value.numel() > 0
            and (
                (
                    "lora_" in name
                    and _lora_state_key_matches_adapter(name, adapter_name)
                )
                or f".modules_to_save.{adapter_name}." in name
            )
        }

    expected_shapes = _lora_parameter_shapes(unwrapped)
    local_state = {
        name.removeprefix("_fsdp_wrapped_module."): value.detach().cpu().reshape(-1)
        for name, value in model.named_parameters()
        if name.removeprefix("_fsdp_wrapped_module.") in expected_shapes
        and value.numel() > 0
    }

    # Each HSDP replica holds a complete copy of the sharded adapter, so the
    # gather runs over the shard group only; the 2-D hsdp mesh has no single
    # process group and would also concatenate duplicate replica shards.
    group = _mesh_process_group(parallel_dims, "dp_shard_cp")
    group_rank = _distributed_group_rank(group)
    group_world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, torch.Tensor] | None] | None
    if group_rank == 0:
        gathered = [None for _ in range(group_world_size)]
    else:
        gathered = None
    torch.distributed.gather_object(
        local_state,
        gathered,
        group=group,
        group_dst=0 if group is not None else None,
        dst=0 if group is None else None,
    )
    if group_rank != 0:
        return None
    if gathered is None:
        raise RuntimeError("FSDP LoRA gather returned no state on rank 0.")

    state: dict[str, torch.Tensor] = {}
    for name, shape in expected_shapes.items():
        parts = _state_parts(gathered, name)
        flat = torch.cat(parts, dim=0)
        expected_numel = _numel_from_shape(shape)
        if flat.numel() != expected_numel:
            raise RuntimeError(
                "FSDP LoRA gather produced the wrong size for "
                f"{name}: {flat.numel()} != {expected_numel}."
            )
        state[name] = flat.reshape(shape).contiguous()
    return state


def _state_parts(
    gathered: list[dict[str, torch.Tensor] | None],
    key: str,
) -> list[torch.Tensor]:
    parts = [
        shard[key]
        for shard in gathered
        if shard is not None and key in shard and shard[key].numel() > 0
    ]
    if not parts:
        raise RuntimeError(f"LoRA state gather found no shards for {key}.")
    return parts


def _numel_from_shape(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _mesh_process_group(
    parallel_dims: ParallelDims | None,
    name: str,
) -> object | None:
    if parallel_dims is None:
        return None
    if name == "tp" and not parallel_dims.tp_enabled:
        return None
    if name == "dp_shard_cp" and not parallel_dims.fsdp_enabled:
        return None
    return parallel_dims.get_mesh(name).get_group()


def _distributed_group_rank(group: object | None) -> int:
    if group is None:
        return torch.distributed.get_rank()
    return torch.distributed.get_rank(group=group)


def _single_peft_adapter_name(model: PeftModel) -> str:
    adapter_names = list(getattr(model, "peft_config", {}).keys())
    if len(adapter_names) > 1:
        raise RuntimeError(
            "Wavelet supports exactly one PEFT LoRA adapter per policy model; "
            f"found {adapter_names}."
        )
    active_adapters = getattr(model, "active_adapters", None)
    adapters = active_adapters() if callable(active_adapters) else active_adapters
    adapters = list(adapters or [])
    if len(adapters) > 1:
        raise RuntimeError(
            "Wavelet supports exactly one active LoRA adapter per policy model; "
            f"found {adapters}."
        )
    if not adapter_names and not adapters:
        return "default"
    adapter_name = str(adapter_names[0] if adapter_names else adapters[0])
    if adapters and str(adapters[0]) != adapter_name:
        raise RuntimeError(
            "Wavelet LoRA adapter mismatch: active adapter "
            f"{adapters[0]!r} is not the configured adapter {adapter_name!r}."
        )
    return adapter_name


def _single_module_lora_adapter_name(module: nn.Module) -> str | None:
    names: set[str] = set()
    for attr in LORA_STATE_ATTRS:
        container = getattr(module, attr, None)
        if isinstance(container, dict) or hasattr(container, "keys"):
            names.update(str(name) for name in container)
    if len(names) > 1:
        raise RuntimeError(
            "Wavelet supports exactly one LoRA adapter per wrapped module; "
            f"found {sorted(names)}."
        )
    return next(iter(names)) if names else None


def _lora_state_key_matches_adapter(key: str, adapter_name: str | None) -> bool:
    if adapter_name is None:
        return True
    key_adapter_name = _lora_state_key_adapter_name(key)
    return key_adapter_name is None or key_adapter_name == adapter_name


def _lora_state_key_adapter_name(key: str) -> str | None:
    for attr in LORA_STATE_ATTRS:
        marker = f".{attr}."
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        if "." not in suffix:
            return None
        return suffix.split(".", 1)[0]
    return None


def _is_lora_wrapped(module: nn.Module) -> bool:
    return hasattr(module, "base_layer") and hasattr(module, "lora_B")


def _resolve_lora_target_modules(
    model: PreTrainedModel,
    config: LoRAConfig,
) -> list[str]:
    configured = list(config.target_modules)
    if configured != DEFAULT_LORA_TARGET_MODULES:
        return configured
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "gpt2":
        return configured
    return DEBUG_LORA_TARGET_MODULES


def _align_lora_dtypes(model: nn.Module) -> None:
    adapter_name = (
        _single_peft_adapter_name(model) if isinstance(model, PeftModel) else None
    )
    for module in model.modules():
        if not _is_lora_wrapped(module):
            continue
        base_weight = getattr(module.base_layer, "weight", None)
        if base_weight is None:
            continue
        target_dtype = base_weight.dtype
        target_device = base_weight.device
        for attr in LORA_STATE_ATTRS:
            container = getattr(module, attr, None)
            if container is None:
                continue
            children = (
                [container[adapter_name]]
                if adapter_name is not None and adapter_name in container
                else list(container.values())
            )
            for child in children:
                if isinstance(child, nn.Module):
                    child.to(device=target_device, dtype=target_dtype)
                elif isinstance(child, nn.Parameter):
                    child.data = child.data.to(
                        device=target_device,
                        dtype=target_dtype,
                    )

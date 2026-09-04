"""Model loading, quantization, wrapping, and LoRA support."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch
import torch.distributed
from huggingface_hub import snapshot_download
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
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.fsdp import (
    CPUOffload,
    CPUOffloadPolicy,
    FSDPModule,
    FullStateDictConfig,
    MixedPrecision,
    MixedPrecisionPolicy,
    OffloadPolicy,
    ShardingStrategy,
    StateDictType,
    fully_shard,
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
from wavelet.configs.sft import (
    ActivationCheckpointingConfig,
    FSDPConfig,
    LoRAConfig,
    ModelConfig,
)
from wavelet.trainer.debug import (
    DEBUG_LORA_TARGET_MODULES,
    DEBUG_MODEL_NAME,
    build_debug_model,
    build_debug_tokenizer,
)
from wavelet.trainer.distributed import ParallelDims, World

logger = logging.getLogger(__name__)


def pre_download_model(model_name: str) -> Path | None:
    """Populate the Hugging Face cache once before launcher roles start."""
    local_path = Path(model_name)
    if model_name == DEBUG_MODEL_NAME or local_path.exists():
        logger.info("Model %s is local; skipping pre-download.", model_name)
        return local_path if local_path.exists() else None

    started_at = perf_counter()
    logger.info("Pre-downloading model %s.", model_name)
    downloaded = Path(snapshot_download(repo_id=model_name, repo_type="model"))
    logger.info(
        "Pre-downloaded model %s to %s in %.2fs.",
        model_name,
        downloaded,
        perf_counter() - started_at,
    )
    return downloaded


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
    if config.smart_gc and config.activation_checkpointing is not None:
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


def _build_pretrained_model_on_meta(
    config: ModelConfig,
    *,
    attention: str,
) -> PreTrainedModel:
    model_config = AutoConfig.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code,
    )
    dtype = resolve_dtype(config.torch_dtype)
    if dtype == "auto":
        dtype = getattr(model_config, "dtype", None) or torch.float32
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=config.trust_remote_code,
            attn_implementation=attention,
            dtype=dtype,
        )
    return cast(PreTrainedModel, model)


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
    initialize_on_meta: bool = False,
) -> PreTrainedModel:
    if config.name == DEBUG_MODEL_NAME:
        if initialize_on_meta:
            raise ValueError(
                "The random debug model cannot be loaded from Hugging Face shards."
            )
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
    if initialize_on_meta:
        if config.load_in_4bit:
            raise ValueError(
                "Meta-device model construction does not support 4-bit load."
            )
        if parallel_dims is not None and parallel_dims.tp_enabled:
            raise ValueError(
                "Meta-device FSDP2 loading with tensor parallelism is not yet "
                "validated. Disable model.meta_device_init for this configuration."
            )
        if config.adapter_path is not None:
            raise ValueError(
                "Meta-device FSDP2 loading cannot resume a PEFT adapter yet. "
                "Disable model.meta_device_init for this configuration."
            )
        model = _build_pretrained_model_on_meta(config, attention=attention)
    else:
        model = _load_pretrained_model(config, model_kwargs, attention=attention)
    model.config.use_cache = False
    if config.smart_gc and config.activation_checkpointing is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if config.load_in_4bit and not model_is_prequantized:
        model = prepare_kbit_model(
            model,
            gradient_checkpointing=config.activation_checkpointing is not None,
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


def load_fsdp2_model_from_hf(
    model: nn.Module,
    config: ModelConfig,
    *,
    world: World,
    cpu_offload: bool = False,
) -> None:
    """Materialize an FSDP2 meta model directly from HF safetensor shards."""
    if not isinstance(model, FSDPModule):
        raise TypeError("Direct Hugging Face shard loading requires an FSDP2 model.")

    snapshot_path = Path(config.name)
    if not snapshot_path.exists():
        snapshot_path = Path(snapshot_download(repo_id=config.name, repo_type="model"))
    reader = HuggingFaceStorageReader(snapshot_path.as_posix())
    checkpoint_keys = set(reader.read_metadata().state_dict_metadata)

    meta_state_dict = model.state_dict()
    _validate_meta_model_buffers(model, set(meta_state_dict))
    tied_checkpoint_keys = _tied_checkpoint_keys(model)
    missing = [
        model_key
        for model_key in meta_state_dict
        if not _is_lora_state_key(model_key)
        and _hf_checkpoint_key(model_key) not in checkpoint_keys
        and _hf_checkpoint_key(model_key) not in tied_checkpoint_keys
    ]
    if missing:
        names = ", ".join(sorted(missing)[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise RuntimeError(
            "Hugging Face checkpoint does not match the meta-initialized model; "
            f"missing {len(missing)} state entries: {names}{suffix}"
        )

    load_device = (
        torch.device("cpu")
        if cpu_offload or world.device.type == "cpu"
        else world.device
    )
    model.to_empty(device=load_device)
    torch.distributed.barrier()
    _initialize_meta_model_buffers(model)

    state_dict = model.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    for model_key, value in state_dict.items():
        if _is_lora_state_key(model_key):
            continue
        checkpoint_key = _hf_checkpoint_key(model_key)
        if checkpoint_key in tied_checkpoint_keys:
            continue
        load_state.setdefault(checkpoint_key, value)

    logger.info("Loading FSDP2 weights directly from %s.", snapshot_path)
    dcp_load(load_state, storage_reader=reader)
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
    _initialize_lora_parameters(model)


def _hf_checkpoint_key(model_key: str) -> str:
    key = model_key.removeprefix("base_model.model.")
    return key.replace(".base_layer.", ".")


def _is_lora_state_key(key: str) -> bool:
    return any(f".{attr}." in key for attr in LORA_STATE_ATTRS)


def _tied_checkpoint_keys(model: nn.Module) -> set[str]:
    if not getattr(model.config, "tie_word_embeddings", False):
        return set()
    tied = getattr(model, "_tied_weights_keys", None)
    if isinstance(tied, dict):
        return set(tied)
    if isinstance(tied, (list, set, tuple)):
        return set(tied)
    return set()


def _validate_meta_model_buffers(
    model: nn.Module,
    persistent_names: set[str],
) -> None:
    unsupported = [
        name
        for name, _ in model.named_buffers()
        if name not in persistent_names
        and not name.endswith(("rotary_emb.inv_freq", "rotary_emb.original_inv_freq"))
    ]
    if unsupported:
        names = ", ".join(unsupported[:5])
        suffix = "..." if len(unsupported) > 5 else ""
        raise RuntimeError(
            "Meta-device initialization cannot reconstruct model buffers: "
            f"{names}{suffix}. Disable model.meta_device_init."
        )


def _initialize_meta_model_buffers(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            inv_freq = getattr(module, "inv_freq", None)
            if not isinstance(inv_freq, torch.Tensor):
                continue
            rope_init_fn = getattr(module, "rope_init_fn", None)
            if rope_init_fn is None:
                rope_init_fn = getattr(module, "compute_default_rope_parameters", None)
            module_config = getattr(module, "config", None)
            if not callable(rope_init_fn) or module_config is None:
                raise RuntimeError(
                    "Meta-device initialization cannot reconstruct rotary buffers "
                    f"for {type(module).__name__}. Disable model.meta_device_init."
                )
            initialized, attention_scaling = rope_init_fn(
                module_config,
                inv_freq.device,
            )
            inv_freq.copy_(initialized)
            if hasattr(module, "original_inv_freq"):
                module.original_inv_freq.copy_(initialized)
            if hasattr(module, "attention_scaling"):
                module.attention_scaling = attention_scaling


def _initialize_lora_parameters(model: nn.Module) -> None:
    for module in model.modules():
        reset = getattr(module, "reset_lora_parameters", None)
        lora_a = getattr(module, "lora_A", None)
        if not callable(reset) or not isinstance(lora_a, nn.ModuleDict):
            continue
        for adapter_name in lora_a:
            reset(adapter_name, True)


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

    if fsdp_config.impl == "fsdp2":
        return _wrap_fsdp2(
            model,
            model_config=model_config,
            fsdp_config=fsdp_config,
            parallel_dims=parallel_dims,
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


def _wrap_fsdp2(
    model: PreTrainedModel,
    *,
    model_config: ModelConfig,
    fsdp_config: FSDPConfig,
    parallel_dims: ParallelDims | None,
) -> PreTrainedModel:
    if parallel_dims is None:
        raise RuntimeError("FSDP2 requires initialized parallel dimensions.")
    if parallel_dims.ep_enabled:
        raise NotImplementedError(
            "Expert parallel execution is not wired into the model stack yet."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context parallel execution is not wired into the attention stack yet."
        )

    mesh = parallel_dims.get_mesh("hsdp")
    mp_policy = _fsdp2_mixed_precision(model_config)
    offload_policy: OffloadPolicy = (
        CPUOffloadPolicy(pin_memory=torch.cuda.is_available())
        if fsdp_config.cpu_offload
        else OffloadPolicy()
    )
    shard_kwargs = {
        "mesh": mesh,
        "mp_policy": mp_policy,
        "offload_policy": offload_policy,
        "reshard_after_forward": fsdp_config.reshard_after_forward,
    }
    layer_classes = _transformer_layer_classes(model)
    transformer_blocks = [
        module
        for module in model.modules()
        if module is not model and type(module) in layer_classes
    ]
    for block in transformer_blocks:
        fully_shard(block, **shard_kwargs)
    fully_shard(model, **shard_kwargs)
    return cast(PreTrainedModel, model)


def _fsdp2_mixed_precision(model_config: ModelConfig) -> MixedPrecisionPolicy:
    dtype = resolve_dtype(model_config.torch_dtype)
    if not isinstance(dtype, torch.dtype):
        return MixedPrecisionPolicy()
    if not torch.cuda.is_available() and dtype is not torch.float32:
        return MixedPrecisionPolicy()
    if dtype is torch.float32 and torch.cuda.is_available():
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)


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
    if not is_fsdp_model(model):
        unwrapped = unwrap_model(model)
        if state_dict_dtype is None:
            return unwrapped, None
        return unwrapped, _state_dict_to_save_dtype(
            unwrapped.state_dict(),
            state_dict_dtype,
        )

    if isinstance(model, FSDPModule):
        gathered = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        state_dict = {
            key: value
            for key, value in gathered.items()
            if isinstance(value, torch.Tensor)
        }
    else:
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            state_dict = model.state_dict()
    if state_dict_dtype is not None:
        state_dict = _state_dict_to_save_dtype(state_dict, state_dict_dtype)
    return unwrap_model(model), state_dict


def is_fsdp_model(model: nn.Module) -> bool:
    """Return whether the root uses either supported FSDP implementation."""
    return isinstance(model, (FSDP, FSDPModule))


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


DEFAULT_SELECTIVE_CHECKPOINT_TARGETS = frozenset(
    {
        "aten::_efficient_attention_forward",
        "aten::_flash_attention_forward",
        "aten::_scaled_dot_product_attention_math",
        "aten::_scaled_dot_product_cudnn_attention",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_flash_attention_for_cpu",
        "aten::addmm",
        "aten::bmm",
        "aten::linear",
        "aten::mm",
        "flash_attn",
        "flash_attn_3",
    }
)


def _selective_checkpoint_policy(
    _context: object,
    operation: object,
    *_args: object,
    targets: frozenset[str],
    **_kwargs: object,
) -> object:
    from torch.utils.checkpoint import CheckpointPolicy

    namespace = getattr(operation, "namespace", None)
    name = operation.name() if callable(getattr(operation, "name", None)) else None
    if namespace in targets or name in targets:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def apply_activation_checkpointing(
    model: nn.Module,
    config: ActivationCheckpointingConfig,
) -> int:
    """Wrap every configured decoder block with non-reentrant checkpointing."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    layer_classes = _transformer_layer_classes(model)
    candidates = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if type(child) in layer_classes
    ]
    if not candidates:
        raise ValueError(
            "model.activation_checkpointing requires a model that identifies "
            "decoder blocks through _no_split_modules."
        )
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    context_fn = None
    if config.mode == "selective":
        targets = (
            DEFAULT_SELECTIVE_CHECKPOINT_TARGETS
            if config.targets is None
            else frozenset(config.targets)
        )
        policy = partial(_selective_checkpoint_policy, targets=targets)
        context_fn = partial(create_selective_checkpoint_contexts, policy)

    wrapped_count = 0
    for layer_index, (parent, name, layer) in enumerate(candidates):
        if layer_index % config.freq != 0:
            continue
        kwargs: dict[str, object] = {
            "checkpoint_impl": CheckpointImpl.NO_REENTRANT,
        }
        if context_fn is not None:
            kwargs["context_fn"] = context_fn
        parent.register_module(name, checkpoint_wrapper(layer, **kwargs))
        wrapped_count += 1
    logger.info(
        "Applied %s activation checkpointing to %s/%s transformer layers",
        config.mode,
        wrapped_count,
        len(candidates),
    )
    return wrapped_count


def compile_transformer_layers(
    model: nn.Module,
    *,
    fullgraph: bool,
    backend: str | None = None,
) -> int:
    """Compile decoder blocks in place while preserving state-dict names."""
    layer_classes = _transformer_layer_classes(model)
    if not layer_classes:
        raise ValueError(
            "model.compile=true requires a model that identifies transformer "
            "blocks through _no_split_modules."
        )

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit,
        16,
    )
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit,
        64,
    )
    compile_kwargs: dict[str, object] = {"fullgraph": fullgraph}
    if backend is not None:
        compile_kwargs["backend"] = backend

    compiled = 0
    for module in model.modules():
        if type(module) not in layer_classes:
            continue
        module.compile(**compile_kwargs)
        compiled += 1
    logger.info("Compiled %s transformer layers (fullgraph=%s)", compiled, fullgraph)
    return compiled


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
    model: nn.Module,
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

    if isinstance(model, FSDPModule):
        gathered = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True,
            ),
        )
        state_dict = {
            key: value
            for key, value in gathered.items()
            if isinstance(value, torch.Tensor)
        }
    else:
        state_dict = _gather_fsdp_lora_state_dict(
            cast(FSDP, model),
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

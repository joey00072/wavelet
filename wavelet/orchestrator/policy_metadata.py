from __future__ import annotations

from typing import Any

from wavelet.configs.rl_config import RLConfig


def precision_metadata(config: RLConfig) -> dict[str, Any]:
    inference_dtype = config.inference.vllm.dtype or config.model.torch_dtype
    return {
        "trainer": {
            "model_name": config.model.name,
            "torch_dtype": config.model.torch_dtype,
            "load_in_4bit": config.model.load_in_4bit,
            "optimizer": config.optim.type,
            "fsdp_enabled": config.fsdp.enabled,
            "tensor_parallel_size": config.fsdp.tp,
        },
        "adapter": {
            "enabled": config.lora is not None,
            "path": str(config.model.adapter_path)
            if config.model.adapter_path is not None
            else None,
            "dtype": config.model.torch_dtype if config.lora is not None else None,
            "rank": config.lora.rank if config.lora is not None else None,
        },
        "inference": {
            "dtype": inference_dtype,
            "quantization": config.inference.vllm.quantization,
            "load_format": config.inference.vllm.load_format,
            "tensor_parallel_size": config.inference.vllm.tensor_parallel_size,
            "data_parallel_size": config.inference.vllm.data_parallel_size,
        },
        "train_serve_dtype_match": config.model.torch_dtype == inference_dtype,
        "train_serve_low_precision_match": _low_precision_match(config),
    }


def policy_metadata(
    *,
    config: RLConfig,
    format_version: int,
    step: int,
    kind: str,
    created_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format_version": format_version,
        "step": step,
        "kind": kind,
        "precision": precision_metadata(config),
    }
    if created_at is not None:
        metadata["created_at"] = created_at
    if extra:
        metadata.update(extra)
    return metadata


def _low_precision_match(config: RLConfig) -> bool:
    inference_uses_low_precision = bool(
        config.inference.vllm.quantization or config.inference.vllm.load_format
    )
    return config.model.load_in_4bit == inference_uses_low_precision

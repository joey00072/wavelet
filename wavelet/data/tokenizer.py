"""Model-adjacent tokenizer and processor loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

from wavelet.configs.config import ModelConfig
from wavelet.data.debug_tokenizer import build_debug_tokenizer

logger = logging.getLogger(__name__)
DEBUG_MODEL_NAME = "debug/tiny-random"


def setup_processor(config: ModelConfig) -> Any | None:
    """Load the processor beside a VLM's adapter or base checkpoint."""
    if config.vlm is None:
        return None
    source = config.adapter_path or config.name
    try:
        processor = AutoProcessor.from_pretrained(
            source, trust_remote_code=config.trust_remote_code
        )
    except (AttributeError, KeyError, OSError, ValueError) as exc:
        if config.adapter_path is None:
            raise ValueError(
                f"VLM model {config.name!r} requires a usable AutoProcessor"
            ) from exc
        logger.warning(
            "Adapter has no usable processor; loading the base checkpoint processor."
        )
        processor = AutoProcessor.from_pretrained(
            config.name, trust_remote_code=config.trust_remote_code
        )
    if not any(
        getattr(processor, name, None) is not None
        for name in ("image_processor", "video_processor")
    ):
        raise ValueError(
            f"VLM model {config.name!r} processor has no image/video processor"
        )
    if config.chat_template is not None:
        processor.chat_template = config.chat_template
    return processor


def pre_download_model(model_name: str) -> Path | None:
    """Populate the Hugging Face cache before launcher roles start."""
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


def setup_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    """Load and normalize the tokenizer for a model or adapter."""
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

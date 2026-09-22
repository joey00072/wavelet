from __future__ import annotations


def transformers_v5_compat() -> None:
    """Apply compatibility fixes that are not included in vLLM 0.28."""
    try:
        from transformers import Qwen3VLMoeTextConfig
    except ImportError:
        Qwen3VLMoeTextConfig = None
    if Qwen3VLMoeTextConfig is not None and not hasattr(
        Qwen3VLMoeTextConfig, "tie_word_embeddings"
    ):
        Qwen3VLMoeTextConfig.tie_word_embeddings = False

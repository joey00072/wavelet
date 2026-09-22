from __future__ import annotations


def transformers_v5_compat() -> None:
    """Apply compatibility fixes that are not included in vLLM 0.28."""
    from wavelet.trainer.models.deepseek_v4.compat import allow_deepseek_v4_layer_types

    allow_deepseek_v4_layer_types()
    try:
        from transformers import Qwen3VLMoeTextConfig
    except ImportError:
        Qwen3VLMoeTextConfig = None
    if Qwen3VLMoeTextConfig is not None and not hasattr(
        Qwen3VLMoeTextConfig, "tie_word_embeddings"
    ):
        Qwen3VLMoeTextConfig.tie_word_embeddings = False

    import torch

    if torch.cuda.is_available():
        from wavelet.inference.deepseek_v4 import (
            monkey_patch_deepseek_v4_bf16_o_proj,
            monkey_patch_deepseek_v4_per_layer_rope,
        )

        monkey_patch_deepseek_v4_per_layer_rope()
        monkey_patch_deepseek_v4_bf16_o_proj()

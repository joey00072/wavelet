"""Transformers compatibility for DeepSeek-V4 layer type validation."""

DEEPSEEK_V4_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)


def allow_deepseek_v4_layer_types() -> None:
    from transformers import configuration_utils

    unknown = tuple(
        value
        for value in DEEPSEEK_V4_LAYER_TYPES
        if value not in configuration_utils.ALLOWED_LAYER_TYPES
    )
    if unknown:
        configuration_utils.ALLOWED_LAYER_TYPES += unknown

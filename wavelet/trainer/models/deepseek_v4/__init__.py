"""Native eager DeepSeek-V4 with explicit Hugging Face auto-loader registration."""

from .configuration_deepseek_v4 import DeepseekV4Config
from .hyperconnections import DeepseekV4HyperConnection, DeepseekV4HyperHead
from .rotary import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb_interleaved

__all__ = [
    "DeepseekV4Config",
    "DeepseekV4HyperConnection",
    "DeepseekV4HyperHead",
    "DeepseekV4RotaryEmbedding",
    "apply_rotary_pos_emb_interleaved",
]


def register_model() -> None:
    """Register the eager native model with Hugging Face's explicit auto loaders."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .modeling import DeepseekV4ForCausalLM

    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
    AutoModelForCausalLM.register(
        DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True
    )

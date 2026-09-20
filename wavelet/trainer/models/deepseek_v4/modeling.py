"""DeepSeek-V4 decoder using the native eager attention and mHC components.

Adapted from PrimeRL's Apache-2.0 decoder structure; see LICENSE. Wavelet's
wrapper returns standard Hugging Face logits and derives packed boundaries
from the actual input positions. Context parallelism and cached decoding are
not supported by this eager implementation.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .attention import DeepseekV4Attention, PackedContext
from .configuration_deepseek_v4 import DeepseekV4Config
from .hyperconnections import DeepseekV4HyperConnection, DeepseekV4HyperHead, RMSNorm
from .moe import DeepseekV4Experts, DeepseekV4MoE
from .rotary import DeepseekV4RotaryEmbedding


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV4Config, index: int):
        super().__init__()
        self.self_attn = DeepseekV4Attention(config, index)
        self.mlp = DeepseekV4MoE(config, index)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self, states: Tensor, input_ids: Tensor, packed: PackedContext
    ) -> Tensor:
        post, combine, collapsed = self.attn_hc(states)
        output, _ = self.self_attn(self.input_layernorm(collapsed), packed)
        states = self.attn_hc.update_states(post, combine, output, states)
        post, combine, collapsed = self.ffn_hc(states)
        output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return self.ffn_hc.update_states(post, combine, output, states)


class DeepseekV4Model(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.config = config

    def init_buffers_post_meta(self) -> None:
        """Restore non-persistent rotary tables after Hugging Face checkpoint loading."""
        for module in self.modules():
            if module is not self and hasattr(module, "init_buffers_post_meta"):
                module.init_buffers_post_meta()
            usage = getattr(module, "tokens_per_expert", None)
            if usage is not None and usage.device.type != "meta":
                usage.zero_()

    def forward(self, input_ids: Tensor, position_ids: Tensor | None = None) -> Tensor:
        batch, length = input_ids.shape
        if length == 0:
            raise ValueError("DeepSeek-V4 requires a nonempty sequence.")
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device).expand(
                batch, -1
            )
        if position_ids.shape != input_ids.shape or (position_ids[:, 0] != 0).any():
            raise ValueError(
                "DeepSeek-V4 positions must match inputs and start at zero in each row."
            )
        starts = (position_ids.reshape(-1) == 0).nonzero().flatten()
        ends = torch.cat((starts[1:], starts.new_tensor([batch * length])))
        seq_lens = ends - starts
        flat_ids = input_ids.reshape(1, -1)
        embedded = self.embed_tokens(flat_ids)
        packed = PackedContext.build(
            rotary_emb=self.rotary_emb,
            seq_lens=seq_lens,
            dtype=embedded.dtype,
            device=embedded.device,
        )
        if not torch.equal(packed.position_ids, position_ids.reshape(1, -1)):
            raise ValueError(
                "DeepSeek-V4 positions must count consecutively within each document."
            )
        states = (
            embedded.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        )
        for layer in self.layers:
            states = layer(states, flat_ids, packed)
        return self.norm(self.hc_head(states)).reshape(batch, length, -1)


class DeepseekV4ForCausalLM(PreTrainedModel):
    config_class = DeepseekV4Config
    checkpoint_format = "deepseek_v4_native_v1"
    base_model_prefix = "model"
    _no_split_modules: ClassVar[list[str]] = ["DeepseekV4DecoderLayer"]
    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "lm_head.weight": "model.embed_tokens.weight"
    }
    # These parameters are numerically sensitive in the V4 reference implementation and
    # must remain FP32 when the rest of a checkpoint is transferred in BF16.
    _keep_in_fp32_modules_strict: ClassVar[tuple[str, ...]] = (
        "attn_hc",
        "ffn_hc",
        "hc_head",
        "sinks",
        "position_bias",
        "selection_bias",
        "q_a_norm",
        "kv_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    )

    def __init__(self, config: DeepseekV4Config):
        config.wavelet_checkpoint_format = self.checkpoint_format
        super().__init__(config)
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, DeepseekV4Experts):
            for parameter in module.parameters():
                nn.init.normal_(parameter, std=self.config.initializer_range)
        elif isinstance(
            module,
            (DeepseekV4Attention, DeepseekV4HyperConnection, DeepseekV4HyperHead),
        ):
            module.init_weights(self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            module.init_buffers_post_meta()

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return any(part in name for part in cls._keep_in_fp32_modules_strict)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        loaded = super().from_pretrained(*args, **kwargs)
        if isinstance(loaded, tuple):
            model, loading_info = loaded
        else:
            model, loading_info = loaded, None
        model.model.init_buffers_post_meta()
        return (model, loading_info) if loading_info is not None else model

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.model.init_buffers_post_meta()
        return result

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        if use_cache or kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Native eager DeepSeek-V4 does not support cached decoding."
            )
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "DeepSeek-V4 accepts only a 2D right-padding mask; use position_ids for packed boundaries."
                )
            if (attention_mask[:, 1:].int() > attention_mask[:, :-1].int()).any():
                raise ValueError("DeepSeek-V4 eager training requires right padding.")
        logits = self.lm_head(self.model(input_ids, position_ids))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

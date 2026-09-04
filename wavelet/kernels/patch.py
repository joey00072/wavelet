"""Patch a PEFT-wrapped model with wavelet's native fused LoRA kernels.

patch_fused_mlp(model)  -> replaces MLP forward with LoRA_MLP
patch_fused_qkv(model)  -> replaces Qwen3Attention forward with fused QKV+O
patch_fused_o(model)    -> replaces o_proj.forward with LoRA_W (standalone)
patch_smart_gc(...)     -> gradient checkpointing with CPU offload (native)

patch_fused_qkv handles both QKV and O projections by replacing the full
attention forward. patch_fused_o is a standalone fallback for when QKV
fusion isn't applied (e.g. non-Qwen3 models or only MLP is requested).
They compose safely: if both are applied, o_proj.forward is the fused
closure and patch_fused_qkv calls self.o_proj(x) which hits it.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────


def _find_mlp_class(model: PreTrainedModel) -> type | None:
    """Return the actual MLP class used by the model's decoder layers."""
    for module in model.modules():
        if (
            hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
        ):
            return type(module)
    return None


def _is_fusable_lora_linear(proj: object) -> bool:
    """True for a bias-free PEFT LoRA linear the fused kernels can replace."""
    if proj is None or not hasattr(proj, "lora_A"):
        return False
    base_layer = getattr(proj, "base_layer", proj)
    return getattr(base_layer, "bias", None) is None


def _iter_decoder_layers(model: PreTrainedModel):
    """Yield each transformer decoder layer (the sub-model layers list)."""
    # Unwrap PeftModel → base model → model → layers
    base = model
    for attr in ("base_model", "model", "model"):
        base = getattr(base, attr, base)
    layers = getattr(base, "layers", None)
    if layers is None:
        return
    yield from layers


# ── MLP fusion ────────────────────────────────────────────────────────────────


def patch_fused_mlp(model: PreTrainedModel) -> bool:
    """Replace MLP.forward with wavelet's LoRA_MLP fused SwiGLU kernel.

    LoRA_MLP fuses gate_proj + up_proj + silu + down_proj + all three LoRA
    pairs into one custom autograd.Function with a hand-written backward.
    Intermediate activations (e, g, h) are NOT saved — recomputed in backward.

    Returns True if patching succeeded.
    """
    from wavelet.kernels.lora import apply_lora_mlp_swiglu

    mlp_cls = _find_mlp_class(model)
    if mlp_cls is None:
        logger.warning("patch_fused_mlp: no SwiGLU MLP found in model, skipping")
        return False

    patched = 0
    for mlp in model.modules():
        if not isinstance(mlp, mlp_cls):
            continue
        if not all(
            _is_fusable_lora_linear(getattr(mlp, name, None))
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        mlp.forward = MethodType(apply_lora_mlp_swiglu, mlp)
        patched += 1

    if patched:
        logger.info(
            f"patch_fused_mlp: patched {patched} {mlp_cls.__name__} modules → LoRA_MLP"
        )
    else:
        logger.warning("patch_fused_mlp: no eligible LoRA MLP modules found")
    return patched > 0


# ── QKV fusion (Qwen3) ────────────────────────────────────────────────────────


def patch_fused_qkv(model: PreTrainedModel) -> bool:
    """Replace Qwen3Attention.forward with a fused QKV + optional fused-O path.

    Patches the class-level forward so that any attention instance with
    _wavelet_qkv_patched set uses LoRA_QKV for Q/K/V projections in one fused
    autograd.Function. Instances without the flag fall through to the original.

    Also calls self.o_proj(x) normally in the patched path — if patch_fused_o
    has already replaced o_proj.forward, the fused LoRA_W kernel runs there.

    Returns True if any layers were patched.
    """
    from wavelet.kernels.lora import apply_lora_qkv

    try:
        from transformers.models.qwen3.modeling_qwen3 import (
            ALL_ATTENTION_FUNCTIONS,
            Qwen3Attention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
    except ImportError:
        logger.warning("patch_fused_qkv: Qwen3Attention not found, skipping")
        return False

    _original_forward = Qwen3Attention.forward

    def _fused_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if not hasattr(self, "_wavelet_qkv_patched"):
            # The upstream signature takes cache_position through **kwargs only.
            if cache_position is not None:
                kwargs["cache_position"] = cache_position
            return _original_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Fused Q/K/V + LoRA in one autograd.Function; X saved once for backward
        Q, K, V = self.apply_qkv(self, hidden_states)

        # Qwen3 applies per-head RMS norms on Q and K before RoPE
        query_states = self.q_norm(Q.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(K.view(hidden_shape)).transpose(1, 2)
        value_states = V.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # o_proj.forward may be the fused LoRA_W closure (from patch_fused_o)
        # or the standard PEFT forward — either way we just call it normally.
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    patched = 0
    for layer in _iter_decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not isinstance(attn, Qwen3Attention):
            continue
        if not all(
            _is_fusable_lora_linear(getattr(attn, name, None))
            for name in ("q_proj", "k_proj", "v_proj")
        ):
            continue
        attn.apply_qkv = apply_lora_qkv
        attn._wavelet_qkv_patched = True
        patched += 1

    if not patched:
        logger.warning("patch_fused_qkv: no eligible Qwen3Attention layers found")
        return False

    # Only swap the class forward once eligible layers exist; a no-op patch
    # would still reroute every Qwen3Attention instance in the process.
    if not getattr(Qwen3Attention, "_wavelet_qkv_class_patched", False):
        Qwen3Attention.forward = _fused_forward
        Qwen3Attention._wavelet_qkv_class_patched = True
    logger.info(f"patch_fused_qkv: patched {patched} Qwen3Attention layers (QKV fused)")
    return True


# ── O-proj fusion (standalone) ────────────────────────────────────────────────


def patch_fused_o(model: PreTrainedModel) -> bool:
    """Replace o_proj.forward with the fused LoRA_W kernel.

    Patches each eligible o_proj module instance's forward so that when the
    attention forward calls self.o_proj(x), the fused kernel runs. Works
    standalone or in combination with patch_fused_qkv.

    Returns True if any layers were patched.
    """
    from wavelet.kernels.lora import LoRA_W
    from wavelet.kernels.utils import get_lora_parameters

    patched = 0
    for layer in _iter_decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        o_proj = getattr(attn, "o_proj", None)
        if not _is_fusable_lora_linear(o_proj):
            continue

        def _make_forward(proj):
            def fused_forward(x):
                W, W_quant, A, B, S = get_lora_parameters(proj)
                return LoRA_W.apply(x, W, W_quant, A, B, S)

            return fused_forward

        o_proj.forward = _make_forward(o_proj)
        patched += 1

    if patched:
        logger.info(f"patch_fused_o: patched forward on {patched} o_proj modules")
    else:
        logger.warning("patch_fused_o: no eligible o_proj layers found")
    return patched > 0


# ── Smart gradient checkpointing ─────────────────────────────────────────────


def patch_smart_gc(
    model: PreTrainedModel,
    *,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """Apply Wavelet smart gradient checkpointing with CPU offloading.

    Delegates to wavelet.kernels.smart_gc.patch_smart_gc which monkey-patches
    torch.utils.checkpoint.CheckpointFunction to offload large (> 2 MB)
    activation tensors to pinned CPU RAM during forward and restore them in
    backward.  For seq_len < 512 the I/O overhead is not worth it.

    Returns True if patching succeeded.
    """
    from wavelet.kernels.smart_gc import patch_smart_gc as _patch

    return _patch(model, seq_len=seq_len, dtype=dtype)

from __future__ import annotations

from transformers import GPT2Config, GPT2LMHeadModel

from wavelet.data.debug_tokenizer import (
    DebugTokenizer,
    _debug_vocab_tokens,
)

DEBUG_MODEL_NAME = "debug/tiny-random"
DEBUG_LORA_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]


def build_debug_tokenizer(*, model_max_length: int) -> DebugTokenizer:
    from wavelet.data.debug_tokenizer import build_debug_tokenizer as build

    return build(model_max_length=model_max_length)


def build_debug_model(*, max_seq_length: int | None) -> GPT2LMHeadModel:
    context_length = max(max_seq_length or 128, 64)
    vocab_size = len(_debug_vocab_tokens())
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=context_length,
        n_ctx=context_length,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(config)

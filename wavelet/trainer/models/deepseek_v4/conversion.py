"""Exact published DeepSeek-V4 checkpoint names to native Wavelet tensors.

Mappings adapted from PrimeRL (Apache-2.0); see LICENSE. Quantized raw tensors
are dequantized on import. Export emits unquantized tensors under raw names.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from .dequantize import dequantize_state_dict_

_LAYER_PAIRS = (
    ("attn_norm.weight", "input_layernorm.weight"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ("hc_attn_fn", "attn_hc.fn"),
    ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"),
    ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_scale", "ffn_hc.scale"),
    ("attn.wq_a.weight", "self_attn.q_a_proj.weight"),
    ("attn.q_norm.weight", "self_attn.q_a_norm.weight"),
    ("attn.wq_b.weight", "self_attn.q_b_proj.weight"),
    ("attn.wkv.weight", "self_attn.kv_proj.weight"),
    ("attn.kv_norm.weight", "self_attn.kv_norm.weight"),
    ("attn.wo_a.weight", "self_attn.o_a_proj.weight"),
    ("attn.wo_b.weight", "self_attn.o_b_proj.weight"),
    ("attn.attn_sink", "self_attn.sinks"),
    ("ffn.gate.weight", "mlp.router.gate.weight"),
    ("ffn.gate.bias", "mlp.router.selection_bias"),
    ("ffn.gate.tid2eid", "mlp.router.tid2eid"),
    ("ffn.shared_experts.w1.weight", "mlp.shared_expert.gate_proj.weight"),
    ("ffn.shared_experts.w2.weight", "mlp.shared_expert.down_proj.weight"),
    ("ffn.shared_experts.w3.weight", "mlp.shared_expert.up_proj.weight"),
)
_COMPRESSOR_PAIRS = (
    ("wkv.weight", "kv_proj.weight"),
    ("wgate.weight", "gate_proj.weight"),
    ("norm.weight", "kv_norm.weight"),
    ("ape", "position_bias"),
)
_INDEXER_PAIRS = (
    ("wq_b.weight", "q_b_proj.weight"),
    ("weights_proj.weight", "weights_proj.weight"),
)


def _layer_pairs(layer_idx: int, layer_type: str) -> list[tuple[str, str]]:
    """Map one layer's published tensor names to native module names."""
    p = f"layers.{layer_idx}"
    ops = [(f"{p}.{raw}", f"{p}.{native}") for raw, native in _LAYER_PAIRS]
    if layer_type in ("compressed_sparse_attention", "heavily_compressed_attention"):
        ops.extend(
            (f"{p}.attn.compressor.{raw}", f"{p}.self_attn.compressor.{native}")
            for raw, native in _COMPRESSOR_PAIRS
        )
    if layer_type == "compressed_sparse_attention":
        ops.extend(
            (f"{p}.attn.indexer.{raw}", f"{p}.self_attn.compressor.indexer.{native}")
            for raw, native in _INDEXER_PAIRS
        )
        ops.extend(
            (
                f"{p}.attn.indexer.compressor.{raw}",
                f"{p}.self_attn.compressor.indexer.compressor.{native}",
            )
            for raw, native in _COMPRESSOR_PAIRS
        )
    return ops


_BASE = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_scale": "model.hc_head.hc_scale",
}
_RAW_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
_NATIVE_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_proj|down_proj|up_proj)$"
)
_PROJECTIONS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _mapping(keys: list[str]) -> dict[str, str]:
    layers = {
        int(match.group(1))
        for key in keys
        if (match := re.match(r"^(?:model\.)?layers\.(\d+)\.", key))
    }
    pairs = dict(_BASE)
    for layer in layers:
        pairs.update(
            {
                raw: "model." + local
                for raw, local in _layer_pairs(layer, "compressed_sparse_attention")
            }
        )
    return pairs


def raw_to_wavelet(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    raw_state = dict(state)
    dequantize_state_dict_(raw_state)
    mapping = _mapping(list(raw_state))
    result: dict[str, Tensor] = {}
    buckets: dict[tuple[int, str], dict[int, Tensor]] = defaultdict(dict)
    for key, tensor in raw_state.items():
        if key.startswith("mtp."):
            continue
        match = _RAW_EXPERT.fullmatch(key)
        if match:
            layer, expert, projection = match.groups()
            buckets[int(layer), projection][int(expert)] = tensor
        elif key in mapping:
            result[mapping[key]] = tensor
        else:
            raise ValueError(f"Unrecognized DeepSeek-V4 raw checkpoint key: {key}")
    for (layer, projection), experts in buckets.items():
        if sorted(experts) != list(range(len(experts))):
            raise ValueError("DeepSeek-V4 expert indices must be contiguous from zero.")
        result[f"model.layers.{layer}.mlp.experts.{_PROJECTIONS[projection]}"] = (
            torch.stack([experts[index] for index in range(len(experts))])
        )
    return result


def wavelet_to_raw(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    mapping = {native: raw for raw, native in _mapping(list(state)).items()}
    projections = {value: key for key, value in _PROJECTIONS.items()}
    result = {}
    for key, tensor in state.items():
        match = _NATIVE_EXPERT.fullmatch(key)
        if match:
            layer, projection = match.groups()
            if tensor.ndim != 3:
                raise ValueError(
                    "DeepSeek-V4 packed expert weights must have three dimensions."
                )
            for expert, weight in enumerate(tensor.unbind(0)):
                result[
                    f"layers.{layer}.ffn.experts.{expert}.{projections[projection]}.weight"
                ] = weight
        elif key in mapping:
            result[mapping[key]] = tensor
        else:
            raise ValueError(f"Unrecognized DeepSeek-V4 native checkpoint key: {key}")
    return result


def convert_checkpoint(source: Path, target: Path, *, to_raw: bool = False) -> None:
    """Convert a local safetensors checkpoint; the full weights must fit in CPU RAM.

    Destination must not exist. Tokenizer assets should be supplied separately.
    Raw quantized input is dequantized; output never advertises quantized weights.
    """
    import json

    from safetensors.torch import load_file, save_file

    from .configuration_deepseek_v4 import DeepseekV4Config

    if target.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {target}")
    config = DeepseekV4Config.from_pretrained(source, local_files_only=True)
    index = source / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    state: dict[str, Tensor] = {}
    for name in names:
        path = source / name
        if path.resolve().parent != source.resolve():
            raise ValueError("Checkpoint shards must be direct children of the source.")
        shard = load_file(path, device="cpu")
        duplicate = state.keys() & shard.keys()
        if duplicate:
            raise ValueError(f"Duplicate checkpoint tensor keys: {sorted(duplicate)}")
        state.update(shard)
    converted = wavelet_to_raw(state) if to_raw else raw_to_wavelet(state)
    if not converted:
        raise ValueError("Checkpoint has no model weights.")
    from .modeling import DeepseekV4ForCausalLM

    with torch.device("meta"):
        expected = DeepseekV4ForCausalLM(config).state_dict()
    native = state if to_raw else converted
    missing = expected.keys() - native.keys()
    unexpected = native.keys() - expected.keys()
    wrong_shapes = [
        key
        for key in expected.keys() & native.keys()
        if expected[key].shape != native[key].shape
    ]
    if missing or unexpected or wrong_shapes:
        raise ValueError(
            f"Invalid DeepSeek-V4 checkpoint: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, wrong_shapes={sorted(wrong_shapes)}"
        )
    target.mkdir(parents=True)
    save_file(
        {key: value.contiguous() for key, value in converted.items()},
        target / "model.safetensors",
    )
    config.wavelet_checkpoint_format = None if to_raw else "deepseek_v4_native_v1"
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config.architectures = ["DeepseekV4ForCausalLM"]
    config.save_pretrained(target)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert local DeepSeek-V4 safetensors checkpoints."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--to-raw", action="store_true")
    arguments = parser.parse_args()
    convert_checkpoint(arguments.source, arguments.target, to_raw=arguments.to_raw)

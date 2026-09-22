import pytest
import torch

from wavelet.trainer.models.deepseek_v4.conversion import (
    _layer_pairs,
    raw_to_wavelet,
    wavelet_to_raw,
)


def _fixture() -> dict[str, torch.Tensor]:
    keys = ["embed.weight", "head.weight", "norm.weight", "hc_head_fn"]
    for layer in range(3):
        keys.extend(raw for raw, _ in _layer_pairs(layer, "compressed_sparse_attention"))
        keys.extend(raw for raw, _ in _layer_pairs(layer, "heavily_compressed_attention"))
        keys.extend(raw for raw, _ in _layer_pairs(layer, "sliding_window_attention"))
        keys.extend(f"layers.{layer}.ffn.experts.{expert}.w{projection}.weight" for expert in range(2) for projection in (1, 2, 3))
    return {key: torch.randn(2, 2) for key in dict.fromkeys(keys)}


def test_all_attention_layer_key_families_and_expert_stack_roundtrip() -> None:
    raw = _fixture()
    native = raw_to_wavelet(raw)
    restored = wavelet_to_raw(native)
    for key, value in raw.items():
        if key.startswith("mtp."):
            continue
        torch.testing.assert_close(restored[key], value)


def test_hash_mhc_compressor_indexer_and_unknown_keys_are_preserved_or_rejected() -> None:
    raw = _fixture()
    raw["layers.0.ffn.gate.tid2eid"] = torch.zeros(4, 2, dtype=torch.long)
    raw["layers.0.attn.compressor.ape"] = torch.randn(2)
    raw["layers.0.hc_attn_fn"] = torch.randn(2)
    native = raw_to_wavelet(raw)
    assert "model.layers.0.mlp.router.tid2eid" in native
    assert "model.layers.0.self_attn.compressor.position_bias" in native
    assert "model.layers.0.attn_hc.fn" in native
    with pytest.raises(ValueError, match="Unrecognized"):
        raw_to_wavelet({"layers.0.unknown.weight": torch.ones(1)})


def test_mtp_is_intentionally_omitted() -> None:
    raw = _fixture() | {"mtp.layers.0.weight": torch.ones(1)}
    assert all("mtp" not in key for key in raw_to_wavelet(raw))


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantized_raw_weight_requires_scale(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="missing its scale"):
        raw_to_wavelet({"layers.0.attn.wkv.weight": torch.zeros(2, 2).to(dtype)})

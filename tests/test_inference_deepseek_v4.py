import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wavelet.inference.deepseek_v4 import (
    _deepseek_v4_rope_parameters,
    monkey_patch_deepseek_v4_bf16_o_proj,
    monkey_patch_deepseek_v4_per_layer_rope,
)


def _install_vllm_stubs(monkeypatch, original):
    vllm = types.ModuleType("vllm")
    models = types.ModuleType("vllm.models")
    dsv4 = types.ModuleType("vllm.models.deepseek_v4")
    nvidia = types.ModuleType("vllm.models.deepseek_v4.nvidia")
    flashinfer = types.ModuleType("vllm.models.deepseek_v4.nvidia.flashinfer_sparse")
    flashmla = types.ModuleType("vllm.models.deepseek_v4.nvidia.flashmla")
    ops = types.ModuleType("vllm.models.deepseek_v4.nvidia.ops")
    o_proj = types.ModuleType("vllm.models.deepseek_v4.nvidia.ops.o_proj")
    rope = types.ModuleType("vllm.models.deepseek_v4.common.rope")
    common = types.ModuleType("vllm.models.deepseek_v4.common")
    attention = types.ModuleType("vllm.models.deepseek_v4.attention")
    o_proj.deep_gemm_fp8_o_proj = original
    rope.build_deepseek_v4_rope = original
    flashinfer.deep_gemm_fp8_o_proj = original
    flashmla.deep_gemm_fp8_o_proj = original
    ops.o_proj = o_proj
    nvidia.flashinfer_sparse = flashinfer
    nvidia.flashmla = flashmla
    dsv4.nvidia = nvidia
    common.rope = rope
    dsv4.common = common
    dsv4.attention = attention
    models.deepseek_v4 = dsv4
    vllm.models = models
    for module in (
        vllm,
        models,
        dsv4,
        nvidia,
        flashinfer,
        flashmla,
        ops,
        o_proj,
        common,
        rope,
        attention,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return o_proj, flashinfer, flashmla, rope, attention


def test_rope_parameters_support_flat_and_nested_without_mutation():
    flat = {"rope_type": "yarn", "factor": 3}
    original = dict(flat)
    result = _deepseek_v4_rope_parameters(
        flat, compress_ratio=1, max_position_embeddings=128
    )
    assert result["rope_type"] == "yarn"
    assert flat == original

    nested = {
        "main": {"rope_type": "default", "factor": 2},
        "compress": {"rope_type": "yarn", "factor": 4},
    }
    before = {key: dict(value) for key, value in nested.items()}
    result = _deepseek_v4_rope_parameters(
        nested, compress_ratio=2, max_position_embeddings=128
    )
    assert result == nested["compress"]
    assert nested == before


def test_rope_patch_is_idempotent_and_restores_shared_config(monkeypatch):
    def original(*args, **kwargs):
        return args, kwargs

    _, _, _, rope, attention = _install_vllm_stubs(monkeypatch, original)
    config = SimpleNamespace(
        rope_parameters={
            "main": {"rope_type": "default"},
            "compress": {"rope_type": "yarn", "factor": 2},
        }
    )
    monkey_patch_deepseek_v4_per_layer_rope()
    patched = rope.build_deepseek_v4_rope
    monkey_patch_deepseek_v4_per_layer_rope()
    assert rope.build_deepseek_v4_rope is patched
    patched(
        config,
        head_dim=8,
        rope_head_dim=4,
        max_position_embeddings=32,
        compress_ratio=2,
    )
    assert config.rope_parameters["compress"]["factor"] == 2
    assert attention.build_deepseek_v4_rope is patched


def test_bf16_o_proj_matches_inverse_rope_grouped_contraction(monkeypatch):
    def original(*args, **kwargs):
        raise AssertionError("FP8 path should not be called")

    o_proj, flashinfer, flashmla, _rope, _attention = _install_vllm_stubs(
        monkeypatch, original
    )
    monkey_patch_deepseek_v4_bf16_o_proj()
    patched = o_proj.deep_gemm_fp8_o_proj
    monkey_patch_deepseek_v4_bf16_o_proj()
    assert o_proj.deep_gemm_fp8_o_proj is patched

    torch.manual_seed(3)
    o = torch.randn(2, 2, 4, dtype=torch.bfloat16)
    cache = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    wo_a = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    wo_b = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1])
    actual = patched(
        o,
        positions,
        cache,
        wo_a,
        wo_b,
        n_groups=2,
        heads_per_group=1,
        rope_dim=2,
        o_lora_rank=2,
    )

    rotated = o[:, :, 2:].float().reshape(2, 2, 1, 2)
    cos = cache[positions, :1].view(2, 1, 1, 1)
    sin = cache[positions, 1:].view(2, 1, 1, 1)
    expected_o = o.clone()
    expected_o[:, :, 2:] = (
        torch.cat(
            (
                rotated[..., :1] * cos + rotated[..., 1:] * sin,
                rotated[..., 1:] * cos - rotated[..., :1] * sin,
            ),
            dim=-1,
        )
        .reshape(2, 2, 2)
        .to(o.dtype)
    )
    x = expected_o.reshape(2, 2, 4).reshape(2, 2, 4)
    weight = wo_a.weight.view(2, 2, 4)
    expected = wo_b(torch.einsum("tgr,gdr->tgd", x, weight).flatten(1))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert flashinfer.deep_gemm_fp8_o_proj is patched
    assert flashmla.deep_gemm_fp8_o_proj is patched


def test_rope_patch_restores_config_after_builder_failure(monkeypatch) -> None:
    def fail(config, **kwargs):
        assert config.rope_parameters["factor"] == 1
        assert "main" not in config.rope_parameters
        config.rope_parameters["mutation"] = True
        raise RuntimeError("builder failed")

    _, _, _, rope, _ = _install_vllm_stubs(monkeypatch, fail)
    parameters = {
        "main": {"rope_type": "default"},
        "compress": {"rope_type": "yarn", "factor": 4},
    }
    config = SimpleNamespace(rope_parameters=parameters)
    monkey_patch_deepseek_v4_per_layer_rope()
    with pytest.raises(RuntimeError, match="builder failed"):
        rope.build_deepseek_v4_rope(
            config,
            head_dim=8,
            rope_head_dim=4,
            max_position_embeddings=32,
            compress_ratio=1,
        )
    assert config.rope_parameters is parameters
    assert "mutation" not in parameters["main"]


def test_o_proj_preserves_fp8_kernel_delegation(monkeypatch) -> None:
    calls = []
    sentinel = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    module, _, _, _, _ = _install_vllm_stubs(monkeypatch, original)
    monkey_patch_deepseek_v4_bf16_o_proj()
    projection = SimpleNamespace(weight=torch.zeros(2, 2).to(torch.float8_e4m3fn))
    inputs = (object(), object(), object(), projection, object())
    result = module.deep_gemm_fp8_o_proj(
        *inputs, n_groups=2, heads_per_group=1, rope_dim=2, o_lora_rank=2
    )
    assert result is sentinel
    assert calls == [
        (inputs, {"n_groups": 2, "heads_per_group": 1, "rope_dim": 2, "o_lora_rank": 2})
    ]

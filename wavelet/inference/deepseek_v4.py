"""DeepSeek-V4 vLLM compatibility, adapted from Apache-2.0 PrimeRL.

See wavelet/trainer/models/deepseek_v4/LICENSE for the source license.
"""

from __future__ import annotations

import torch

_DEEPSEEK_V4_YARN_ROPE_TYPES = frozenset(
    {"yarn", "deepseek_yarn", "deepseek_llama_scaling"}
)


def monkey_patch_deepseek_v4_bf16_o_proj() -> None:
    """Serve unquantized DeepSeek-V4 grouped output projections without FP8 scales."""
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse, flashmla
    from vllm.models.deepseek_v4.nvidia.ops import o_proj as o_proj_module

    original_o_proj = o_proj_module.deep_gemm_fp8_o_proj
    if getattr(original_o_proj, "_wavelet_has_bf16_fallback", False):
        return

    def _inverse_rope(o, positions, cos_sin_cache, rope_dim):
        num_tokens, num_heads, head_dim = o.shape
        half_rope = rope_dim // 2
        cos_sin = cos_sin_cache[positions].float()
        cos = cos_sin[:, :half_rope].view(num_tokens, 1, half_rope)
        sin = cos_sin[:, half_rope:].view(num_tokens, 1, half_rope)
        rotated = (
            o[:, :, head_dim - rope_dim :]
            .float()
            .reshape(num_tokens, num_heads, half_rope, 2)
        )
        even, odd = (rotated[..., 0], rotated[..., 1])
        pairs = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
        out = o.clone()
        out[:, :, head_dim - rope_dim :] = pairs.reshape(
            num_tokens, num_heads, rope_dim
        ).to(o.dtype)
        return out

    def _patched_o_proj(
        o, positions, cos_sin_cache, wo_a, wo_b, *, n_groups, heads_per_group, **kwargs
    ):
        if wo_a.weight.dtype == torch.float8_e4m3fn:
            return original_o_proj(
                o,
                positions,
                cos_sin_cache,
                wo_a,
                wo_b,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                **kwargs,
            )
        x = _inverse_rope(o, positions, cos_sin_cache, kwargs["rope_dim"])
        x = x.reshape(o.shape[0], n_groups, heads_per_group * o.shape[-1])
        weight = wo_a.weight.view(n_groups, kwargs["o_lora_rank"], -1)
        z = torch.einsum("tgr,gdr->tgd", x, weight)
        return wo_b(z.flatten(1))

    _patched_o_proj._wavelet_has_bf16_fallback = True
    o_proj_module.deep_gemm_fp8_o_proj = _patched_o_proj
    flashmla.deep_gemm_fp8_o_proj = _patched_o_proj
    flashinfer_sparse.deep_gemm_fp8_o_proj = _patched_o_proj


def _deepseek_v4_rope_parameters(
    rope_parameters, *, compress_ratio, max_position_embeddings
):
    """Resolve per-layer RoPE from flat or main/compress checkpoint parameters."""
    rope_parameters = rope_parameters if isinstance(rope_parameters, dict) else {}
    is_sliding = compress_ratio <= 1
    if any(isinstance(value, dict) for value in rope_parameters.values()):
        label = "main" if is_sliding else "compress"
        branch = rope_parameters.get(label)
        if not isinstance(branch, dict):
            raise ValueError(
                f"DeepSeek V4 rope_parameters is nested by rope type but has no {label!r} sub-dict; found {sorted(rope_parameters)}."
            )
        parameters = dict(branch)
    else:
        parameters = dict(rope_parameters)
    rope_type = parameters.get("rope_type", parameters.get("type", "default"))
    if is_sliding or rope_type not in _DEEPSEEK_V4_YARN_ROPE_TYPES:
        rope_type = "yarn"
        parameters["factor"] = 1
        parameters["original_max_position_embeddings"] = max_position_embeddings
    parameters["rope_type"] = rope_type
    return parameters


def monkey_patch_deepseek_v4_per_layer_rope() -> None:
    """Preserve per-layer RoPE scaling and avoid mutating shared config."""
    from vllm.models.deepseek_v4.common import rope as dsv4_rope

    original_build = dsv4_rope.build_deepseek_v4_rope
    if getattr(original_build, "_wavelet_matches_deepseek_reference", False):
        return

    def _patched_build(
        config, *, head_dim, rope_head_dim, max_position_embeddings, compress_ratio
    ):
        original_parameters = config.rope_parameters
        config.rope_parameters = _deepseek_v4_rope_parameters(
            original_parameters,
            compress_ratio=compress_ratio,
            max_position_embeddings=max_position_embeddings,
        )
        try:
            return original_build(
                config,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                compress_ratio=compress_ratio,
            )
        finally:
            config.rope_parameters = original_parameters

    _patched_build._wavelet_matches_deepseek_reference = True
    dsv4_rope.build_deepseek_v4_rope = _patched_build
    from vllm.models.deepseek_v4 import attention as dsv4_attention

    dsv4_attention.build_deepseek_v4_rope = _patched_build

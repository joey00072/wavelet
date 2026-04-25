from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024
_INT32_MAX = 2**31
_INT32_SAFE = _INT32_MAX - BLOCK_SIZE * 4


@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr, LONG: tl.constexpr):
    idx = tl.program_id(0)
    if LONG:
        offs = idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offs = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    e_val = tl.load(e + offs, mask=mask, other=0).to(tl.float32)
    g_val = tl.load(g + offs, mask=mask, other=0)

    f_val = (e_val * tl.sigmoid(e_val)).to(g_val.dtype)
    tl.store(h + offs, f_val * g_val, mask=mask)


def swiglu_fg_kernel(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Forward: h = silu(e) * g"""
    n = e.numel()
    h = torch.empty_like(e)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _fg_kernel[grid](e, g, h, n, BLOCK_SIZE=BLOCK_SIZE, LONG=int(n > _INT32_SAFE))
    return h


@triton.jit
def _DWf_DW_dfg_kernel(
    DW, e, g, n_elements, BLOCK_SIZE: tl.constexpr, LONG: tl.constexpr
):
    """In-place backward: overwrites DW←h, e←df, g←de."""
    idx = tl.program_id(0)
    if LONG:
        offs = idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offs = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    DW_val = tl.load(DW + offs, mask=mask, other=0)
    e_val = tl.load(e + offs, mask=mask, other=0).to(tl.float32)
    g_val = tl.load(g + offs, mask=mask, other=0)

    se = tl.sigmoid(e_val)
    f = (se * e_val).to(DW_val.dtype)
    h = f * g_val
    df = DW_val * f
    de = (
        DW_val.to(tl.float32) * g_val.to(tl.float32) * se * (1.0 + e_val * (1.0 - se))
    ).to(DW_val.dtype)

    tl.store(DW + offs, h, mask=mask)
    tl.store(e + offs, df, mask=mask)
    tl.store(g + offs, de, mask=mask)


def swiglu_DWf_DW_dfg_kernel(
    DW: torch.Tensor, e: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward: in-place update DW→h, e→df, g→de."""
    n = e.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _DWf_DW_dfg_kernel[grid](
        DW, e, g, n, BLOCK_SIZE=BLOCK_SIZE, LONG=int(n > _INT32_SAFE)
    )
    return DW, e, g

"""
Each class is a torch.autograd.Function that:
  1. Dequantizes 4-bit weights on the fly (never materialises the full bf16 tensor
     in Python — the dequant lives only inside the kernel invocation).
  2. Fuses the base matmul + LoRA adapter (A @ B * s) in one pass.
  3. Does NOT save intermediate activations for backward — they are recomputed
     from the saved inputs and weights, reducing peak memory.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel
from .utils import (
    fast_dequantize,
    get_lora_parameters,
    matmul_lora,
)


def _transpose_adapters(adapters: tuple, dtype: torch.dtype) -> tuple:
    """Cast/transpose LoRA weights for backward; None marks an inactive adapter."""
    return tuple(None if t is None else t.to(dtype).t() for t in adapters)


def _lora_adapter_grads(
    X: Tensor,
    dY: Tensor,
    A_t: Tensor | None,
    B_t: Tensor | None,
    S: float | None,
) -> tuple[Tensor | None, Tensor | None]:
    """Return (dA, dB) in PEFT layout, or (None, None) when no adapter is active."""
    if A_t is None or B_t is None:
        return None, None
    d_A, d_B = torch.empty_like(A_t), torch.empty_like(B_t)
    d_A.addmm_(X.t(), dY @ B_t.t(), alpha=S, beta=0)
    d_B.addmm_(A_t.t() @ X.t(), dY, alpha=S, beta=0)
    return d_A.t(), d_B.t()


def _add_lora_input_grad(
    dX: Tensor,
    dY: Tensor,
    A_t: Tensor | None,
    B_t: Tensor | None,
    S: float | None,
) -> None:
    if A_t is None or B_t is None:
        return
    dX.addmm_(dY @ B_t.t(), A_t.t(), alpha=S)


# ── LoRA_MLP ─────────────────────────────────────────────────────────────────


class LoRA_MLP(torch.autograd.Function):
    """Fused SwiGLU MLP + three LoRA adapter pairs (gate, up, down).

    Forward  : e = X@G+XAg@Bg*gs,  g = X@U+XAu@Bu*us,  h = silu(e)*g,  out = h@W+h@Ad@Bd*ds
    Backward : hand-written; e,g,h are recomputed — not saved for backward.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        X: Tensor,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        _fwd_fn,
        _bwd_fn,
        inplace=True,
    ):
        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
        h = _fwd_fn(e, g)
        out = matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateW,
            gateW_quant,
            gateS,
            upW,
            upW_quant,
            upS,
            downW,
            downW_quant,
            downS,
            _bwd_fn,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        ctx.inplace = inplace
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dY: Tensor):
        (
            gateW,
            gateW_quant,
            gateS,
            upW,
            upW_quant,
            upS,
            downW,
            downW_quant,
            downS,
            _bwd_fn,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors

        input_shape = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = _transpose_adapters(
            (gateA, gateB, upA, upB, downA, downB), dtype
        )

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, df, de = _bwd_fn(DW, e, g)  # DW→h, e→df, g→de  (in-place)
        h = DW

        d_downA, d_downB = _lora_adapter_grads(h, dY, downA, downB, downS)
        d_upA, d_upB = _lora_adapter_grads(X, df, upA, upB, upS)
        d_gateA, d_gateB = _lora_adapter_grads(X, de, gateA, gateB, gateS)

        upW_fp = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW_fp.t(), out=X if ctx.inplace else None)
        del upW_fp
        _add_lora_input_grad(dX, df, upA, upB, upS)

        gateW_fp = fast_dequantize(gateW.t(), gateW_quant)
        dX.addmm_(de, gateW_fp.t())
        del gateW_fp
        _add_lora_input_grad(dX, de, gateA, gateB, gateS)

        return (
            dX.view(input_shape),
            None,
            None,
            d_gateA,
            d_gateB,
            None,
            None,
            None,
            d_upA,
            d_upB,
            None,
            None,
            None,
            d_downA,
            d_downB,
            None,
            None,
            None,
            None,  # _fwd_fn, _bwd_fn, inplace
        )


# ── LoRA_QKV ─────────────────────────────────────────────────────────────────


class LoRA_QKV(torch.autograd.Function):
    """Fused QKV projection + three LoRA adapter pairs."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        X: Tensor,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace=True,
    ):
        orig = X.shape
        Xf = X.view(-1, X.shape[-1]) if X.dim() == 3 else X
        Q = matmul_lora(Xf, QW, QW_quant, QA, QB, QS)
        K = matmul_lora(Xf, KW, KW_quant, KA, KB, KS)
        V = matmul_lora(Xf, VW, VW_quant, VA, VB, VS)
        if len(orig) == 3:
            Q = Q.view(orig[0], orig[1], -1)
            K = K.view(orig[0], orig[1], -1)
            V = V.view(orig[0], orig[1], -1)

        ctx.custom_saved_tensors = (
            QW,
            QW_quant,
            QS,
            KW,
            KW_quant,
            KS,
            VW,
            VW_quant,
            VS,
        )
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB)
        ctx.inplace = inplace
        return Q, K, V

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = ctx.custom_saved_tensors
        X, QA, QB, KA, KB, VA, VB = ctx.saved_tensors

        input_shape = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])
        dV = dV.view(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype

        QA, QB, KA, KB, VA, VB = _transpose_adapters((QA, QB, KA, KB, VA, VB), dtype)

        d_QA, d_QB = _lora_adapter_grads(X, dQ, QA, QB, QS)
        d_KA, d_KB = _lora_adapter_grads(X, dK, KA, KB, KS)
        d_VA, d_VB = _lora_adapter_grads(X, dV, VA, VB, VS)

        QW_fp = fast_dequantize(QW.t(), QW_quant)
        dX = torch.matmul(dQ, QW_fp.t(), out=X if ctx.inplace else None)
        del QW_fp
        _add_lora_input_grad(dX, dQ, QA, QB, QS)

        KW_fp = fast_dequantize(KW.t(), KW_quant)
        dX.addmm_(dK, KW_fp.t())
        del KW_fp
        _add_lora_input_grad(dX, dK, KA, KB, KS)

        VW_fp = fast_dequantize(VW.t(), VW_quant)
        dX.addmm_(dV, VW_fp.t())
        del VW_fp
        _add_lora_input_grad(dX, dV, VA, VB, VS)

        return (
            dX.view(input_shape),
            None,
            None,
            d_QA,
            d_QB,
            None,
            None,
            None,
            d_KA,
            d_KB,
            None,
            None,
            None,
            d_VA,
            d_VB,
            None,
            None,  # inplace
        )


# ── LoRA_W ───────────────────────────────────────────────────────────────────


class LoRA_W(torch.autograd.Function):
    """Fused single linear projection + LoRA (e.g. o_proj)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, X: Tensor, W, W_quant, A, B, S):
        out = matmul_lora(X, W, W_quant, A, B, S)
        ctx.custom_saved_tensors = (W, W_quant, S)
        ctx.save_for_backward(A, B, X)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dY: Tensor):
        W, W_quant, S = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        input_shape = X.shape
        dY = dY.reshape(-1, dY.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        dtype = X.dtype

        A_t, B_t = _transpose_adapters((A, B), dtype)
        d_A, d_B = _lora_adapter_grads(X, dY, A_t, B_t, S)

        W_fp = fast_dequantize(W.t(), W_quant)
        dX = dY @ W_fp.t()
        del W_fp
        _add_lora_input_grad(dX, dY, A_t, B_t, S)

        return dX.view(input_shape), None, None, d_A, d_B, None


# ── apply_* helpers (used as module.forward replacements) ────────────────────


def apply_lora_mlp_swiglu(self, X: Tensor, inplace: bool = True) -> Tensor:
    """Drop-in replacement for Qwen3MLP / LigerSwiGLUMLP forward."""
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    return LoRA_MLP.apply(
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        swiglu_fg_kernel,
        swiglu_DWf_DW_dfg_kernel,
        inplace,
    )


def apply_lora_qkv(self, X: Tensor, inplace: bool = True):
    """Return (Q, K, V) using fused LoRA_QKV kernel."""
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    return LoRA_QKV.apply(
        X,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace,
    )

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

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = (
            t.to(dtype).t() for t in (gateA, gateB, upA, upB, downA, downB)
        )

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, df, de = _bwd_fn(DW, e, g)  # DW→h, e→df, g→de  (in-place)
        h = DW

        d_downA, d_downB = torch.empty_like(downA), torch.empty_like(downB)
        d_gateA, d_gateB = torch.empty_like(gateA), torch.empty_like(gateB)
        d_upA, d_upB = torch.empty_like(upA), torch.empty_like(upB)

        d_downA.addmm_(h.t(), dY @ downB.t(), alpha=downS, beta=0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha=downS, beta=0)
        d_upA.addmm_(X.t(), df @ upB.t(), alpha=upS, beta=0)
        d_upB.addmm_(upA.t() @ X.t(), df, alpha=upS, beta=0)
        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha=gateS, beta=0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha=gateS, beta=0)

        upW_fp = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW_fp.t(), out=X if ctx.inplace else None)
        del upW_fp
        dX.addmm_(df @ upB.t(), upA.t(), alpha=upS)

        gateW_fp = fast_dequantize(gateW.t(), gateW_quant)
        dX.addmm_(de, gateW_fp.t())
        del gateW_fp
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha=gateS)

        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            d_gateA.t(),
            d_gateB.t(),
            None,
            None,
            None,
            d_upA.t(),
            d_upB.t(),
            None,
            None,
            None,
            d_downA.t(),
            d_downB.t(),
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

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])
        dV = dV.view(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype

        QA, QB, KA, KB, VA, VB = (t.to(dtype).t() for t in (QA, QB, KA, KB, VA, VB))

        d_QA, d_QB = torch.empty_like(QA), torch.empty_like(QB)
        d_KA, d_KB = torch.empty_like(KA), torch.empty_like(KB)
        d_VA, d_VB = torch.empty_like(VA), torch.empty_like(VB)

        d_QA.addmm_(X.t(), dQ @ QB.t(), alpha=QS, beta=0)
        d_QB.addmm_(QA.t() @ X.t(), dQ, alpha=QS, beta=0)
        d_KA.addmm_(X.t(), dK @ KB.t(), alpha=KS, beta=0)
        d_KB.addmm_(KA.t() @ X.t(), dK, alpha=KS, beta=0)
        d_VA.addmm_(X.t(), dV @ VB.t(), alpha=VS, beta=0)
        d_VB.addmm_(VA.t() @ X.t(), dV, alpha=VS, beta=0)

        QW_fp = fast_dequantize(QW.t(), QW_quant)
        dX = torch.matmul(dQ, QW_fp.t(), out=X if ctx.inplace else None)
        del QW_fp
        dX.addmm_(dQ @ QB.t(), QA.t(), alpha=QS)

        KW_fp = fast_dequantize(KW.t(), KW_quant)
        dX.addmm_(dK, KW_fp.t())
        del KW_fp
        dX.addmm_(dK @ KB.t(), KA.t(), alpha=KS)

        VW_fp = fast_dequantize(VW.t(), VW_quant)
        dX.addmm_(dV, VW_fp.t())
        del VW_fp
        dX.addmm_(dV @ VB.t(), VA.t(), alpha=VS)

        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            d_QA.t(),
            d_QB.t(),
            None,
            None,
            None,
            d_KA.t(),
            d_KB.t(),
            None,
            None,
            None,
            d_VA.t(),
            d_VB.t(),
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

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        dtype = X.dtype

        A_t, B_t = A.to(dtype).t(), B.to(dtype).t()
        d_A, d_B = torch.empty_like(A_t), torch.empty_like(B_t)
        d_A.addmm_(X.t(), dY @ B_t.t(), alpha=S, beta=0)
        d_B.addmm_(A_t.t() @ X.t(), dY, alpha=S, beta=0)

        W_fp = fast_dequantize(W.t(), W_quant)
        dX = dY @ W_fp.t()
        del W_fp
        dX.addmm_(dY @ B_t.t(), A_t.t(), alpha=S)

        return dX.view(batch, seq_len, hd), None, None, d_A.t(), d_B.t(), None


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


def apply_lora_o(self, X: Tensor) -> Tensor:
    """Apply fused LoRA_W kernel to o_proj."""
    OW, OW_quant, OA, OB, OS = get_lora_parameters(self.o_proj)
    return LoRA_W.apply(X, OW, OW_quant, OA, OB, OS)

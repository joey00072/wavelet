from __future__ import annotations

import torch
from torch import Tensor


@torch.inference_mode()
def fast_dequantize(W: Tensor, quant_state=None, out: Tensor | None = None) -> Tensor:
    """Dequantize a bitsandbytes NF4 weight tensor to bf16/fp16.

    bitsandbytes stores 4-bit weights as (n_elements, 1); the transposed view
    is (1, n_elements).  dequantize_4bit detects this via W.shape[0]==1 and
    returns the result with axes swapped, i.e.:
      dequantize(W)    → quant_state.shape       e.g. (d_out, d_in)
      dequantize(W.t()) → quant_state.shape.T    e.g. (d_in, d_out)
    No additional transposition is needed here.
    """
    if quant_state is None:
        return W
    import bitsandbytes.functional as bnb_F

    return bnb_F.dequantize_4bit(W, quant_state, out=out)


def matmul_lora(
    X: Tensor,
    W: Tensor,
    W_quant,
    A: Tensor | None,
    B: Tensor | None,
    s: float,
    out: Tensor | None = None,
) -> Tensor:
    """Matrix multiply X @ (W_dequant + A @ B * s), efficient for 4-bit LoRA.

    X : (*, d_in)
    W : NF4 4-bit weight  shape (d_out/2, d_in)  [bitsandbytes storage]
    A : LoRA A weight     shape (rank, d_in)
    B : LoRA B weight     shape (d_out, rank)
    """
    reshape = X.dim() == 3
    if reshape:
        batch, seq_len, d = X.shape
        X = X.view(-1, d)

    W_fp = fast_dequantize(W, W_quant)
    out = torch.matmul(X, W_fp.t(), out=out)
    if W_quant is not None:
        del W_fp

    if A is not None:
        dtype = X.dtype
        A_t, B_t = A.t(), B.t()
        out.addmm_(X @ A_t.to(dtype), B_t.to(dtype), alpha=s)

    return out.view(batch, seq_len, -1) if reshape else out


def get_lora_parameters(proj) -> tuple:
    """Extract (W, W_quant, A, B, scaling) from a PEFT-wrapped 4-bit linear.

    Returns (W, W_quant, None, None, None) if LoRA adapters are not active.
    """
    base_layer = getattr(proj, "base_layer", proj)
    W = base_layer.weight
    W_quant = getattr(W, "quant_state", None)

    if getattr(proj, "disable_adapters", True) or getattr(proj, "merged", False):
        return W, W_quant, None, None, None

    adapters = getattr(proj, "active_adapters", None)
    if callable(adapters):
        adapters = adapters()
    if adapters is None:
        adapters = getattr(proj, "active_adapter", ("default",))
    adapters = list(adapters)
    if len(adapters) != 1:
        raise RuntimeError(
            "Wavelet fused LoRA kernels support exactly one active adapter; "
            f"found {adapters}."
        )
    adapter = adapters[0]

    A = proj.lora_A[adapter].weight
    B = proj.lora_B[adapter].weight
    return W, W_quant, A, B, proj.scaling[adapter]

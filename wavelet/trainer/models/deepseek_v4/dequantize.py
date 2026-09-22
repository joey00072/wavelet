"""DeepSeek-V4 FP8 and packed MXFP4 checkpoint dequantization.

Adapted from PrimeRL's Apache-2.0 DeepSeek-V4 preprocessing implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor

FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _unpack_mxfp4(packed: Tensor) -> Tensor:
    lut = FP4_E2M1_LUT.to(packed.device)
    u8 = packed.contiguous().view(torch.uint8)
    low = (u8 & 0xF).long()
    high = ((u8 >> 4) & 0xF).long()
    return torch.stack([lut[low], lut[high]], -1).reshape(
        *packed.shape[:-1], 2 * packed.shape[-1]
    )


def dequantize_weight(weight: Tensor, scale: Tensor) -> Tensor:
    if weight.dtype == torch.int8:
        values = _unpack_mxfp4(weight)
    elif weight.dtype == torch.float8_e4m3fn:
        values = weight.float()
    else:
        raise ValueError(f"Unsupported quantized weight dtype: {weight.dtype}")
    rows, cols = values.shape[-2:]
    scale_rows, scale_cols = scale.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(
            f"Weight shape {tuple(values.shape[-2:])} not divisible by scale grid {tuple(scale.shape[-2:])}"
        )
    expanded = (
        scale.float()
        .repeat_interleave(rows // scale_rows, -2)
        .repeat_interleave(cols // scale_cols, -1)
    )
    return (values * expanded).bfloat16()


def dequantize_state_dict_(state_dict: dict[str, Tensor]) -> None:
    for key in [key for key in state_dict if key.endswith(".weight")]:
        scale_key = key.removesuffix(".weight") + ".scale"
        if (
            state_dict[key].dtype in {torch.int8, torch.float8_e4m3fn}
            and scale_key not in state_dict
        ):
            raise ValueError(f"Quantized weight {key} is missing its scale tensor.")
        scale = state_dict.pop(scale_key, None)
        if scale is not None:
            state_dict[key] = dequantize_weight(state_dict[key], scale)

"""Wire-format shaping for checkpoint tensors moving through weight transport."""

from __future__ import annotations

from torch import Tensor, nn

from wavelet.utils.modules import strip_training_wrapper_segments


def _convert_layer_to_hf(
    model: nn.Module,
    state_dict: dict[str, Tensor],
    layer_index: int,
) -> dict[str, Tensor]:
    """Convert one trainer layer to the checkpoint names vLLM consumes."""
    state_dict = {
        strip_training_wrapper_segments(name): tensor
        for name, tensor in state_dict.items()
    }
    convert_layer = getattr(model, "convert_layer_to_hf", None)
    if callable(convert_layer):
        converted = convert_layer(state_dict, layer_index)
        return state_dict if converted is None else converted
    try:
        from transformers.core_model_loading import revert_weight_conversion
    except ImportError:
        return state_dict
    return revert_weight_conversion(model, state_dict)

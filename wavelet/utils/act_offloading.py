from __future__ import annotations

import logging
from contextlib import nullcontext

import psutil
import torch
from torch.autograd.graph import saved_tensors_hooks

from wavelet.configs.sft import ActivationOffloadingConfig


class OffloadActivations(saved_tensors_hooks):
    """Offload large saved activations to CPU during forward.

    This keeps the implementation intentionally conservative:
    - single-stream only
    - only offload non-parameter tensors
    - only offload tensors above a small size threshold
    """

    def __init__(
        self,
        *,
        use_pin_memory: bool = True,
        min_offload_size: int = 1024,
    ) -> None:
        if not torch.cuda.is_available():
            raise ValueError("Activation offloading requires a CUDA device.")

        self.use_pin_memory = use_pin_memory
        self.min_tensor_size_bytes = min_offload_size
        self.virtual_memory_safe_pct = 60
        self._tracker: dict[int, tuple[torch.Tensor, bool]] = {}
        self._tensor_id = 0
        self._is_first_forward_call = True
        self._is_first_backward_call = True
        self._warned_on_ram = False
        super().__init__(self._pack_tensor, self._unpack_tensor)

    def _next_tensor_id(self) -> int:
        self._tensor_id += 1
        return self._tensor_id

    def _verify_virtual_memory(self) -> None:
        if self._warned_on_ram:
            return
        current_pct = psutil.virtual_memory().percent
        if current_pct > self.virtual_memory_safe_pct:
            logging.getLogger(__name__).warning(
                "CPU memory usage is high during activation offloading: %s%% > %s%%",
                current_pct,
                self.virtual_memory_safe_pct,
            )
        self._warned_on_ram = True

    def _pack_tensor(self, activation: torch.Tensor) -> int:
        if self._is_first_forward_call:
            assert not self._tracker, (
                "Activation tracker should be empty at the start of a forward pass."
            )
            self._is_first_forward_call = False
            self._is_first_backward_call = True

        tensor_id = self._next_tensor_id()
        num_bytes = activation.element_size() * activation.nelement()
        should_offload = (
            activation.device.type == "cuda"
            and num_bytes >= self.min_tensor_size_bytes
            and not isinstance(activation, torch.nn.Parameter)
            and not (
                hasattr(torch.nn, "Buffer") and isinstance(activation, torch.nn.Buffer)
            )
        )
        if should_offload:
            cpu_tensor = torch.empty_like(
                activation,
                pin_memory=self.use_pin_memory,
                device="cpu",
            )
            cpu_tensor.copy_(activation, non_blocking=True)
            self._tracker[tensor_id] = (cpu_tensor, True)
        else:
            self._tracker[tensor_id] = (activation, False)
        return tensor_id

    def _unpack_tensor(self, tensor_id: int) -> torch.Tensor:
        if self._is_first_backward_call:
            self._is_first_backward_call = False
            self._is_first_forward_call = True
            if self.use_pin_memory:
                self._verify_virtual_memory()
        if tensor_id not in self._tracker:
            raise KeyError(f"Untracked activation tensor id: {tensor_id}")
        tensor, offloaded = self._tracker.pop(tensor_id)
        return tensor.to("cuda", non_blocking=True) if offloaded else tensor


def maybe_activation_offloading(
    config: ActivationOffloadingConfig | None,
) -> OffloadActivations | nullcontext:
    if config is None:
        return nullcontext()
    return OffloadActivations(use_pin_memory=config.pin_memory)

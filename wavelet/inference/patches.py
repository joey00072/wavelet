from __future__ import annotations

import torch


def transformers_v5_compat() -> None:
    """vLLM general plugin for compatibility and DP pause/resume fixes."""
    _patch_qwen3_vl_config_attrs()
    monkey_patch_fp32_lm_head()
    monkey_patch_dp_engine_core_pause_resume_deadlock()


def _patch_qwen3_vl_config_attrs() -> None:
    try:
        from transformers import Qwen3VLMoeTextConfig
    except ImportError:
        return

    if not hasattr(Qwen3VLMoeTextConfig, "tie_word_embeddings"):
        Qwen3VLMoeTextConfig.tie_word_embeddings = False


def _fp32_lm_head_logits(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    embedding_bias: torch.Tensor | None,
) -> torch.Tensor:
    original_shape = hidden_states.shape[:-1]
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = torch.mm(flat, weight.t(), out_dtype=torch.float32)
    if embedding_bias is not None:
        logits = logits + embedding_bias.to(torch.float32)
    if hidden_states.dim() > 2:
        logits = logits.reshape(*original_shape, -1)
    return logits


def monkey_patch_fp32_lm_head() -> None:
    """Let vLLM opt into a native half-precision-to-fp32 LM-head GEMM."""
    try:
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
    except ImportError:
        return

    if getattr(LogitsProcessor, "_wavelet_fp32_lm_head_patch", False):
        return

    original_init = LogitsProcessor.__init__
    original_get_logits = LogitsProcessor._get_logits

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        vllm_config = get_current_vllm_config()
        additional_config = vllm_config.additional_config or {}
        self._wavelet_fp32_lm_head = bool(additional_config.get("fp32_lm_head", False))

    def patched_get_logits(  # type: ignore[no-untyped-def]
        self,
        hidden_states,
        lm_head,
        embedding_bias,
    ):
        if not getattr(self, "_wavelet_fp32_lm_head", False):
            return original_get_logits(self, hidden_states, lm_head, embedding_bias)
        logits = _fp32_lm_head_logits(
            hidden_states,
            lm_head.weight,
            embedding_bias,
        )
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    LogitsProcessor.__init__ = patched_init
    LogitsProcessor._get_logits = patched_get_logits
    LogitsProcessor._wavelet_fp32_lm_head_patch = True


def _patched_add_request(base_add_request, pause_state, engine_core_outputs):
    def patched(self, request, request_wave: int = 0):
        base_add_request(self, request, request_wave)
        if not self.has_coordinator or request_wave == self.current_wave:
            return
        if request_wave > self.current_wave:
            self.current_wave = request_wave
        elif not self.engines_running and self.scheduler.pause_state == pause_state:
            self.engines_running = True
            self.output_queue.put_nowait(
                (-1, engine_core_outputs(start_wave=self.current_wave))
            )

    return patched


def _patched_client_request(base_handler, start_wave_type, pause_state):
    def patched(self, request_type, request):
        if request_type != start_wave_type:
            return base_handler(self, request_type, request)
        new_wave, exclude_engine_index = request
        if exclude_engine_index == self.engine_index or new_wave < self.current_wave:
            return None
        self.current_wave = new_wave
        if not self.engines_running and self.scheduler.pause_state == pause_state:
            self.engines_running = True
        return None

    return patched


def _patched_resume_scheduler(base_resume_scheduler, pause_state):
    def patched(self):
        was_paused = self.scheduler.pause_state != pause_state
        base_resume_scheduler(self)
        if was_paused:
            self.engines_running = True
            self._force_dp_running_state_sync = True

    return patched


def _patched_unfinished_requests(parallel_config):
    def patched(self, local_unfinished: bool) -> bool:
        self.step_counter += 1
        if getattr(self, "_force_dp_running_state_sync", False):
            self._force_dp_running_state_sync = False
            return parallel_config.has_unfinished_dp(self.dp_group, local_unfinished)
        if self.step_counter % 32 != 0:
            return True
        return parallel_config.has_unfinished_dp(self.dp_group, local_unfinished)

    return patched


def monkey_patch_dp_engine_core_pause_resume_deadlock() -> None:
    """Avoid vLLM data-parallel deadlocks while pausing for weight updates."""
    try:
        from vllm.config import ParallelConfig
        from vllm.v1.core.sched.interface import PauseState
        from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
        from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc
    except ImportError:
        return

    if getattr(DPEngineCoreProc, "_wavelet_pause_resume_patch", False):
        return

    base_add_request = EngineCore.add_request
    base_handle_client_request = EngineCoreProc._handle_client_request
    base_resume_scheduler = DPEngineCoreProc.resume_scheduler

    DPEngineCoreProc.add_request = _patched_add_request(
        base_add_request,
        PauseState.UNPAUSED,
        EngineCoreOutputs,
    )
    DPEngineCoreProc._handle_client_request = _patched_client_request(
        base_handle_client_request,
        EngineCoreRequestType.START_DP_WAVE,
        PauseState.UNPAUSED,
    )
    DPEngineCoreProc.resume_scheduler = _patched_resume_scheduler(
        base_resume_scheduler,
        PauseState.UNPAUSED,
    )
    DPEngineCoreProc._has_global_unfinished_reqs = _patched_unfinished_requests(
        ParallelConfig
    )
    DPEngineCoreProc._wavelet_pause_resume_patch = True

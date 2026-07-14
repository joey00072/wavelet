from __future__ import annotations


def transformers_v5_compat() -> None:
    """vLLM general plugin for compatibility and DP pause/resume fixes."""
    _patch_qwen3_vl_config_attrs()
    monkey_patch_dp_engine_core_pause_resume_deadlock()


def _patch_qwen3_vl_config_attrs() -> None:
    try:
        from transformers import Qwen3VLMoeTextConfig
    except ImportError:
        return

    if not hasattr(Qwen3VLMoeTextConfig, "tie_word_embeddings"):
        Qwen3VLMoeTextConfig.tie_word_embeddings = False


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

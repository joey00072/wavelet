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


def monkey_patch_dp_engine_core_pause_resume_deadlock() -> None:
    """Avoid vLLM data-parallel deadlocks while pausing for weight updates."""
    try:
        from vllm.config import ParallelConfig
        from vllm.v1.core.sched.interface import PauseState
        from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
        from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc
        from vllm.v1.request import Request
    except ImportError:
        return

    if getattr(DPEngineCoreProc, "_wavelet_pause_resume_patch", False):
        return

    base_add_request = EngineCore.add_request
    base_handle_client_request = EngineCoreProc._handle_client_request
    base_resume_scheduler = DPEngineCoreProc.resume_scheduler

    def patched_add_request(self, request: Request, request_wave: int = 0):
        base_add_request(self, request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                self.engines_running = True
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    def patched_handle_client_request(self, request_type, request):
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and new_wave >= self.current_wave:
                self.current_wave = new_wave
                if (
                    not self.engines_running
                    and self.scheduler.pause_state == PauseState.UNPAUSED
                ):
                    self.engines_running = True
        else:
            base_handle_client_request(self, request_type, request)

    def patched_resume_scheduler(self):
        was_paused = self.scheduler.pause_state != PauseState.UNPAUSED
        base_resume_scheduler(self)
        if was_paused:
            self.engines_running = True
            self._force_dp_running_state_sync = True

    def patched_has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        self.step_counter += 1
        if getattr(self, "_force_dp_running_state_sync", False):
            self._force_dp_running_state_sync = False
            return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)
        if self.step_counter % 32 != 0:
            return True
        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    DPEngineCoreProc.add_request = patched_add_request
    DPEngineCoreProc._handle_client_request = patched_handle_client_request
    DPEngineCoreProc.resume_scheduler = patched_resume_scheduler
    DPEngineCoreProc._has_global_unfinished_reqs = patched_has_global_unfinished_reqs
    DPEngineCoreProc._wavelet_pause_resume_patch = True

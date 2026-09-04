from __future__ import annotations

from pathlib import Path

import torch


class StepProfiler:
    """Capture one Chrome trace across an inclusive optimizer-step range."""

    def __init__(
        self,
        trace_path: Path,
        *,
        start_step: int,
        end_step: int,
        record_shapes: bool,
        profile_memory: bool,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.start_step = start_step
        self.end_step = end_step
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self._profiler: torch.profiler.profile | None = None
        self._closed = False

    def before_step(self, step: int) -> None:
        if self._closed or self._profiler is not None:
            return
        if not self.start_step <= step <= self.end_step:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            acc_events=True,
        )
        self._profiler.__enter__()

    def after_step(self, step: int) -> None:
        if self._profiler is not None and step >= self.end_step:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        self._profiler.export_chrome_trace(str(self.trace_path))

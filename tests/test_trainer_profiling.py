from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
import torch

from wavelet.configs.config import TrainerConfig
from wavelet.trainer.distributed import World
from wavelet.trainer.profiling import StepProfiler
from wavelet.trainer.trainer import BaseTrainer


def test_step_profiler_exports_requested_cpu_range(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "trace.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    profiler = StepProfiler(
        trace_path,
        start_step=2,
        end_step=3,
        record_shapes=True,
        profile_memory=True,
    )

    profiler.before_step(1)
    assert not trace_path.exists()

    profiler.before_step(2)
    value = torch.ones((2, 2))
    torch.add(value, value)
    profiler.after_step(2)
    assert not trace_path.exists()

    profiler.after_step(3)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["traceEvents"]


def test_profiler_step_range_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="end_step"):
        TrainerConfig(profiler={"start_step": 3, "end_step": 2})


def test_distributed_trainer_uses_rank_specific_trace_path(
    tmp_path,
    monkeypatch,
) -> None:
    trainer = BaseTrainer(
        TrainerConfig(
            output_dir=tmp_path,
            profiler={"start_step": 2, "end_step": 4},
        )
    )
    trainer.world = World(
        rank=1,
        local_rank=1,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
    )
    profiler = Mock()
    create_profiler = Mock(return_value=profiler)
    monkeypatch.setattr("wavelet.trainer.trainer.StepProfiler", create_profiler)

    trainer._maybe_start_step_profiler(2)

    create_profiler.assert_called_once_with(
        tmp_path / "profiler" / "trace-2-4.rank-1.json",
        start_step=2,
        end_step=4,
        record_shapes=True,
        profile_memory=True,
    )
    profiler.before_step.assert_called_once_with(2)

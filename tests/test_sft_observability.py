from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from wavelet.configs.sft import SFTConfig
from wavelet.trainer import trainer as trainer_module
from wavelet.trainer.distributed import World
from wavelet.trainer.trainer import SFTTrainer, _dataloader_progress
from wavelet.trainer.types import LossOutput, TrainOutput


class _Loader:
    def __init__(self, state: dict[str, object]) -> None:
        self._state = state

    def state_dict(self) -> dict[str, object]:
        return self._state


def _cpu_world(*, world_size: int = 1) -> World:
    return World(
        rank=0,
        local_rank=0,
        world_size=world_size,
        local_world_size=world_size,
        device=torch.device("cpu"),
    )


def _worker_state() -> dict[str, object]:
    return {
        "_snapshot": {
            "_worker_snapshots": {
                "worker_0": {
                    "dataset_state": {
                        "step": 8,
                        "epoch": 1,
                        "num_samples": {"alpha": 1},
                        "num_tokens": {"alpha": 10},
                    }
                },
                "worker_1": {
                    "dataset_state": {
                        "dataset": {
                            "step": 12,
                            "epoch": 2,
                            "num_samples": {"beta": 3},
                            "num_tokens": {"beta": 10},
                        },
                        "pending": {},
                    }
                },
            }
        }
    }


def test_dataloader_progress_aggregates_worker_and_packed_dataset_state() -> None:
    progress = _dataloader_progress(
        _Loader(_worker_state()),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )

    assert progress == {
        "step": 12,
        "epoch": 2,
        "num_samples": {"alpha": 1, "beta": 3},
        "num_tokens": {"alpha": 10, "beta": 10},
    }


def test_sft_log_includes_performance_and_source_progress() -> None:
    trainer = SFTTrainer(SFTConfig())
    trainer.world = _cpu_world()
    trainer.dataset = object()  # type: ignore[assignment]
    trainer.dataloader = _Loader(_worker_state())
    trainer.monitor = Mock()
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 0.25}])
    trainer.step = 3
    progress = Mock()
    output = TrainOutput(
        loss=LossOutput(loss=torch.tensor(1.5)),
        stepped=True,
        step=3,
        micro_step=3,
        metrics={
            "perf/tokens_per_second": 128.0,
            "perf/peak_memory_gib": 4.0,
        },
    )

    trainer._log_train_output(output, progress)

    metrics, step = trainer.monitor.log.call_args.args
    assert step == 3
    assert metrics["perf/tokens_per_second"] == pytest.approx(128.0)
    assert metrics["perf/peak_memory_gib"] == pytest.approx(4.0)
    assert metrics["progress/epoch"] == pytest.approx(2.0)
    assert metrics["progress/alpha/ratio_samples"] == pytest.approx(0.25)
    assert metrics["progress/beta/ratio_samples"] == pytest.approx(0.75)
    assert metrics["progress/alpha/ratio_tokens"] == pytest.approx(0.5)
    assert metrics["progress/beta/ratio_tokens"] == pytest.approx(0.5)


def test_sft_performance_counts_global_data_parallel_tokens(monkeypatch) -> None:
    trainer = SFTTrainer(SFTConfig())
    trainer.world = _cpu_world(world_size=2)
    trainer._step_started_at = 10.0
    trainer._step_model_tokens = 5
    monkeypatch.setattr(trainer_module.time, "perf_counter", lambda: 12.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    metrics = trainer._finish_step_performance_metrics()

    assert metrics["perf/tokens_per_second"] == pytest.approx(5.0)
    assert metrics["perf/peak_memory_gib"] == pytest.approx(0.0)
    assert trainer._step_started_at is None
    assert trainer._step_model_tokens == 0

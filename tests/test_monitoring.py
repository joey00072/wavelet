from __future__ import annotations

import sys
import types

from wavelet.configs.sft import WandbConfig
from wavelet.utils.monitoring import RunMonitor


class _FakeRun:
    def define_metric(self, *_args, **_kwargs) -> None:
        return None

    def log(self, *_args, **_kwargs) -> None:
        return None

    def finish(self) -> None:
        return None


def test_wandb_group_and_tags_are_forwarded(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return _FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))

    monitor = RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
        wandb=WandbConfig(
            enabled=True,
            project="wavelet-tests",
            entity="test-entity",
            name="test-run",
            group="test-group",
            tags=["rl", "multi-gpu"],
            mode="offline",
        ),
    )
    monitor.start_run(run_config={"case": "wandb"})

    assert captured["project"] == "wavelet-tests"
    assert captured["entity"] == "test-entity"
    assert captured["name"] == "test-run"
    assert captured["group"] == "test-group"
    assert captured["tags"] == ["rl", "multi-gpu"]


def test_wandb_alias_metrics_include_lr() -> None:
    aliases = RunMonitor._wandb_alias_metrics({"lr": 1e-6})

    assert aliases["train/lr"] == 1e-6
    assert aliases["scheduler/lr"] == 1e-6

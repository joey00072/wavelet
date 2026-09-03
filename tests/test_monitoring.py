from __future__ import annotations

import sys
import types
from collections import namedtuple

from wavelet.configs.sft import WandbConfig
from wavelet.monitor import read_jsonl
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


def test_disk_metrics_cover_run_and_checkpoint_volumes(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "uncreated" / "checkpoints"
    run_dir.mkdir()
    usage = namedtuple("usage", "total used free")
    calls = []

    def fake_disk_usage(path):
        calls.append(path)
        if path == run_dir:
            return usage(100, 60, 40)
        assert path == tmp_path
        return usage(200, 150, 50)

    monkeypatch.setattr("wavelet.monitor.shutil.disk_usage", fake_disk_usage)
    monitor = RunMonitor(
        run_dir,
        checkpoint_dir=checkpoint_dir,
        log_cuda_memory=False,
    )

    metrics = monitor._resource_metrics()

    assert calls == [run_dir, tmp_path]
    assert metrics["disk_free_bytes"] == 40
    assert metrics["disk_free_ratio"] == 0.4
    assert metrics["checkpoint_disk_free_bytes"] == 50
    assert metrics["checkpoint_disk_free_ratio"] == 0.25


def test_sample_history_compacts_to_recent_rows(tmp_path) -> None:
    monitor = RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
        sample_history_size=3,
    )
    monitor.start_run()

    monitor.log_samples([{"completion": "zero"}, {"completion": "one"}], step=0)
    monitor.log_samples([{"completion": "two"}, {"completion": "three"}], step=1)
    monitor.log_samples([{"completion": "four"}], step=2)

    rows = read_jsonl(tmp_path / "samples.jsonl")
    assert [row["completion"] for row in rows] == ["two", "three", "four"]
    assert [row["step"] for row in rows] == [1, 1, 2]


def test_sample_history_compacts_existing_file_on_first_write(tmp_path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        "".join(f'{{"completion": "{index}"}}\n' for index in range(5)),
        encoding="utf-8",
    )
    monitor = RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
        sample_history_size=3,
    )
    monitor.start_run(resumed_from="checkpoint-1")

    monitor.log_samples([{"completion": "new"}], step=2)

    rows = read_jsonl(samples_path)
    assert [row["completion"] for row in rows] == ["3", "4", "new"]

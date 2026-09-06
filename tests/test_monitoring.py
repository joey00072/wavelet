from __future__ import annotations

import json
import logging
import sys
import types
from collections import namedtuple

from wavelet import wandb_overview
from wavelet.configs.sft import WandbConfig
from wavelet.monitor import RunMonitor, read_jsonl


class _FakeRun:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.metric_definitions: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def define_metric(self, *args, **kwargs) -> None:
        self.metric_definitions.append((args, kwargs))

    def log(self, row, *_args, **_kwargs) -> None:
        self.rows.append(row)

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
    monitor.start_run(run_config={"case": "wandb", "nested": {"api_token": "secret"}})

    assert captured["project"] == "wavelet-tests"
    assert captured["entity"] == "test-entity"
    assert captured["name"] == "test-run"
    assert captured["group"] == "test-group"
    assert captured["tags"] == ["rl", "multi-gpu"]
    assert captured["config"] == {
        "case": "wandb",
        "nested": {"api_token": "<redacted>"},
    }
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["nested"]["api_token"] == "<redacted>"


def test_online_primary_creates_curated_wandb_overview(monkeypatch, tmp_path) -> None:
    created: dict[str, object] = {}
    run = _FakeRun()
    run.id = "online-123"
    run.entity = "test-entity"
    run.project = "wavelet-tests"

    monkeypatch.delenv("WANDB_SHARED_MODE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(
            init=lambda **_kwargs: run,
            Settings=lambda **kwargs: kwargs,
        ),
    )

    def fake_ensure(entity, project, **kwargs):
        created.update({"entity": entity, "project": project, **kwargs})
        return "https://wandb.example/overview"

    monkeypatch.setattr(wandb_overview, "ensure_overview_view", fake_ensure)
    monitor = RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
        wandb=WandbConfig(enabled=True, mode="online"),
    )

    monitor.start_run(
        run_config={
            "orchestrator": {"verifier_env_id": "reverse-text@1"},
            "eval": {"env": [{"id": "aime2025"}]},
        }
    )

    assert created == {
        "entity": "test-entity",
        "project": "wavelet-tests",
        "flavor": "rl",
        "train_envs": ["reverse-text"],
        "eval_envs": ["aime2025"],
    }


def test_wandb_alias_metrics_include_lr() -> None:
    aliases = RunMonitor._wandb_alias_metrics({"lr": 1e-6})

    assert aliases["train/lr"] == 1e-6
    assert aliases["scheduler/lr"] == 1e-6


def test_nonfinite_metrics_are_null_locally_and_omitted_from_wandb(
    caplog, tmp_path
) -> None:
    monitor = RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
    )
    wandb_run = _FakeRun()
    monitor._wandb_run = wandb_run

    with caplog.at_level(logging.WARNING, logger="wavelet.monitor"):
        monitor.log(
            {"loss": float("nan"), "scale": float("inf"), "finite": 1.5},
            step=1,
        )
        monitor.log({"loss": float("-inf"), "finite": 2.5}, step=2)

    rows = read_jsonl(tmp_path / "metrics.jsonl")
    assert rows[0]["loss"] is None
    assert rows[0]["scale"] is None
    assert rows[0]["finite"] == 1.5
    assert rows[1]["loss"] is None
    assert "loss" not in wandb_run.rows[0]
    assert "train/loss" not in wandb_run.rows[0]
    assert "scale" not in wandb_run.rows[0]
    assert wandb_run.rows[0]["finite"] == 1.5
    assert "loss" not in wandb_run.rows[1]
    warnings = [
        record.message
        for record in caplog.records
        if record.message.startswith("Replacing non-finite metric")
    ]
    assert len(warnings) == 2
    assert any("'loss'" in message for message in warnings)
    assert any("'scale'" in message for message in warnings)


def test_step_less_metrics_use_wandb_wall_time(monkeypatch, tmp_path) -> None:
    monitor = RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
    )
    wandb_run = _FakeRun()
    monitor._wandb_run = wandb_run
    monkeypatch.setattr("wavelet.monitor.time.time", lambda: 123.5)

    monitor.log({"inference/queue_depth": 4.0}, step=None)
    monitor.log({"inference/throughput": 2.0}, step=None)

    rows = read_jsonl(tmp_path / "metrics.jsonl")
    assert rows[0]["step"] is None
    assert wandb_run.rows[0] == {
        "inference/queue_depth": 4.0,
        "_timestamp": 123.5,
    }
    assert wandb_run.metric_definitions == [
        (("inference/*",), {"step_metric": "_timestamp"})
    ]


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


def test_heartbeat_carries_per_rank_table_without_growing_metrics(tmp_path) -> None:
    monitor = RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
        write_metrics_csv=False,
    )
    monitor.start_run(run_config={})
    ranks = [{"rank": 0, "node": "host-a", "tokens_per_second": 120.0}]

    monitor.log({"loss": 1.0, "node/host-a/tokens_per_second": 120.0}, 3, ranks=ranks)
    monitor.log({"loss": 0.5}, 4)

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["step"] == 4
    assert "ranks" not in heartbeat
    monitor.log({"loss": 0.4}, 5, ranks=ranks)
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["ranks"] == ranks
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert all("ranks" not in row for row in rows)
    assert rows[0]["node/host-a/tokens_per_second"] == 120.0

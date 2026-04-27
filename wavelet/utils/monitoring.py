from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from wavelet.configs.sft import WandbConfig
from wavelet.distributed.world import World, get_world


class RunMonitor:
    def __init__(
        self,
        output_dir: Path,
        *,
        enabled: bool = True,
        write_events: bool = True,
        write_metrics_jsonl: bool = True,
        write_metrics_csv: bool = True,
        write_run_metadata: bool = True,
        write_heartbeat: bool = True,
        log_cuda_memory: bool = True,
        log_disk_usage: bool = True,
        wandb: WandbConfig | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.enabled = enabled
        self.write_events = write_events
        self.write_metrics_jsonl = write_metrics_jsonl
        self.write_metrics_csv = write_metrics_csv
        self.write_run_metadata = write_run_metadata
        self.write_heartbeat = write_heartbeat
        self.log_cuda_memory = log_cuda_memory
        self.log_disk_usage = log_disk_usage
        self.wandb = wandb or WandbConfig()
        self.metrics_file = output_dir / "metrics.jsonl"
        self.csv_file = output_dir / "metrics.csv"
        self.samples_file = output_dir / "samples.jsonl"
        self.events_file = output_dir / "events.jsonl"
        self.heartbeat_file = output_dir / "heartbeat.json"
        self.run_metadata_file = output_dir / "run_metadata.json"
        self._csv_headers: list[str] = []
        self._wandb_run: Any | None = None
        self._wandb_samples_table: Any | None = None
        self._wandb_samples_columns: list[str] = []

    def start_run(
        self,
        *,
        run_config: dict[str, Any] | None = None,
        world: World | None = None,
        resumed_from: str | None = None,
    ) -> None:
        if not self._should_write():
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        world = world or get_world()

        if self.write_run_metadata:
            metadata = {
                "started_at": self._timestamp(),
                "pid": os.getpid(),
                "output_dir": str(self.output_dir),
                "world": {
                    "rank": world.rank,
                    "local_rank": world.local_rank,
                    "world_size": world.world_size,
                    "local_world_size": world.local_world_size,
                    "device": str(world.device),
                },
                "resumed_from": resumed_from,
                "config": run_config,
            }
            self.run_metadata_file.write_text(json.dumps(metadata, indent=2))

        self._init_wandb(run_config, resumed_from)
        event = "run_resumed" if resumed_from is not None else "run_started"
        self.log_event(event, payload={"resumed_from": resumed_from})
        self._write_heartbeat(status="running", step=None)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self._should_write():
            return

        row = dict(metrics)
        row["step"] = step
        row["timestamp"] = self._timestamp()
        row.update(self._resource_metrics())

        if self.write_metrics_jsonl:
            with self.metrics_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

        if self.write_metrics_csv:
            self._append_csv_row(row)

        if self._wandb_run is not None:
            wandb_metrics = {k: v for k, v in row.items() if k != "timestamp"}
            wandb_metrics.update(self._wandb_alias_metrics(row))
            self._wandb_run.log(wandb_metrics, step=step)

        self._write_heartbeat(status="running", step=step, metrics=row)

    def log_event(
        self,
        name: str,
        *,
        step: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self._should_write() or not self.write_events:
            return
        record = {
            "timestamp": self._timestamp(),
            "event": name,
            "step": step,
            "payload": payload or {},
        }
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def log_samples(self, samples: list[dict[str, Any]], step: int) -> None:
        if not self._should_write() or not samples:
            return

        timestamp = self._timestamp()
        with self.samples_file.open("a", encoding="utf-8") as handle:
            for sample in samples:
                row = {"timestamp": timestamp, "step": step, **sample}
                handle.write(json.dumps(row) + "\n")

        if self._wandb_run is None:
            return

        columns = ["step", *samples[0].keys()]
        if self._wandb_samples_table is None:
            import wandb

            self._wandb_samples_columns = columns
            self._wandb_samples_table = wandb.Table(
                columns=columns,
                log_mode="INCREMENTAL",
            )
        if columns != self._wandb_samples_columns:
            raise ValueError("Sample log columns changed during the run.")

        for sample in samples:
            self._wandb_samples_table.add_data(
                *[step if column == "step" else sample[column] for column in columns]
            )
        self._wandb_run.log({"samples": self._wandb_samples_table}, step=step)

    def finish(
        self,
        *,
        status: str,
        step: int | None,
    ) -> None:
        if not self._should_write():
            return
        self.log_event("run_finished", step=step, payload={"status": status})
        self._write_heartbeat(status=status, step=step)
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def _should_write(self) -> bool:
        if not self.enabled:
            return False
        world = get_world()
        return world.is_main

    def _resource_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if self.log_cuda_memory:
            metrics.update(self._cuda_metrics())
        if self.log_disk_usage:
            metrics.update(self._disk_metrics())
        return metrics

    def _cuda_metrics(self) -> dict[str, Any]:
        fields = {
            "cuda_memory_allocated_bytes": None,
            "cuda_memory_reserved_bytes": None,
            "cuda_max_memory_allocated_bytes": None,
            "cuda_max_memory_reserved_bytes": None,
        }
        if not torch.cuda.is_available():
            return fields
        fields["cuda_memory_allocated_bytes"] = torch.cuda.memory_allocated()
        fields["cuda_memory_reserved_bytes"] = torch.cuda.memory_reserved()
        fields["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        fields["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        return fields

    def _disk_metrics(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.output_dir)
        return {
            "disk_total_bytes": usage.total,
            "disk_used_bytes": usage.used,
            "disk_free_bytes": usage.free,
        }

    def _append_csv_row(self, row: dict[str, Any]) -> None:
        headers = list(row.keys())
        if not self.csv_file.exists():
            self._csv_headers = headers
            with self.csv_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_headers)
                writer.writeheader()
                writer.writerow(row)
            return

        if not self._csv_headers:
            with self.csv_file.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self._csv_headers = list(reader.fieldnames or [])

        new_headers = [key for key in headers if key not in self._csv_headers]
        if new_headers:
            existing_rows: list[dict[str, Any]] = []
            with self.csv_file.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                existing_rows.extend(reader)
            self._csv_headers.extend(new_headers)
            with self.csv_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_headers)
                writer.writeheader()
                for existing in existing_rows:
                    writer.writerow(existing)
                writer.writerow(row)
            return

        with self.csv_file.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_headers)
            writer.writerow(row)

    def _write_heartbeat(
        self,
        *,
        status: str,
        step: int | None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.write_heartbeat:
            return
        payload = {
            "timestamp": self._timestamp(),
            "pid": os.getpid(),
            "status": status,
            "step": step,
        }
        if metrics is not None:
            payload["metrics"] = metrics
        self.heartbeat_file.write_text(json.dumps(payload, indent=2))

    def _init_wandb(
        self,
        run_config: dict[str, Any] | None,
        resumed_from: str | None,
    ) -> None:
        if not self.wandb.enabled or self.wandb.mode == "disabled":
            return
        import wandb

        self._wandb_run = wandb.init(
            project=self.wandb.project or "wavelet",
            entity=self.wandb.entity,
            name=self.wandb.name,
            mode=self.wandb.mode,
            dir=str(self.output_dir),
            config=run_config,
            resume="allow" if resumed_from is not None else None,
        )
        self._wandb_run.define_metric("step")
        self._wandb_run.define_metric("*", step_metric="step")

    @staticmethod
    def _wandb_alias_metrics(row: dict[str, Any]) -> dict[str, Any]:
        aliases: dict[str, Any] = {}
        if "reward_mean" in row:
            aliases["reward"] = row["reward_mean"]
            aliases["train/reward_mean"] = row["reward_mean"]
        if "reward/all/mean" in row:
            aliases["rollout/reward_mean"] = row["reward/all/mean"]
        if "rollout/count" in row:
            aliases["rollout/count"] = row["rollout/count"]
        if "loss" in row:
            aliases["train/loss"] = row["loss"]
        return aliases

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()


def setup_logger(name: str, level: str = "info") -> Any:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)

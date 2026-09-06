"""Deterministic synthetic RL run directories for dashboard development and tests.

The writer produces the same artifact layout a real ``wavelet rl`` process-mode
run leaves behind: trainer and orchestrator metrics, eval metrics and eval
rollout sets, stable rollout batches with manifests and claim/consumed records,
queue lifecycle events, policy directories, heartbeat, run metadata, resolved
config, and role logs. No GPU or model is required.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from wavelet.transport.queue import (
    MANIFEST_FILENAME,
    POLICY_META_FILENAME,
    QUEUE_EVENT_FILENAME,
    STABLE_BATCH_MARKER,
    ClaimRecord,
    ConsumedRecord,
    QueueEvent,
    RolloutManifest,
    get_policy_step_dir,
    get_step_dir,
)

DEFAULT_ENVS = ("equation-builder", "reverse-text")
_PROMPTS = {
    "equation-builder": "Use the numbers {a}, {b}, {c}, and {d} exactly once with + and - to reach {target}.",
    "reverse-text": "Reverse the following text character by character: '{text}'.",
}
_WORDS = [
    "wavelet",
    "policy",
    "rollout",
    "gradient",
    "queue",
    "lattice",
    "prism",
    "orbit",
]


def write_synthetic_run(
    output_dir: Path,
    *,
    steps: int = 24,
    groups: int = 8,
    rollouts_per_group: int = 8,
    seed: int = 7,
    envs: tuple[str, ...] = DEFAULT_ENVS,
    eval_interval: int = 4,
    eval_examples: int = 12,
    eval_rollouts: int = 2,
    status: str = "completed",
    started_at: datetime | None = None,
) -> Path:
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = started_at or datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    clock = _Clock(start)
    max_steps = steps
    _write_config(
        output_dir, envs=envs, max_steps=max_steps, groups=groups, k=rollouts_per_group
    )
    _write_run_metadata(output_dir, start)
    _append(output_dir / "events.jsonl", _run_event("run_started", None, clock.now()))
    queue_events: list[QueueEvent] = []
    _write_policy(output_dir, step=0, clock=clock, events=queue_events)

    for step in range(steps):
        clock.advance(rng.uniform(20.0, 45.0))
        rows = _rollout_rows(
            rng, step=step, envs=envs, groups=groups, k=rollouts_per_group
        )
        _write_rollout_batch(
            output_dir, step=step, rows=rows, clock=clock, events=queue_events
        )
        _append(
            output_dir / "orchestrator_metrics.jsonl",
            _orchestrator_metrics(rng, step=step, rows=rows, envs=envs, clock=clock),
        )
        clock.advance(rng.uniform(4.0, 9.0))
        _write_claim_consume(output_dir, step=step, clock=clock, events=queue_events)
        _append(
            output_dir / "metrics.jsonl",
            _trainer_metrics(rng, step=step, rows=rows, clock=clock),
        )
        _write_policy(output_dir, step=step + 1, clock=clock, events=queue_events)
        if step % 3 == 0:
            _append_samples(output_dir, step=step, rows=rows, clock=clock)
        if eval_interval > 0 and (step + 1) % eval_interval == 0:
            _write_eval(
                output_dir,
                rng,
                step=step + 1,
                envs=envs,
                examples=eval_examples,
                rollouts=eval_rollouts,
                clock=clock,
            )
    if eval_interval > 0:
        _write_eval(
            output_dir,
            rng,
            step=0,
            envs=envs,
            examples=eval_examples,
            rollouts=eval_rollouts,
            clock=_Clock(start + timedelta(seconds=5)),
            baseline=True,
        )
    _write_queue_events(output_dir, queue_events)
    _append(
        output_dir / "events.jsonl",
        _run_event("run_finished", steps, clock.now(), {"status": status}),
    )
    _write_heartbeat(output_dir, status=status, step=steps, clock=clock)
    _write_logs(output_dir, steps=steps, clock=clock)
    return output_dir


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.current = start

    def advance(self, seconds: float) -> datetime:
        self.current += timedelta(seconds=seconds)
        return self.current

    def now(self) -> str:
        return self.current.isoformat()


def _write_jsonl(
    path: Path, rows: Iterable[dict[str, Any]], mode: str = "w", sort_keys: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=sort_keys) + "\n")


def _append(path: Path, row: dict[str, Any]) -> None:
    _write_jsonl(path, [row], mode="a", sort_keys=True)


def _link_latest(parent: Path) -> None:
    if not (latest := parent / "latest").exists():
        latest.symlink_to("attempt_1", target_is_directory=True)


def _run_event(
    name: str, step: int | None, timestamp: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "event": name,
        "step": step,
        "payload": payload or {},
    }


def _write_config(
    output_dir: Path, *, envs: tuple[str, ...], max_steps: int, groups: int, k: int
) -> None:
    config_dir = output_dir / "configs" / "attempt_1" / "resolved"
    config_dir.mkdir(parents=True, exist_ok=True)
    _link_latest(output_dir / "configs")
    config = {
        "model": {"name": "Qwen/Qwen3-0.6B", "torch_dtype": "bfloat16"},
        "data": {"batch_size": groups * k, "micro_batch_size": 1, "seq_len": 2048},
        "loss": {"type": "dppo", "dppo_mask_high": 0.2, "dppo_mask_low": 0.2},
        "algo": {"type": "grpo"},
        "orchestrator": {
            "enabled": True,
            "custom_rollout_function": "wavelet.orchestrator.verifiers:generate_rollouts",
            "verifier_env_id": envs[0],
            "verifier_model": "Qwen/Qwen3-0.6B",
            "examples_per_step": groups,
            "rollouts_per_example": k,
            "max_async_level": 1,
            "max_off_policy_steps": 1,
            "state_server": {"enabled": True, "host": "0.0.0.0", "port": 8765},
        },
        "eval": {
            "interval": 4,
            "num_examples": 12,
            "rollouts_per_example": 2,
            "env": [{"id": env, "name": env} for env in envs],
        },
        "inference": {"enabled": True, "mode": "vllm_http"},
        "policy_transfer": {"type": "filesystem", "export_every_steps": 1},
        "optim": {"type": "adamw", "lr": 3.0e-6},
        "lora": {"rank": 32, "alpha": 64.0},
        "max_steps": max_steps,
        "seed": 42,
        "launcher": {"mode": "process", "backend": "local"},
        "output_dir": str(output_dir),
    }
    (config_dir / "rl_orchestrator.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def _write_run_metadata(output_dir: Path, start: datetime) -> None:
    payload = {
        "started_at": start.isoformat(),
        "pid": 4242,
        "output_dir": str(output_dir),
        "world": {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "local_world_size": 1,
            "device": "cuda:0",
        },
        "resumed_from": None,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2))


SYNTH_NODES = ("synth-a", "synth-b")
_RANKS_PER_NODE = 2


def _node_metrics(rng: random.Random) -> dict[str, float]:
    metrics: dict[str, float] = {"perf/nodes": float(len(SYNTH_NODES))}
    rates: list[float] = []
    for index, node in enumerate(SYNTH_NODES):
        rate = 1900.0 - 150.0 * index + rng.uniform(-120.0, 120.0)
        rates.append(rate)
        metrics.update(
            {
                f"node/{node}/ranks": float(_RANKS_PER_NODE),
                f"node/{node}/tokens": rate * 6.0,
                f"node/{node}/tokens_per_second": rate,
                f"node/{node}/step_seconds": 6.0 + rng.uniform(-0.3, 0.3),
                f"node/{node}/peak_memory_gib": 19.7 - 0.4 * index,
                f"node/{node}/memory_allocated_gib": 15.2 - 0.4 * index,
            }
        )
    metrics["perf/rank_tokens_per_second_min"] = min(rates) / _RANKS_PER_NODE * 0.9
    metrics["perf/rank_tokens_per_second_max"] = max(rates) / _RANKS_PER_NODE * 1.1
    metrics["perf/rank_peak_memory_gib_max"] = 19.7
    return metrics


def _rank_rows(rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rank = 0
    for node_index, node in enumerate(SYNTH_NODES):
        for local_rank in range(_RANKS_PER_NODE):
            tokens = 5700.0 + rng.uniform(-400.0, 400.0)
            rows.append(
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "node": node,
                    "device": f"cuda:{local_rank}",
                    "tokens": tokens,
                    "seconds": 6.0,
                    "tokens_per_second": tokens / 6.0,
                    "memory_allocated_gib": 15.2 - 0.4 * node_index,
                    "peak_memory_gib": 19.7 - 0.4 * node_index,
                }
            )
            rank += 1
    return rows


def _write_heartbeat(
    output_dir: Path, *, status: str, step: int, clock: _Clock
) -> None:
    payload = {
        "timestamp": clock.now(),
        "pid": 4242,
        "status": status,
        "step": step,
        "ranks": _rank_rows(random.Random(step)),
    }
    (output_dir / "heartbeat.json").write_text(json.dumps(payload, indent=2))


def _rollout_rows(
    rng: random.Random,
    *,
    step: int,
    envs: tuple[str, ...],
    groups: int,
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    learning = min(0.85, 0.3 + 0.02 * step)
    for group_index in range(groups):
        env = envs[group_index % len(envs)]
        example_id = f"{env}-{step * groups + group_index:05d}"
        group_key = f"{step}:{group_index}"
        difficulty = rng.random()
        prompt = _prompt(rng, env)
        rewards = [
            1.0
            if rng.random() < learning * (1.2 - difficulty)
            else rng.choice([0.0, 0.0, 0.5])
            for _ in range(k)
        ]
        mean = sum(rewards) / k
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / k)
        for sample_index, reward in enumerate(rewards):
            tokens = rng.randint(40, 700)
            truncated = tokens > 640
            logprobs = [-abs(rng.gauss(0.6, 0.5)) for _ in range(tokens)]
            rows.append(
                {
                    "prompt": [{"role": "user", "content": prompt}],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": _completion(rng, env, reward, tokens),
                        }
                    ],
                    "reward": reward,
                    "advantage": 0.0 if std == 0 else (reward - mean),
                    "input_ids": [rng.randint(0, 150_000) for _ in range(tokens + 60)],
                    "target_ids": [rng.randint(0, 150_000) for _ in range(tokens + 60)],
                    "loss_mask": [False] * 60 + [True] * tokens,
                    "inference_logprobs": logprobs,
                    "temperatures": 1.0,
                    "source": env,
                    "metadata": {
                        "group_key": group_key,
                        "rollout_key": f"{group_key}:{sample_index}",
                        "stop_condition": "generation_truncated"
                        if truncated
                        else "stop",
                        "is_truncated": truncated,
                        "completion_token_count": tokens,
                        "input_token_count": 60,
                        "turn_count": 1,
                        "policy_step": max(step - rng.choice([0, 0, 1]), 0),
                        "task": {"name": env, "example_id": example_id},
                        "rollout": {
                            "group_key": group_key,
                            "rollout_key": f"{group_key}:{sample_index}",
                            "num_turns": 1,
                            "tool_calls": 0,
                            "error": None,
                        },
                    },
                }
            )
    return rows


def _prompt(rng: random.Random, env: str) -> str:
    if env == "equation-builder":
        numbers = [rng.randint(1, 40) for _ in range(4)]
        return _PROMPTS[env].format(
            a=numbers[0],
            b=numbers[1],
            c=numbers[2],
            d=numbers[3],
            target=rng.randint(0, 99),
        )
    text = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 8)))
    return _PROMPTS.get(env, "Solve: {text}").format(text=text)


def _completion(rng: random.Random, env: str, reward: float, tokens: int) -> str:
    verdict = "correct" if reward >= 1.0 else "incorrect"
    body = " ".join(rng.choice(_WORDS) for _ in range(min(tokens // 4, 80)))
    return f"<think>{body}</think>\n\nFinal answer ({env}, {verdict}): {rng.randint(0, 99)}"


def _write_rollout_batch(
    output_dir: Path,
    *,
    step: int,
    rows: list[dict[str, Any]],
    clock: _Clock,
    events: list[QueueEvent],
) -> None:
    step_dir = get_step_dir(output_dir / "rollouts", step)
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / "rollouts.jsonl"
    _write_jsonl(path, rows)
    rewards = [row["reward"] for row in rows]
    policy_step = min(row["metadata"]["policy_step"] for row in rows)
    manifest = RolloutManifest(
        format_version=1,
        queue_step=step,
        optimizer_step=step,
        chunk_index=None,
        policy_step=policy_step,
        rows=len(rows),
        tokens=sum(len(row["input_ids"]) for row in rows),
        reward_mean=sum(rewards) / len(rewards),
        producer_id="rl-inference:synthetic:1",
        created_at=clock.now(),
        payload_bytes=path.stat().st_size,
        transfer_seconds=0.01,
    )
    (step_dir / MANIFEST_FILENAME).write_text(json.dumps(asdict(manifest), indent=2))
    (step_dir / STABLE_BATCH_MARKER).touch()
    events.append(
        QueueEvent(
            time=clock.now(),
            kind="rollout_published",
            queue_step=step,
            optimizer_step=step,
            policy_step=policy_step,
            producer_id=manifest.producer_id,
            details={"payload_bytes": manifest.payload_bytes, "transfer_seconds": 0.01},
        )
    )


def _write_claim_consume(
    output_dir: Path, *, step: int, clock: _Clock, events: list[QueueEvent]
) -> None:
    step_dir = get_step_dir(output_dir / "rollouts", step)
    consumer = "rl-trainer:synthetic:2"
    received = clock.advance(0.2)
    events.append(
        QueueEvent(
            time=received.isoformat(),
            kind="rollout_received",
            queue_step=step,
            optimizer_step=step,
            policy_step=step,
            consumer_id=consumer,
            details={"mode": "wait", "wait_seconds": 30.0},
        )
    )
    claim = ClaimRecord(
        format_version=1,
        queue_step=step,
        consumer_id=consumer,
        trainer_step_before=step,
        claimed_at=clock.advance(0.1).isoformat(),
    )
    (step_dir / "claim.json").write_text(json.dumps(asdict(claim), indent=2))
    events.append(
        QueueEvent(
            time=claim.claimed_at,
            kind="rollout_claimed",
            queue_step=step,
            optimizer_step=step,
            policy_step=step,
            consumer_id=consumer,
        )
    )
    consumed = ConsumedRecord(
        format_version=1,
        queue_step=step,
        consumer_id=consumer,
        trainer_step_before=step,
        trainer_step_after=step + 1,
        optimizer_step_completed=True,
        consumed_at=clock.advance(6.0).isoformat(),
    )
    (step_dir / "consumed.json").write_text(json.dumps(asdict(consumed), indent=2))
    events.append(
        QueueEvent(
            time=consumed.consumed_at,
            kind="rollout_consumed",
            queue_step=step,
            optimizer_step=step,
            policy_step=step,
            consumer_id=consumer,
        )
    )


def _write_policy(
    output_dir: Path, *, step: int, clock: _Clock, events: list[QueueEvent]
) -> None:
    step_dir = get_policy_step_dir(output_dir / "policies", step)
    adapter_dir = step_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}")
    meta = {
        "format_version": 1,
        "step": step,
        "kind": "adapter",
        "created_at": clock.now(),
        "tensor_bytes": 80_792_848,
        "precision": {"trainer_dtype": "bfloat16", "inference_dtype": "bfloat16"},
    }
    (step_dir / POLICY_META_FILENAME).write_text(json.dumps(meta, indent=2))
    (step_dir / STABLE_BATCH_MARKER).touch()
    exported = clock.now()
    events.append(
        QueueEvent(time=exported, kind="policy_export_completed", policy_step=step)
    )
    received = clock.advance(0.3)
    events.append(
        QueueEvent(
            time=received.isoformat(),
            kind="policy_received",
            policy_step=step,
            consumer_id="rl-inference:synthetic:1",
            details={"mode": "wait", "payload_bytes": 80_792_848, "wait_seconds": 6.5},
        )
    )
    loaded = clock.advance(0.05)
    events.append(
        QueueEvent(
            time=loaded.isoformat(),
            kind="policy_load_completed",
            policy_step=step,
            details={"load_seconds": 0.05, "wait_seconds": 6.5},
        )
    )


def _write_queue_events(output_dir: Path, events: list[QueueEvent]) -> None:
    rows = [asdict(event) for event in sorted(events, key=lambda e: e.time)]
    _write_jsonl(output_dir / "events" / QUEUE_EVENT_FILENAME, rows, sort_keys=True)


def _series_metrics(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return {
        f"{prefix}/mean": mean,
        f"{prefix}/std": std,
        f"{prefix}/min": min(values),
        f"{prefix}/max": max(values),
    }


def _orchestrator_metrics(
    rng: random.Random,
    *,
    step: int,
    rows: list[dict[str, Any]],
    envs: tuple[str, ...],
    clock: _Clock,
) -> dict[str, Any]:
    rewards = [row["reward"] for row in rows]
    advantages = [row["advantage"] for row in rows]
    lengths = [float(row["metadata"]["completion_token_count"]) for row in rows]
    truncated = [1.0 if row["metadata"]["is_truncated"] else 0.0 for row in rows]
    lags = [float(step - row["metadata"]["policy_step"]) for row in rows]
    metrics: dict[str, Any] = {
        "timestamp": clock.now(),
        "step": step,
        **_series_metrics(rewards, "reward/all"),
        **_series_metrics(advantages, "advantage/all"),
        **_series_metrics(lengths, "decode_len/all"),
        **_series_metrics(lengths, "seq_len/all"),
        "is_truncated/all/mean": sum(truncated) / len(truncated),
        "fate/all/produced": float(len(rows)),
        "fate/all/trainable": float(len(rows)),
        "fate/all/trainable_rate": 1.0,
        "fate/all/truncated": sum(truncated),
        "fate/all/truncated_rate": sum(truncated) / len(truncated),
        "fate/all/filtered": 0.0,
        "fate/all/errored": 0.0,
        "fate/all/zero_loss": 0.0,
        "generation/reward/mean": sum(rewards) / len(rewards) - 0.05,
        "generation/groups/admitted": float(
            len({r["metadata"]["group_key"] for r in rows})
        ),
        "generation/groups/rejected": float(rng.randint(0, 3)),
        "generation/groups/admission_rate": rng.uniform(0.7, 1.0),
        "generation/solve_all/rate": rng.uniform(0.0, 0.25),
        "generation/solve_none/rate": max(
            0.0, 0.4 - 0.01 * step + rng.uniform(-0.05, 0.05)
        ),
        "generation/concurrency/limit": float(64 + rng.randint(-8, 8)),
        "generation/executor_concurrency": 32.0,
        "generation/data/cursor": float(step * len(rows)),
        "generation/data/epoch": float(step * len(rows) // 4096),
        "off_policy/mean": sum(lags) / len(lags),
        "off_policy/max": max(lags),
        "policy/step": float(step),
        "policy/lag": float(rng.choice([0, 0, 1])),
        "progress/optimizer_step": float(step),
        "progress/queue_step": float(step),
        "progress/samples": float(len(rows)),
        "progress/tokens": float(sum(lengths)) + 60.0 * len(rows),
        "time/generate_completions": rng.uniform(18.0, 40.0),
        "time/publish": rng.uniform(0.01, 0.05),
        "time/step": rng.uniform(20.0, 45.0),
        "time/rollout/generation/mean": rng.uniform(4.0, 12.0),
        "time/rollout/scoring/mean": rng.uniform(0.05, 0.4),
        "inference/replica_0/scrape_success": 1.0,
        "inference/replica_0/kv_cache_usage": rng.uniform(0.3, 0.95),
        "inference/replica_0/requests_running": float(rng.randint(20, 64)),
        "inference/replica_0/requests_waiting": float(rng.randint(0, 12)),
        "inference/replica_0/preemptions_delta": float(rng.choice([0, 0, 0, 1])),
        "inference/replica_0/generation_tokens_per_second": 2400.0
        + rng.uniform(-200.0, 200.0),
        "inference/replica_0/prompt_tokens_per_second": 900.0
        + rng.uniform(-80.0, 80.0),
    }
    for env in envs:
        env_rows = [row for row in rows if row["source"] == env]
        env_rewards = [row["reward"] for row in env_rows]
        metrics.update(_series_metrics(env_rewards, f"reward/{env}"))
        metrics[f"train/{env}/batch_fraction"] = len(env_rows) / len(rows)
        metrics[f"train/{env}/solve_all"] = rng.uniform(0.0, 0.3)
        metrics[f"train/{env}/solve_none"] = rng.uniform(0.0, 0.4)
    return metrics


def _trainer_metrics(
    rng: random.Random, *, step: int, rows: list[dict[str, Any]], clock: _Clock
) -> dict[str, Any]:
    rewards = [row["reward"] for row in rows]
    decay = math.exp(-0.08 * step)
    return {
        "timestamp": clock.now(),
        "step": step,
        "progress/step": step,
        "progress/total_samples": float((step + 1) * len(rows)),
        "progress/total_tokens": float((step + 1) * 30_000),
        "loss": 0.02 * decay + rng.uniform(-0.004, 0.004),
        "train/loss": 0.02 * decay + rng.uniform(-0.004, 0.004),
        "train/policy_loss": 0.04 * decay + rng.uniform(-0.01, 0.01),
        "train/kl_loss": 0.001 + 0.0004 * step + rng.uniform(-0.0002, 0.0002),
        "kl/mismatch": 0.0005 + rng.uniform(0.0, 0.0004),
        "entropy/mean": max(0.05, 0.9 * decay + rng.uniform(-0.03, 0.03)),
        "dppo/is_masked": rng.uniform(0.0, 0.03),
        "dppo/is_masked_low": rng.uniform(0.0, 0.015),
        "dppo/is_masked_high": rng.uniform(0.0, 0.015),
        "optim/grad_norm": 0.2 * decay + rng.uniform(0.01, 0.05),
        "optim/lr": 3.0e-6,
        "reward/all/mean": sum(rewards) / len(rewards),
        "reward/all/min": min(rewards),
        "reward/all/max": max(rewards),
        "advantage/all/std": 0.45 + rng.uniform(-0.05, 0.05),
        "rollout/count": float(len(rows)),
        "tokens/train": float(
            sum(r["metadata"]["completion_token_count"] for r in rows)
        ),
        "tokens/model": float(sum(len(r["input_ids"]) for r in rows)),
        "seq_len/all/mean": sum(len(r["input_ids"]) for r in rows) / len(rows),
        "seq_len/all/max": float(max(len(r["input_ids"]) for r in rows)),
        "micro_batch/count": float(len(rows) // 2),
        "perf/tokens_per_second": 3800.0 + rng.uniform(-300.0, 300.0),
        "perf/peak_memory_gib": 19.7 + rng.uniform(-0.2, 0.2),
        "perf/model_flops_per_token": 2_975_648.0,
        "perf/mfu": rng.uniform(0.18, 0.24),
        **_node_metrics(rng),
        "time/wait_for_batch": rng.uniform(10.0, 35.0),
        "time/load_data": rng.uniform(0.3, 0.9),
        "time/train_until": rng.uniform(4.0, 7.0),
        "time/export_policy": rng.uniform(0.05, 0.2),
        "cuda_memory_allocated_bytes": float(
            15.2 * 2**30 + rng.uniform(-(2**28), 2**28)
        ),
        "cuda_memory_reserved_bytes": float(20.1 * 2**30),
        "cuda_max_memory_allocated_bytes": float(19.7 * 2**30),
        "cuda_max_memory_reserved_bytes": float(20.4 * 2**30),
        "disk_used_bytes": float(400 * 2**30 + step * 2**28),
        "disk_free_bytes": float(600 * 2**30 - step * 2**28),
        "disk_total_bytes": float(1000 * 2**30),
        "disk_free_ratio": 0.6 - step * 0.00025,
        "checkpoint_disk_free_ratio": 0.6 - step * 0.00025,
    }


def _append_samples(
    output_dir: Path, *, step: int, rows: list[dict[str, Any]], clock: _Clock
) -> None:
    for row in rows[:4]:
        _append(
            output_dir / "samples.jsonl",
            {
                "timestamp": clock.now(),
                "step": step,
                "prompt": row["prompt"][0]["content"],
                "completion": row["completion"][0]["content"],
                "reward": row["reward"],
                "advantage": row["advantage"],
            },
        )


def _write_eval(
    output_dir: Path,
    rng: random.Random,
    *,
    step: int,
    envs: tuple[str, ...],
    examples: int,
    rollouts: int,
    clock: _Clock,
    baseline: bool = False,
) -> None:
    metrics: dict[str, Any] = {"step": float(step), "progress/policy_step": float(step)}
    skill = 0.35 if baseline else min(0.9, 0.35 + 0.02 * step)
    for env in envs:
        outputs: list[dict[str, Any]] = []
        for example_index in range(examples):
            prompt = _prompt(rng, env)
            difficulty = rng.random()
            for _ in range(rollouts):
                tokens = rng.randint(60, 760)
                if rng.random() < 0.03:
                    outputs.append(
                        {
                            "example_id": f"{env}-eval-{example_index}",
                            "error": "TimeoutError: verifier client timed out",
                            "completion": [],
                        }
                    )
                    continue
                reward = 1.0 if rng.random() < skill * (1.3 - difficulty) else 0.0
                outputs.append(
                    {
                        "example_id": f"{env}-eval-{example_index}",
                        "task": env,
                        "prompt": [{"role": "user", "content": prompt}],
                        "completion": [
                            {
                                "role": "assistant",
                                "content": _completion(rng, env, reward, tokens),
                            }
                        ],
                        "answer": str(rng.randint(0, 99)),
                        "reward": reward,
                        "metrics": {"correct_answer_reward_func": reward},
                        "is_truncated": tokens > 700,
                        "stop_condition": "generation_truncated"
                        if tokens > 700
                        else "stop",
                        "trajectory": [{"tokens": {"completion_ids": [1] * tokens}}],
                    }
                )
        path = output_dir / "evals" / f"step-{step:06d}" / f"{env}.jsonl"
        _write_jsonl(path, outputs)
        scored = [o["reward"] for o in outputs if "reward" in o]
        total = examples * rollouts
        by_example: dict[str, list[float]] = {}
        for output in outputs:
            by_example.setdefault(output["example_id"], []).append(
                output.get("reward", 0.0)
            )
        pass_at_k = sum(
            1.0 for group in by_example.values() if any(r >= 1.0 for r in group)
        )
        pass_all = sum(
            1.0 for group in by_example.values() if all(r >= 1.0 for r in group)
        )
        lengths = [
            len(o["trajectory"][0]["tokens"]["completion_ids"])
            for o in outputs
            if "trajectory" in o
        ]
        prefix = f"eval/{env}"
        metrics.update(
            {
                f"{prefix}/avg@{rollouts}": sum(scored) / total,
                f"{prefix}/effective/avg@{rollouts}": sum(scored) / max(len(scored), 1),
                f"{prefix}/pass@1": sum(scored) / total,
                f"{prefix}/pass@{rollouts}": pass_at_k / len(by_example),
                f"{prefix}/pass^{rollouts}": pass_all / len(by_example),
                f"{prefix}/failed_rollouts": (total - len(scored)) / total,
                f"{prefix}/time": rng.uniform(8.0, 14.0),
                f"{prefix}/completion_len/mean": sum(lengths) / len(lengths),
                f"{prefix}/completion_len/max": float(max(lengths)),
                f"{prefix}/completion_len/min": float(min(lengths)),
                f"{prefix}/is_truncated/mean": sum(
                    1 for o in outputs if o.get("is_truncated")
                )
                / total,
                f"{prefix}/no_response/mean": sum(
                    1 for o in outputs if not o.get("completion")
                )
                / total,
            }
        )
    metrics["timestamp"] = clock.now()
    _append(output_dir / "eval_metrics.jsonl", metrics)


def _write_logs(output_dir: Path, *, steps: int, clock: _Clock) -> None:
    log_dir = output_dir / "logs" / "attempt_1"
    log_dir.mkdir(parents=True, exist_ok=True)
    _link_latest(output_dir / "logs")
    for name, lines in {
        "rl_trainer.log": [
            f"step {step}: optimizer step complete" for step in range(steps)
        ],
        "rl_inference.log": [
            f"published rollout batch {step}" for step in range(steps)
        ],
        "inference_server_0.log": [
            "INFO vLLM engine ready",
            "INFO Serving policy adapters",
        ],
    }.items():
        (log_dir / name).write_text(
            "\n".join([*lines, f"finished at {clock.now()}"]) + "\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wavelet synth-run",
        description="Write a synthetic RL run directory for dashboard development.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--rollouts-per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--status", default="completed", choices=["completed", "running", "failed"]
    )
    args = parser.parse_args(argv)
    path = write_synthetic_run(
        args.output,
        steps=args.steps,
        groups=args.groups,
        rollouts_per_group=args.rollouts_per_group,
        seed=args.seed,
        status=args.status,
    )
    print(f"Wrote synthetic run to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

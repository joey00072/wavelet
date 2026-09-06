from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wavelet.dashboard.artifacts import RunArtifacts, discover_runs, is_run_dir
from wavelet.dashboard.jsonl import JsonlCache
from wavelet.dashboard.rows import (
    CompactRowCache,
    RowFilters,
    compact_eval_row,
    compact_rollout_row,
    group_rollouts,
    histogram,
    normalize_messages,
    sort_rows,
)
from wavelet.dashboard.server import (
    RunRegistry,
    build_dashboard_app,
    current_run_id,
)
from wavelet.dashboard.synth import write_synthetic_run


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("runs")
    return write_synthetic_run(root / "demo", steps=6, groups=4, rollouts_per_group=4)


@pytest.fixture(scope="module")
def client(synthetic_run: Path) -> TestClient:
    registry = RunRegistry(roots=[synthetic_run.parent])
    return TestClient(build_dashboard_app(registry))


def test_jsonl_cache_reads_appended_rows_incrementally(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0}\n{"step": 1}\n', encoding="utf-8")
    cache = JsonlCache()

    assert [row["step"] for row in cache.rows(path)] == [0, 1]

    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write('{"step": 2}\n')
        handle.write('{"step": 3')

    assert [row["step"] for row in cache.rows(path)] == [0, 1, 2]
    assert cache.parse_errors(path) == 1

    path.write_text('{"step": 9}\n', encoding="utf-8")

    assert [row["step"] for row in cache.rows(path)] == [9]


def test_discover_runs_finds_markers_one_level_deep(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "rollouts").mkdir()
    (tmp_path / "not_a_run").mkdir()
    other = tmp_path / "elsewhere" / "a"
    other.mkdir(parents=True)
    (other / "heartbeat.json").write_text("{}", encoding="utf-8")

    runs = discover_runs([tmp_path], [other])

    assert is_run_dir(tmp_path / "a")
    assert not is_run_dir(tmp_path / "not_a_run")
    assert len(runs) == 3
    assert set(runs.values()) == {
        (tmp_path / "a").resolve(),
        (tmp_path / "b").resolve(),
        other.resolve(),
    }
    assert runs["b"] == (tmp_path / "b").resolve()


def test_synthetic_run_is_deterministic(tmp_path: Path) -> None:
    first = write_synthetic_run(
        tmp_path / "one", steps=3, groups=2, rollouts_per_group=2
    )
    second = write_synthetic_run(
        tmp_path / "two", steps=3, groups=2, rollouts_per_group=2
    )

    assert (first / "metrics.jsonl").read_text() == (
        second / "metrics.jsonl"
    ).read_text()
    assert (first / "rollouts" / "step-000002" / "rollouts.jsonl").read_text() == (
        second / "rollouts" / "step-000002" / "rollouts.jsonl"
    ).read_text()


def test_summary_reports_completed_run_and_config(synthetic_run: Path) -> None:
    summary = RunArtifacts("demo", synthetic_run).summary()

    assert summary["status"] == "completed"
    assert summary["has_config"] is True
    assert summary["target_step"] == 6
    assert summary["trainer_step"] == 5
    assert summary["algo"] == "grpo"
    assert summary["envs"] == ["equation-builder"]
    assert summary["eval_envs"] == ["equation-builder", "reverse-text"]
    assert summary["queue_summary"]["consumed_count"] == 6
    assert summary["policy"]["latest_exported_step"] == 6
    assert {entry["name"] for entry in summary["logs"]} >= {"rl_trainer.log"}


def test_summary_without_config_does_not_leak_defaults(tmp_path: Path) -> None:
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 3, "loss": 0.1}\n', encoding="utf-8"
    )
    (tmp_path / "heartbeat.json").write_text(
        json.dumps({"timestamp": "2026-09-05T00:00:00+00:00", "status": "running"}),
        encoding="utf-8",
    )

    summary = RunArtifacts("bare", tmp_path).summary()

    assert summary["has_config"] is False
    assert summary["model"] is None
    assert summary["target_step"] is None
    assert summary["status"] == "stale"
    assert RunArtifacts("bare", tmp_path).sanitized_config() == {
        "_dashboard_error": "No resolved config artifact was found for this run."
    }


def test_summary_with_invalid_config_does_not_publish_defaults(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "rl.yaml").write_text("max_steps: not-a-number\n", encoding="utf-8")
    (tmp_path / "heartbeat.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-09-05T00:00:00+00:00",
                "status": "completed",
                "step": 3,
            }
        ),
        encoding="utf-8",
    )

    summary = RunArtifacts("invalid", tmp_path).summary()

    assert summary["has_config"] is True
    assert summary["config_error"]
    assert summary["model"] is None
    assert summary["target_step"] is None
    assert summary["batch"] is None
    assert summary["status_reason"] == "finished at step 3"


def test_config_endpoint_reports_malformed_yaml_without_crashing(
    tmp_path: Path,
) -> None:
    run = tmp_path / "malformed"
    config_dir = run / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "rl.yaml").write_text("model: [unterminated\n", encoding="utf-8")
    (run / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    client = TestClient(build_dashboard_app(RunRegistry(runs=[run])))

    response = client.get("/api/runs/malformed/config")

    assert response.status_code == 200
    assert response.json()["_dashboard_error"] == (
        "The resolved config could not be parsed."
    )


def test_legacy_naive_timestamps_and_missing_metric_timestamp_are_safe(
    tmp_path: Path,
) -> None:
    (tmp_path / "metrics.jsonl").write_text('{"step": 3, "loss": 0.1}\n')
    (tmp_path / "heartbeat.json").write_text(
        json.dumps({"timestamp": "2024-01-01T00:00:00", "status": "running"})
    )

    summary = RunArtifacts("legacy", tmp_path).summary()

    assert summary["status"] == "stale"
    assert summary["updated_at"] == "2024-01-01T00:00:00+00:00"


def test_timeline_accepts_mixed_naive_and_aware_timestamps(tmp_path: Path) -> None:
    run = write_synthetic_run(tmp_path / "mixed", steps=2)
    events_path = run / "events" / "queue.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[0]["time"] = str(events[0]["time"]).replace("+00:00", "")
    events_path.write_text("".join(f"{json.dumps(event)}\n" for event in events))

    timeline = RunArtifacts("mixed", run).timeline(limit=100)

    assert timeline["queue_steps"]


def test_compact_row_cache_reports_scan_limit(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(f'{{"reward": {index}}}\n' for index in range(3)))

    rows, scanned, limited = CompactRowCache().rows(
        path, kind="rollout", max_scan_rows=2
    )

    assert len(rows) == 2
    assert scanned == 2
    assert limited is True


def test_missing_advantage_is_not_classified_as_zero() -> None:
    assert RowFilters(advantage_sign="zero").matches({"advantage": None}) is False
    assert RowFilters(advantage_sign="zero").matches({"advantage": 0.0}) is True


def test_saved_eval_samples_are_visible_without_metric_history(tmp_path: Path) -> None:
    eval_dir = tmp_path / "evals" / "step-000003"
    eval_dir.mkdir(parents=True)
    (eval_dir / "legacy-env.jsonl").write_text(
        '{"example_id": "a", "reward": 1, "prompt": "p", "completion": "c"}\n'
    )

    reader = RunArtifacts("samples-only", tmp_path)

    assert reader.evals()["envs"] == [{"name": "legacy-env", "metrics": []}]
    assert reader.summary()["eval_envs"] == ["legacy-env"]


def test_series_returns_aligned_columns(synthetic_run: Path) -> None:
    reader = RunArtifacts("demo", synthetic_run)

    series = reader.series("trainer", ["loss", "missing"], limit=4)

    assert series["steps"] == [2, 3, 4, 5]
    assert len(series["series"]["loss"]) == 4
    assert series["series"]["missing"] == [None] * 4
    assert {entry["key"] for entry in reader.metric_keys()["orchestrator"]} >= {
        "reward/all/mean",
        "off_policy/mean",
    }
    with pytest.raises(KeyError):
        reader.series("nope", ["loss"], limit=1)


def test_evals_group_metrics_by_environment(synthetic_run: Path) -> None:
    reader = RunArtifacts("demo", synthetic_run)

    evals = reader.evals()

    assert [env["name"] for env in evals["envs"]] == [
        "equation-builder",
        "reverse-text",
    ]
    assert "avg@2" in evals["envs"][0]["metrics"]
    steps = sorted(row["step"] for row in evals["history"])
    assert steps == [0, 4]
    assert {(entry["step"], entry["env"]) for entry in evals["sets"]} >= {
        (4, "equation-builder"),
        (0, "reverse-text"),
    }

    rows = reader.eval_rows(
        4,
        "equation-builder",
        sort="reward",
        descending=True,
        offset=0,
        limit=5,
        filters=RowFilters(),
    )
    assert rows["available"] is True
    assert rows["total"] == 24
    rewards = [row["reward"] for row in rows["rows"] if row["reward"] is not None]
    assert rewards == sorted(rewards, reverse=True)
    assert rows["examples"][0]["attempts"] == 2
    assert (
        reader.eval_rows(
            99,
            "equation-builder",
            sort="reward",
            descending=False,
            offset=0,
            limit=1,
            filters=RowFilters(),
        )["available"]
        is False
    )
    assert reader.eval_set_path(4, "../escape") is None


def test_rollout_rows_sort_filter_and_group(synthetic_run: Path) -> None:
    reader = RunArtifacts("demo", synthetic_run)

    latest = reader.rollout_rows(
        None,
        sort="completion_token_count",
        descending=True,
        offset=0,
        limit=3,
        filters=RowFilters(env="reverse-text"),
    )

    assert latest["queue_step"] == 5
    assert latest["total"] == 16
    assert latest["filtered"] == 8
    assert all(row["env"] == "reverse-text" for row in latest["rows"])
    tokens = [row["completion_token_count"] for row in latest["rows"]]
    assert tokens == sorted(tokens, reverse=True)
    assert latest["stats"]["reward_histogram"]["counts"]
    assert len(latest["groups"]) == 4
    assert sum(group["size"] for group in latest["groups"]) == 16

    positive = reader.rollout_rows(
        5,
        sort="advantage",
        descending=False,
        offset=0,
        limit=50,
        filters=RowFilters(advantage_sign="positive"),
    )
    assert all(row["advantage"] > 0 for row in positive["rows"])

    detail = reader.rollout_row(5, 0)
    assert detail is not None
    assert "input_ids" not in detail
    assert detail["arrays"]["loss_mask"]["true_count"] > 0
    assert detail["arrays"]["inference_logprobs"]["length"] > 0
    assert reader.rollout_row(5, 10_000) is None
    assert (
        reader.rollout_rows(
            77, sort="reward", descending=False, offset=0, limit=1, filters=RowFilters()
        )["available"]
        is False
    )


def test_compact_row_summarizes_tokens_and_filters_match() -> None:
    row = {
        "prompt": [{"role": "user", "content": "Reverse abc"}],
        "completion": [{"role": "assistant", "content": "cba"}],
        "reward": 1.0,
        "advantage": 0.5,
        "input_ids": [1, 2, 3, 4],
        "loss_mask": [False, True, True, True],
        "inference_logprobs": [-0.1, -0.5, -0.2],
        "source": "reverse-text",
        "metadata": {
            "group_key": "g",
            "rollout_key": "g:0",
            "is_truncated": False,
            "completion_token_count": 3,
            "task": {"name": "reverse-text", "example_id": "ex-1"},
        },
    }

    compact = compact_rollout_row(row, row_index=4)

    assert compact["sequence_tokens"] == 4
    assert compact["trainable_tokens"] == 3
    assert compact["logprob_min"] == -0.5
    assert compact["example_id"] == "ex-1"
    assert RowFilters(search="reverse").matches(compact)
    assert not RowFilters(min_reward=1.5).matches(compact)
    assert RowFilters(truncated=False, advantage_sign="positive").matches(compact)
    assert group_rollouts([compact])[0]["solve_all"] is True


def test_sort_rows_places_missing_values_last() -> None:
    rows = [{"reward": None}, {"reward": 0.2}, {"reward": 1.0}]

    ascending = sort_rows(rows, key="reward", descending=False)
    descending = sort_rows(rows, key="reward", descending=True)

    assert [row["reward"] for row in ascending] == [0.2, 1.0, None]
    assert [row["reward"] for row in descending] == [1.0, 0.2, None]
    assert histogram([1.0, 1.0])["counts"] == [2]
    assert histogram([])["counts"] == []


def test_timeline_pairs_queue_and_policy_events(synthetic_run: Path) -> None:
    timeline = RunArtifacts("demo", synthetic_run).timeline(limit=100)

    assert [entry["queue_step"] for entry in timeline["queue_steps"]] == list(range(6))
    first = timeline["queue_steps"][0]
    assert first["publish_to_claim_seconds"] is not None
    assert first["claim_to_consume_seconds"] == pytest.approx(6.0)
    assert [entry["policy_step"] for entry in timeline["policies"]] == list(range(7))
    assert timeline["policies"][1]["export_to_load_seconds"] == pytest.approx(0.35)


def test_log_tail_rejects_traversal(synthetic_run: Path) -> None:
    reader = RunArtifacts("demo", synthetic_run)

    tail = reader.log_tail("rl_trainer.log", lines=2)

    assert tail is not None and len(tail["lines"]) == 2
    assert reader.log_tail("../heartbeat.json", lines=2) is None
    assert reader.log_tail("missing.log", lines=2) is None


def test_api_routes_serve_run_artifacts(client: TestClient) -> None:
    runs = client.get("/api/runs").json()
    assert [run["id"] for run in runs] == ["demo"]

    assert client.get("/api/health").json()["runs"] == 1
    assert client.get("/api/runs/demo/summary").json()["status"] == "completed"
    assert client.get("/api/runs/missing/summary").status_code == 404

    series = client.get(
        "/api/runs/demo/series",
        params={"source": "trainer", "keys": "loss", "limit": 2},
    ).json()
    assert len(series["steps"]) == 2
    assert (
        client.get("/api/runs/demo/series", params={"source": "bad"}).status_code == 400
    )

    rows = client.get(
        "/api/runs/demo/rollouts/rows",
        params={"sort": "reward", "order": "desc", "limit": 4, "truncated": "true"},
    ).json()
    assert rows["available"] is True
    assert all(row["is_truncated"] for row in rows["rows"])

    batches = client.get("/api/runs/demo/rollouts").json()
    assert batches[0]["queue_step"] == 5
    detail = client.get("/api/runs/demo/rollouts/5/rows/0")
    assert detail.status_code == 200
    assert detail.json()["row_index"] == 0
    assert client.get("/api/runs/demo/rollouts/5/rows/999").status_code == 404

    evals = client.get("/api/runs/demo/evals").json()
    assert evals["envs"]
    eval_rows = client.get(
        "/api/runs/demo/evals/4/reverse-text/rows", params={"has_error": "true"}
    ).json()
    assert all(row["has_error"] for row in eval_rows["rows"])

    assert client.get("/api/runs/demo/timeline").json()["queue_steps"]
    assert client.get("/api/runs/demo/queue").json()["summary"]["consumed_count"] == 6
    assert client.get("/api/runs/demo/logs/rl_trainer.log").status_code == 200
    assert client.get("/api/runs/demo/logs/nope.log").status_code == 404
    assert client.get("/api/runs/demo/config").json()["algo"] == {"type": "grpo"}
    assert client.get("/api/runs/demo/events").json()[0]["event"] == "run_started"
    assert client.get("/api/runs/demo/samples").json()


def test_live_state_server_mounts_run_api(synthetic_run: Path) -> None:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware

    from wavelet.configs.rl_config import RLConfig
    from wavelet.orchestrator.state_server import (
        OrchestratorRunState,
        _build_state_app,
    )

    state = OrchestratorRunState(RLConfig(output_dir=synthetic_run), target_step=6)
    app = _build_state_app(
        state,
        fastapi=FastAPI,
        query=Query,
        cors_middleware=CORSMiddleware,
        http_exception=HTTPException,
    )
    client = TestClient(app)

    assert client.get("/health").json()["ok"] is True
    assert [run["id"] for run in client.get("/api/runs").json()] == [synthetic_run.name]
    assert (
        client.get(f"/api/runs/{synthetic_run.name}/summary").json()["trainer_step"]
        == 5
    )


def test_current_run_prefers_running_then_most_recent() -> None:
    finished_late = {
        "id": "late",
        "status": "completed",
        "updated_at": "2026-09-05T13:00:00+00:00",
    }
    running = {
        "id": "live",
        "status": "running",
        "updated_at": "2026-09-05T12:00:00+00:00",
    }
    stale = {
        "id": "stale",
        "status": "stale",
        "updated_at": "2026-09-05T12:30:00+00:00",
    }

    assert current_run_id([]) is None
    assert current_run_id([finished_late]) == "late"
    assert current_run_id([finished_late, stale]) == "stale"
    assert current_run_id([finished_late, stale, running]) == "live"


def test_registry_marks_current_run_and_resolves_alias(tmp_path: Path) -> None:
    older = write_synthetic_run(
        tmp_path / "older", steps=2, groups=2, rollouts_per_group=2
    )
    newer = write_synthetic_run(
        tmp_path / "newer",
        steps=3,
        groups=2,
        rollouts_per_group=2,
        started_at=datetime(2026, 9, 6, 8, 0, tzinfo=UTC),
    )
    registry = RunRegistry(roots=[tmp_path])

    summaries = registry.summaries()

    assert [s["id"] for s in summaries] == ["newer", "older"]
    assert [s["is_current"] for s in summaries] == [True, False]
    assert registry._current_run_id() == "newer"
    assert registry.reader("current") is not None
    assert registry.reader("current").output_dir == newer.resolve()
    assert registry.reader("older").output_dir == older.resolve()

    client = TestClient(build_dashboard_app(registry))
    assert client.get("/api/current").json() == {"id": "newer", "runs": 2}
    assert client.get("/api/runs/current/summary").json()["id"] == "newer"
    assert client.get("/api/runs/current/rollouts").json()[0]["queue_step"] == 2


def test_current_alias_without_runs_is_404(tmp_path: Path) -> None:
    client = TestClient(build_dashboard_app(RunRegistry(roots=[tmp_path / "empty"])))

    assert client.get("/api/current").json() == {"id": None, "runs": 0}
    assert client.get("/api/runs/current/summary").status_code == 404


def test_metric_table_windows_and_downsamples(tmp_path: Path) -> None:
    from wavelet.dashboard.metrics import MetricStore

    path = tmp_path / "metrics.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for step in range(100):
            row = {"step": step, "timestamp": f"t{step}", "loss": float(step)}
            if step % 2 == 0:
                row["sparse"] = float(step) * 10
            handle.write(json.dumps(row) + "\n")
    store = MetricStore()
    table = store.table(path)

    assert len(table) == 100
    assert {
        entry["key"]: entry["count"]
        for entry in table.keys()  # noqa: SIM118 - MetricTable is not a mapping
    } == {
        "loss": 100,
        "sparse": 50,
    }
    assert table.value("sparse", 1) is None
    assert table.row(2) == {"step": 2, "timestamp": "t2", "loss": 2, "sparse": 20}

    full = table.series(["loss"], table.indices())
    assert full["downsampled"] is False and full["rows"] == 100

    after = table.series(["loss"], table.indices(after=95))
    assert after["steps"] == [96, 97, 98, 99]

    window = table.series(["loss"], table.indices(start=10, end=12))
    assert window["series"]["loss"] == [10, 11, 12]

    sampled = table.series(["loss", "sparse"], table.indices(), points=10)
    assert sampled["downsampled"] is True and sampled["bucket"] == 10
    assert len(sampled["steps"]) == 10 and sampled["steps"][0] == 9
    assert sampled["series"]["loss"][0] == 4.5
    assert sampled["envelope"]["loss"]["min"][0] == 0
    assert sampled["envelope"]["loss"]["max"][0] == 9
    assert sampled["envelope"]["sparse"]["max"][-1] == 980

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 100, "loss": 1.0, "new_key": 3}) + "\n")
    table = store.table(path)
    assert len(table) == 101
    assert table.value("new_key", 100) == 3 and table.value("new_key", 0) is None
    assert len(table.columns["loss"]) == 101


def test_series_endpoint_supports_points_and_after(client: TestClient) -> None:
    sampled = client.get(
        "/api/runs/demo/series", params={"keys": "loss", "points": 3}
    ).json()
    assert sampled["downsampled"] is True
    assert len(sampled["steps"]) == 3
    assert sampled["total_rows"] == 6
    assert "envelope" in sampled

    incremental = client.get(
        "/api/runs/demo/series", params={"keys": "loss", "after": 3}
    ).json()
    assert incremental["steps"] == [4, 5]
    assert incremental["downsampled"] is False


def test_nodes_endpoint_reports_nodes_ranks_and_replicas(client: TestClient) -> None:
    payload = client.get("/api/runs/demo/nodes").json()

    assert [node["name"] for node in payload["nodes"]] == ["synth-a", "synth-b"]
    assert payload["nodes"][0]["ranks"] == 2
    assert payload["nodes"][0]["tokens_per_second"] > 0
    assert [rank["rank"] for rank in payload["ranks"]] == [0, 1, 2, 3]
    assert payload["ranks"][2]["node"] == "synth-b"
    assert payload["replicas"][0]["name"] == "replica_0"
    assert payload["replicas"][0]["generation_tokens_per_second"] > 0
    assert payload["step"] == 5
    assert payload["world"]["world_size"] >= 1


def test_jsonl_cache_bounds_rows_and_counts_dropped(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "".join(json.dumps({"i": i}) + "\n" for i in range(10)), encoding="utf-8"
    )
    cache = JsonlCache(max_rows=4)

    assert [row["i"] for row in cache.rows(path)] == [6, 7, 8, 9]
    assert cache.dropped(path) == 6

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"i": 10}) + "\n")
    assert [row["i"] for row in cache.rows(path)] == [7, 8, 9, 10]
    assert cache.dropped(path) == 7


def test_registry_evicts_idle_readers_but_keeps_live_runs(tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        write_synthetic_run(tmp_path / name, steps=1, groups=1, rollouts_per_group=2)
    registry = RunRegistry(roots=[tmp_path], live={"a": tmp_path / "a"}, max_readers=2)

    first_a = registry.reader("a")
    registry.reader("b")
    registry.reader("c")

    assert set(registry._readers) == {"a", "c"}
    assert registry.reader("a") is first_a


def test_latest_trainer_row_merges_rows_sharing_the_newest_step(tmp_path: Path) -> None:
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"step": 3, "loss": 0.5, "timestamp": "2026-09-05T10:00:00+00:00"}
                ),
                json.dumps(
                    {"step": 4, "loss": 0.4, "timestamp": "2026-09-05T10:01:00+00:00"}
                ),
                json.dumps(
                    {
                        "step": 4,
                        "perf/step_seconds": 40.0,
                        "timestamp": "2026-09-05T10:01:01+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    latest = RunArtifacts("merge", tmp_path).latest_rows()["trainer"]

    assert latest is not None
    assert latest["loss"] == 0.4
    assert latest["perf/step_seconds"] == 40.0
    assert latest["step"] == 4


def test_normalize_messages_parses_repr_serialized_chat_messages() -> None:
    legacy = [
        "role='user' content='Sort: a\\nb'",
        "role='assistant' content=\"It's <x>\" reasoning_content=None tool_calls=None",
    ]

    messages = normalize_messages(legacy)

    assert messages == [
        {"role": "user", "content": "Sort: a\nb"},
        {"role": "assistant", "content": "It's <x>"},
    ]
    assert normalize_messages("plain text") == "plain text"
    assert normalize_messages([{"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "x"}
    ]
    compact = compact_eval_row(
        {
            "example_id": "e",
            "reward": 1.0,
            "prompt": legacy[:1],
            "completion": legacy[1:],
        },
        row_index=0,
    )
    assert compact["prompt"] == "user: Sort: a\nb"
    assert compact["completion"] == "assistant: It's <x>"


def test_trackio_source_reads_sqlite_history(tmp_path: Path) -> None:
    import sqlite3

    from wavelet.configs.rl_config import RLConfig
    from wavelet.dashboard.external import trackio_source

    root = tmp_path / "trackio"
    root.mkdir()
    connection = sqlite3.connect(root / "wavelet.db")
    connection.execute(
        "CREATE TABLE metrics (id INTEGER PRIMARY KEY, timestamp TEXT, "
        "run_name TEXT, step INTEGER, metrics TEXT)"
    )
    for step, loss in enumerate([0.9, 0.5, 0.2]):
        connection.execute(
            "INSERT INTO metrics (timestamp, run_name, step, metrics) VALUES (?, ?, ?, ?)",
            (
                f"2026-09-05T12:0{step}:00+00:00",
                "demo",
                step,
                json.dumps({"loss": loss}),
            ),
        )
    connection.commit()
    connection.close()
    run_dir = tmp_path / "demo"
    run_dir.mkdir()
    (run_dir / "trackio_run.json").write_text(
        json.dumps({"project": "wavelet", "run": "demo"}), encoding="utf-8"
    )

    source = trackio_source(RLConfig(output_dir=run_dir), run_dir, root=root)

    assert source.available
    source.refresh_now()
    table = source.table()
    assert len(table) == 3
    assert [
        entry["key"]
        for entry in table.keys()  # noqa: SIM118 - MetricTable is not a mapping
    ] == ["loss"]
    assert source.status()["status"] == "ready"

    missing = trackio_source(
        RLConfig(output_dir=tmp_path / "other"), tmp_path / "other", root=root
    )
    assert not missing.available
    assert missing.status()["status"] == "unavailable"


def test_wandb_history_maps_private_fields() -> None:
    from wavelet.dashboard.external import fetch_wandb_history

    class FakeRun:
        def scan_history(self):
            return [
                {"_step": 0, "_timestamp": 1_800_000_000, "_runtime": 1.0, "loss": 0.5},
                {
                    "_step": 1,
                    "_timestamp": 1_800_000_060,
                    "loss": 0.25,
                    "reward/all/mean": 0.7,
                },
            ]

    class FakeApi:
        def run(self, path: str) -> FakeRun:
            assert path == "team/proj/abc123"
            return FakeRun()

    rows = fetch_wandb_history("team/proj/abc123", api=FakeApi())

    assert rows[0]["step"] == 0
    assert rows[0]["timestamp"].startswith("2027-")
    assert "_runtime" not in rows[0]
    assert rows[1]["reward/all/mean"] == 0.7


def test_external_status_reports_disabled_wandb(synthetic_run: Path) -> None:
    reader = RunArtifacts("demo", synthetic_run)

    status = {entry["source"]: entry for entry in reader.external_status()}

    assert status["wandb"]["status"] == "unavailable"
    assert "disabled" in status["wandb"]["reason"]
    assert status["trackio"]["status"] == "unavailable"
    assert "wandb" not in reader.metric_keys()

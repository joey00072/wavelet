import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from wavelet.dashboard.live import list_episodes, read_episode
from wavelet.dashboard.server import RunRegistry, build_dashboard_app


def test_live_episode_api_returns_latest_transcript_and_rejects_bad_ids(tmp_path):
    episode_id = str(uuid4())
    directory = tmp_path / "traces" / "live"
    directory.mkdir(parents=True)
    snapshot = {
        "id": episode_id,
        "env": "reverse",
        "kind": "eval",
        "status": "running",
        "phase": "get_model_response.started",
        "events": [{"prompt": [{"role": "user", "content": "hello"}]}],
    }
    path = directory / f"{episode_id}.json"
    path.write_text(json.dumps(snapshot))
    meta = directory / f"{episode_id}.meta.json"
    meta.write_text(
        json.dumps(
            {key: value for key, value in snapshot.items() if key != "events"}
            | {"revision": "1"}
        )
    )
    client = TestClient(build_dashboard_app(RunRegistry(live={"run": tmp_path})))
    listing = client.get("/api/runs/run/episodes").json()
    assert listing["episodes"][0]["id"] == episode_id
    assert "events" not in listing["episodes"][0]
    first_revision = listing["episodes"][0]["revision"]
    response = client.get(f"/api/runs/run/episodes/{episode_id}")
    assert response.json()["events"][0]["prompt"][0]["content"] == "hello"
    snapshot["status"] = "completed"
    path.write_text(json.dumps(snapshot))
    meta.write_text(
        json.dumps(
            {key: value for key, value in snapshot.items() if key != "events"}
            | {"revision": "2"}
        )
    )
    assert (
        client.get("/api/runs/run/episodes").json()["episodes"][0]["revision"]
        != first_revision
    )
    assert (
        client.get(f"/api/runs/run/episodes/{episode_id}").json()["status"]
        == "completed"
    )
    assert client.get("/api/runs/run/episodes/not-a-uuid").status_code == 404
    assert client.get("/api/runs/missing/episodes").status_code == 404


def test_live_reader_skips_corrupt_and_symlink_snapshots(tmp_path):
    directory = tmp_path / "traces" / "live"
    directory.mkdir(parents=True)
    broken_id, linked_id = str(uuid4()), str(uuid4())
    (directory / f"{broken_id}.json").write_text('{"unfinished":')
    external = tmp_path / "outside.json"
    external.write_text(json.dumps({"id": linked_id, "private": "data"}))
    (directory / f"{linked_id}.json").symlink_to(external)
    assert read_episode(tmp_path, linked_id) is None
    assert list_episodes(tmp_path)["episodes"] == []


def test_live_reader_rejects_symlinked_directory_outside_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    episode_id = str(uuid4())
    (other / f"{episode_id}.json").write_text(json.dumps({"id": episode_id}))
    (run / "traces").mkdir()
    (run / "traces" / "live").symlink_to(other, target_is_directory=True)
    assert read_episode(run, episode_id) is None
    assert list_episodes(run)["episodes"] == []


@pytest.mark.parametrize("depth", [600, 1500])
def test_live_reader_skips_excessively_nested_snapshots(tmp_path, depth):
    episode_id = str(uuid4())
    directory = tmp_path / "traces" / "live"
    directory.mkdir(parents=True)
    # Exercise both successful parsing followed by deep redaction, and parser
    # recursion failure, while remaining well below either byte limit.
    nested = '{"nested":' * depth + "0" + "}" * depth
    payload = '{"id":"' + episode_id + '","events":' + nested + "}"
    (directory / f"{episode_id}.json").write_text(payload)
    (directory / f"{episode_id}.meta.json").write_text(payload)

    assert read_episode(tmp_path, episode_id) is None
    assert list_episodes(tmp_path)["episodes"] == []


def test_live_reader_skips_named_pipes_without_blocking(tmp_path, monkeypatch):
    episode_id = str(uuid4())
    directory = tmp_path / "traces" / "live"
    directory.mkdir(parents=True)
    paths = {
        directory / f"{episode_id}.json",
        directory / f"{episode_id}.meta.json",
    }
    for path in paths:
        os.mkfifo(path)

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        # A regression to ordinary buffered opening must fail instead of
        # hanging the test waiting for a FIFO writer.
        assert path not in paths, "A live snapshot pipe must not be opened blocking"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    assert read_episode(tmp_path, episode_id) is None
    assert list_episodes(tmp_path)["episodes"] == []

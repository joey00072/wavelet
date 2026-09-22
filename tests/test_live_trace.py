import asyncio
import json
from unittest.mock import patch

import pytest

from wavelet.orchestrator.live import install_verifier_trace_hooks, live_episode


class FakeVerifier:
    class Rubric:
        async def score_rollout(self, state):
            return {"reward": 1, "state": state}

    def __init__(self):
        self.rubric = self.Rubric()

    async def setup_state(self, state, **kwargs):
        await asyncio.sleep(0)
        return {"state": state}

    async def get_model_response(self, state, prompt, **kwargs):
        await asyncio.sleep(0)
        return {"reply": prompt}

    async def add_model_response(self, state, response, **kwargs):
        return state | {"response": response}

    async def env_response(self, messages, state, **kwargs):
        return {"messages": messages, "state": state}


def test_live_trace_captures_phases_and_is_idempotent(tmp_path):
    env = install_verifier_trace_hooks(FakeVerifier())
    install_verifier_trace_hooks(env)
    with live_episode(
        tmp_path, "fake", "rollout", step=3, policy_step=4, example_id="x"
    ):

        async def run():
            state = await env.setup_state("prompt")
            reply = await env.get_model_response(state, "hello")
            state = await env.add_model_response(state, reply)
            await env.env_response([reply], state)
            await env.rubric.score_rollout(state)

        asyncio.run(run())
    paths = [
        path
        for path in (tmp_path / "traces" / "live").glob("*.json")
        if not path.name.endswith(".meta.json")
    ]
    assert len(paths) == 1
    data = json.loads(paths[0].read_text())
    assert data["status"] == "completed"
    assert data["policy_step"] == 4
    phases = [event["phase"] for event in data["events"]]
    assert phases.count("setup_state.started") == 1
    assert "get_model_response.finished" in phases
    assert "score_rollout.finished" in phases


def test_live_trace_isolated_across_async_tasks(tmp_path):
    env = install_verifier_trace_hooks(FakeVerifier())

    async def run(example_id):
        with live_episode(tmp_path, "fake", "rollout", example_id=example_id):
            await env.get_model_response({}, example_id)

    async def main():
        await asyncio.gather(run("a"), run("b"))

    asyncio.run(main())
    completed = [
        path
        for path in (tmp_path / "traces" / "live").glob("*.json")
        if not path.name.endswith(".meta.json")
    ]
    assert len(completed) == 2
    assert {json.loads(path.read_text())["example_id"] for path in completed} == {
        "a",
        "b",
    }


def test_live_trace_cancellation_and_disk_failure_are_safe(tmp_path):
    async def cancelled():
        with live_episode(tmp_path, "fake", "rollout"):
            raise asyncio.CancelledError

    try:
        asyncio.run(cancelled())
    except asyncio.CancelledError:
        pass
    data = [
        json.loads(path.read_text())
        for path in (tmp_path / "traces" / "live").glob("*.json")
        if not path.name.endswith(".meta.json")
    ]
    assert data[0]["status"] == "cancelled"
    with (
        patch("wavelet.orchestrator.live.atomic_json", side_effect=OSError("full")),
        live_episode(tmp_path, "fake", "rollout") as episode,
    ):
        episode.event("prompt", prompt="exact hello")
    assert episode.data["status"] == "completed"


def test_live_trace_preserves_prompt(tmp_path):
    env = install_verifier_trace_hooks(FakeVerifier())
    with live_episode(tmp_path, "fake", "rollout"):
        asyncio.run(env.get_model_response({}, "exact hello"))
    data = json.loads(
        next(
            path
            for path in (tmp_path / "traces" / "live").glob("*.json")
            if not path.name.endswith(".meta.json")
        ).read_text()
    )
    event = next(
        item for item in data["events"] if item["phase"] == "get_model_response.started"
    )
    assert event["prompt"] == "exact hello"


def test_cyclic_prompt_cannot_break_verifier(tmp_path):
    env = install_verifier_trace_hooks(FakeVerifier())
    prompt = []
    prompt.append(prompt)
    with live_episode(tmp_path, "fake", "rollout") as episode:
        result = asyncio.run(env.get_model_response({}, prompt))
    assert result["reply"] is prompt
    assert episode.path.stat().st_size <= 256 * 1024


def test_group_members_have_separate_traces(tmp_path):
    class GroupVerifier(FakeVerifier):
        async def rollout(self, prompt):
            await self.get_model_response({}, prompt)
            return {"reward": 1.0}

    env = install_verifier_trace_hooks(GroupVerifier())

    async def run():
        with live_episode(tmp_path, "fake", "group") as group:
            await asyncio.gather(env.rollout("a"), env.rollout("b"))
        return group.data["id"]

    group_id = asyncio.run(run())
    children = [
        json.loads(p.read_text())
        for p in (tmp_path / "traces/live").glob("*.json")
        if not p.name.endswith(".meta.json") and p.stem != group_id
    ]
    assert len(children) == 2
    assert all(child["parent_id"] == group_id for child in children)
    prompts = [
        [
            event["prompt"]
            for event in child["events"]
            if event["phase"] == "get_model_response.started"
        ]
        for child in children
    ]
    assert sorted(prompts) == [["a"], ["b"]]


def test_dead_writer_is_reported_abandoned(tmp_path):
    from wavelet.dashboard.live import list_episodes, read_episode

    with live_episode(tmp_path, "fake", "rollout") as episode:
        episode.data["process_start"] = "not-this-process"
        episode.event("waiting")
        assert read_episode(tmp_path, episode.data["id"])["status"] == "abandoned"
        assert list_episodes(tmp_path)["episodes"][0]["status"] == "abandoned"


def test_retention_ignores_fifo_metadata(tmp_path):
    import os

    directory = tmp_path / "traces/live"
    directory.mkdir(parents=True)
    os.mkfifo(directory / "invalid.meta.json")
    with live_episode(tmp_path, "fake", "rollout") as episode:
        pass
    assert episode.data["status"] == "completed"


def test_async_trace_writes_leave_event_loop_responsive(tmp_path, monkeypatch):
    import threading

    from wavelet.orchestrator import live

    original = live.atomic_json
    main_thread = threading.get_ident()
    writer_threads = []
    started = threading.Event()
    release = threading.Event()

    def slow_write(path, value):
        writer_threads.append(threading.get_ident())
        started.set()
        assert release.wait(timeout=5)
        original(path, value)

    monkeypatch.setattr(live, "atomic_json", slow_write)
    env = install_verifier_trace_hooks(FakeVerifier())

    async def run():
        async with live_episode(tmp_path, "fake", "rollout") as episode:
            await env.get_model_response({}, "hello")
        return episode

    async def main():
        task = asyncio.create_task(run())
        assert await asyncio.to_thread(started.wait, 5)
        # This coroutine must run while the filesystem worker is blocked.
        assert not task.done()
        release.set()
        return await task

    episode = asyncio.run(main())
    assert writer_threads and main_thread not in writer_threads
    assert json.loads(episode.path.read_text())["status"] == "completed"


@pytest.mark.parametrize("cancel_phase", ["episode_started", "blocked"])
def test_cancelled_async_trace_finishes_write_before_terminal_state(
    tmp_path, monkeypatch, cancel_phase
):
    import threading

    from wavelet.orchestrator import live

    original = live.atomic_json
    started = threading.Event()
    release = threading.Event()

    def slow_event_write(path, value):
        if value.get("phase") == cancel_phase and value.get("status") == "running":
            started.set()
            assert release.wait(timeout=5)
        original(path, value)

    monkeypatch.setattr(live, "atomic_json", slow_event_write)

    async def run():
        async with live_episode(tmp_path, "fake", "rollout") as episode:
            await episode.aevent("blocked")

    async def main():
        task = asyncio.create_task(run())
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation was swallowed")

    asyncio.run(main())
    paths = [
        p
        for p in (tmp_path / "traces/live").glob("*.json")
        if not p.name.endswith(".meta.json")
    ]
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["status"] == "cancelled"
    assert not list(paths[0].parent.glob("*.tmp"))


def test_concurrent_async_events_publish_serially(tmp_path):
    async def run():
        async with live_episode(tmp_path, "fake", "rollout") as episode:
            await asyncio.gather(*(episode.aevent(f"event-{i}") for i in range(32)))
        return episode

    episode = asyncio.run(run())
    data = json.loads(episode.path.read_text())
    assert data["status"] == "completed"
    assert {event["phase"] for event in data["events"]} == {
        "episode_started",
        *(f"event-{i}" for i in range(32)),
    }

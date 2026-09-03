from __future__ import annotations

import threading
from pathlib import Path

from fastapi import HTTPException

from wavelet.configs.rl_config import RLConfig
from wavelet.inference import native_server


class _FakeEngine:
    def __init__(self, _config: RLConfig) -> None:
        self.policy_step: int | None = None

    def load_policy(self, path: Path, *, step: int) -> None:
        self.loaded_policy = (path, step)
        self.policy_step = step


def _endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if route.path == path)


def test_native_server_registers_compatibility_routes(monkeypatch) -> None:
    monkeypatch.setattr(
        native_server,
        "VLLMPolicyInferenceEngine",
        _FakeEngine,
    )

    app = native_server._build_app(RLConfig())
    paths = {route.path for route in app.routes}

    assert {
        "/health",
        "/pause",
        "/resume",
        "/load_policy",
        "/load_lora_adapter",
        "/update_weights",
        "/annotate",
        "/v1/chat/completions",
        "/v1/chat/completions/tokens",
        "/chat/completions/tokens",
        "/v1/tokenize",
        "/tokenize",
    } <= paths


def test_generation_gate_drains_and_blocks_until_resume() -> None:
    gate = native_server._GenerationGate()
    active = threading.Event()
    release_active = threading.Event()
    pause_returned = threading.Event()
    queued_returned = threading.Event()

    def active_generation() -> None:
        with gate.generation():
            active.set()
            release_active.wait()

    def pause() -> None:
        gate.pause()
        pause_returned.set()

    def queued_generation() -> None:
        with gate.generation():
            queued_returned.set()

    active_thread = threading.Thread(target=active_generation)
    pause_thread = threading.Thread(target=pause)
    queued_thread = threading.Thread(target=queued_generation)
    active_thread.start()
    assert active.wait(timeout=1)
    pause_thread.start()
    assert not pause_returned.wait(timeout=0.05)
    release_active.set()
    assert pause_returned.wait(timeout=1)
    queued_thread.start()
    assert not queued_returned.wait(timeout=0.05)

    gate.resume()

    assert queued_returned.wait(timeout=1)
    active_thread.join()
    pause_thread.join()
    queued_thread.join()


def test_load_policy_route_updates_engine(monkeypatch) -> None:
    monkeypatch.setattr(
        native_server,
        "VLLMPolicyInferenceEngine",
        _FakeEngine,
    )
    app = native_server._build_app(RLConfig())

    response = _endpoint(app, "/load_policy")({"policy_dir": "/tmp/policy", "step": 7})

    assert response == {"status": "ok", "policy_step": 7}


def test_update_weights_route_requires_a_source(monkeypatch) -> None:
    monkeypatch.setattr(
        native_server,
        "VLLMPolicyInferenceEngine",
        _FakeEngine,
    )
    app = native_server._build_app(RLConfig())

    try:
        _endpoint(app, "/update_weights")({})
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("Expected a missing update source to fail")

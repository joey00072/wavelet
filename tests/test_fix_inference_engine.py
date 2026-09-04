from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.inference import engine as engine_module
from wavelet.inference.engine import (
    ADMIN_CONTROL_TIMEOUT_SECONDS,
    NCCL_READY_MARKER,
    POLICY_LOAD_TIMEOUT_SECONDS,
    HTTPPolicyInferenceEngine,
    VLLMPolicyInferenceEngine,
)


class _FakeLLMEngine:
    def __init__(self) -> None:
        self.events: list[str] = []

    def add_lora(self, request: Any) -> bool:
        self.events.append(f"add_lora:{request.lora_path}")
        return True

    def reset_prefix_cache(self) -> bool:
        self.events.append("reset_prefix_cache")
        return True


class _FakeLLM:
    def __init__(self, request_outputs: list[Any] | None = None) -> None:
        self.llm_engine = _FakeLLMEngine()
        self.generate_calls: list[dict[str, Any]] = []
        self._request_outputs = request_outputs or []

    def generate(self, prompts, sampling_params, *, use_tqdm, lora_request):
        del use_tqdm
        self.generate_calls.append(
            {"prompts": prompts, "sampling_params": sampling_params}
        )
        return self._request_outputs


class _FakeTokenizer:
    def __init__(self) -> None:
        self.template_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        del messages
        self.template_kwargs.append(kwargs)
        return [11, 12, 13]


class _HTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_inplace_adapter_reload_resets_prefix_cache() -> None:
    config = RLConfig(
        policy_transfer={"adapter_name": "policy", "adapter_id": 7},
        lora={"rank": 4, "target_modules": ["q_proj"]},
    )
    engine = VLLMPolicyInferenceEngine(config)
    engine.llm = _FakeLLM()

    engine._load_adapter_policy(Path("policies/step_1/adapter"), step=1)
    engine._load_adapter_policy(Path("policies/step_2/adapter"), step=2)

    assert engine.llm.llm_engine.events == [
        "add_lora:policies/step_1/adapter",
        "reset_prefix_cache",
        "add_lora:policies/step_2/adapter",
        "reset_prefix_cache",
    ]


def test_offline_chat_batch_forwards_salt_template_kwargs_and_finish_reason(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        engine_module, "_sampling_params_type", lambda: lambda **kwargs: kwargs
    )
    outputs = [
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    text="hi",
                    token_ids=[21],
                    logprobs=[{21: -0.5}],
                    finish_reason="abort",
                )
            ]
        )
    ]
    engine = VLLMPolicyInferenceEngine(RLConfig(data={"seq_len": 64}))
    engine.llm = _FakeLLM(outputs)
    engine.tokenizer = _FakeTokenizer()

    [result] = engine._openai_chat_completion_batch(
        [
            {
                "messages": [{"role": "user", "content": "hi"}],
                "cache_salt": "step-3",
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ]
    )

    [call] = engine.llm.generate_calls
    assert call["prompts"][0]["cache_salt"] == "step-3"
    assert engine.tokenizer.template_kwargs[0]["enable_thinking"] is False
    # Aborted generations are not reported as clean stops.
    assert result["choices"][0]["finish_reason"] == "abort"


def test_nccl_ready_marker_is_written_with_custom_rollout_function(tmp_path) -> None:
    config = RLConfig(
        lora=None,
        policy_transfer={"type": "nccl"},
        orchestrator={"custom_rollout_function": "my_rollouts:generate"},
    )
    engine = HTTPPolicyInferenceEngine(config)
    engine._load_policy_while_generation_paused = lambda payload: [  # type: ignore[method-assign]
        {"policy_step": payload["step"]}
    ]

    engine.load_policy(tmp_path, step=1)

    assert (tmp_path / NCCL_READY_MARKER).exists()
    assert engine.policy_step == 1


@pytest.mark.parametrize("failure_kind", ["transport", "server"])
def test_admin_request_retries_transient_errors_then_succeeds(
    monkeypatch, tmp_path: Path, failure_kind: str
) -> None:
    engine = HTTPPolicyInferenceEngine(RLConfig())
    attempts: list[float] = []
    delays: list[float] = []

    def flaky_open(_request: object, *, timeout: float) -> _HTTPResponse:
        attempts.append(timeout)
        if len(attempts) < 3:
            if failure_kind == "transport":
                raise urllib.error.URLError("server restarting")
            raise urllib.error.HTTPError(
                "http://server/load_policy",
                503,
                "unavailable",
                {},
                io.BytesIO(b"busy"),
            )
        return _HTTPResponse({"policy_step": 4})

    monkeypatch.setattr(engine_module.urllib.request, "urlopen", flaky_open)
    monkeypatch.setattr(engine_module.time, "sleep", delays.append)

    engine.load_policy(tmp_path, step=4)

    assert attempts == [POLICY_LOAD_TIMEOUT_SECONDS] * 3
    assert delays == [1.0, 2.0]
    assert engine.policy_step == 4


def test_policy_step_mismatch_is_not_retried(monkeypatch, tmp_path: Path) -> None:
    engine = HTTPPolicyInferenceEngine(RLConfig())
    attempts = 0

    def wrong_step(_request: object, *, timeout: float) -> _HTTPResponse:
        nonlocal attempts
        attempts += 1
        assert timeout == POLICY_LOAD_TIMEOUT_SECONDS
        return _HTTPResponse({"policy_step": 3})

    monkeypatch.setattr(engine_module.urllib.request, "urlopen", wrong_step)

    with pytest.raises(RuntimeError, match="wrong policy step"):
        engine.load_policy(tmp_path, step=4)

    assert attempts == 1


def test_admin_operations_use_separate_timeouts(monkeypatch) -> None:
    config = RLConfig(inference={"http": {"request_timeout_seconds": 17.0}})
    engine = HTTPPolicyInferenceEngine(config)
    observed: list[float] = []

    def capture_timeout(_request: object, *, timeout: float) -> _HTTPResponse:
        observed.append(timeout)
        return _HTTPResponse({})

    monkeypatch.setattr(engine_module.urllib.request, "urlopen", capture_timeout)

    engine._request("POST", "/pause")
    engine._request("POST", "/load_policy")
    engine._request("POST", "/resume")
    engine._request("POST", "/annotate")

    assert observed == [
        ADMIN_CONTROL_TIMEOUT_SECONDS,
        POLICY_LOAD_TIMEOUT_SECONDS,
        ADMIN_CONTROL_TIMEOUT_SECONDS,
        17.0,
    ]


def test_http_setup_checks_served_model_identity(monkeypatch) -> None:
    config = RLConfig(
        model={"name": "expected/model"},
        orchestrator={"custom_rollout_function": "custom:rollouts"},
    )
    engine = HTTPPolicyInferenceEngine(config)
    paths: list[str] = []

    def request(
        _method: str,
        path: str,
        _payload: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        del base_url
        paths.append(path)
        if path == "/v1/models":
            return {"data": [{"id": "expected/model"}]}
        return {}

    monkeypatch.setattr(engine, "_request", request)

    engine.setup()

    assert paths == ["/health", "/v1/models"]


def test_http_setup_rejects_wrong_served_model(monkeypatch) -> None:
    config = RLConfig(
        model={"name": "expected/model"},
        orchestrator={"custom_rollout_function": "custom:rollouts"},
    )
    engine = HTTPPolicyInferenceEngine(config)

    def request(
        _method: str,
        path: str,
        _payload: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        del base_url
        if path == "/v1/models":
            return {"data": [{"id": "other/model"}]}
        return {}

    monkeypatch.setattr(engine, "_request", request)

    with pytest.raises(ValueError, match="expected/model.*other/model"):
        engine.setup()

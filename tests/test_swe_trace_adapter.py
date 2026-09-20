from __future__ import annotations

import asyncio
import copy
import sys
from types import SimpleNamespace

import httpx
import pytest

from examples.qwen30b_swe.bridge_env import NativeSWEEnvironment
from examples.qwen30b_swe.runtime_compat import install
from examples.qwen30b_swe.trace_adapter import sample_digest, trace_to_output
from wavelet.orchestrator.envs import _interleave_output


def _trace():
    return {
        "id": "test",
        "ok": True,
        "calls": [],
        "nodes": [
            {
                "parent": None,
                "message": {"role": "user", "content": "fix"},
                "token_ids": [1, 2],
                "mask": [False, False],
                "sampled": False,
            },
            {
                "parent": 0,
                "message": {"role": "assistant", "content": "tool"},
                "token_ids": [3, 4, 5],
                "mask": [False, True, True],
                "sampled": True,
                "logprobs": [-0.2, -0.3],
            },
            {
                "parent": 1,
                "message": {"role": "tool", "content": "result"},
                "token_ids": [6, 7],
                "mask": [False, False],
                "sampled": False,
            },
            {
                "parent": 2,
                "message": {"role": "assistant", "content": "patch"},
                "token_ids": [8, 9],
                "mask": [False, True],
                "sampled": True,
                "logprobs": [-0.4],
            },
        ],
    }


def test_native_swe_adapter_preserves_all_turns_and_tool_masks():
    output = trace_to_output(_trace(), reward=1)
    samples = _interleave_output(output, 1)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["input_ids"] == list(range(1, 9))
    assert sample["target_ids"] == list(range(2, 10))
    assert sample["loss_mask"] == [False, False, True, True, False, False, False, True]
    assert [
        p for p, mask in zip(sample["inference_logprobs"], sample["loss_mask"]) if mask
    ] == [-0.2, -0.3, -0.4]


def test_native_swe_adapter_retains_rerendered_branches():
    trace = _trace()
    trace["nodes"][2]["parent"] = 0
    samples = _interleave_output(trace_to_output(trace, reward=0), 1)
    assert len(samples) == 2
    assert sum(sum(s["loss_mask"]) for s in samples) == 3
    assert samples[1]["input_ids"] == [1, 2, 6, 7, 8]


def test_native_swe_eval_keeps_messages_without_training_token_data():
    trace = _trace()
    for node in trace["nodes"]:
        node["token_ids"] = []
        node["mask"] = []
        node["logprobs"] = []
    output = trace_to_output(trace, reward=1)
    assert output["reward"] == 1
    assert output["completion"] == [
        trace["nodes"][1]["message"],
        trace["nodes"][3]["message"],
    ]
    assert output["trajectory"] == []


def test_native_swe_adapter_rejects_invalid_logprob_alignment():
    trace = _trace()
    trace["nodes"][1]["logprobs"] = []
    with pytest.raises(ValueError, match="logprobs"):
        trace_to_output(trace, reward=0)


def test_native_swe_adapter_rejects_parent_cycle():
    trace = _trace()
    trace["nodes"][0]["parent"] = 1
    with pytest.raises(ValueError, match="parent graph"):
        trace_to_output(trace, reward=0)


def test_swe_digest_detects_changed_context_mask_or_logprob():
    samples = _interleave_output(trace_to_output(_trace(), reward=1), 1)
    original = sample_digest(samples)
    for field, index, value in [
        ("input_ids", 0, 99),
        ("loss_mask", 2, False),
        ("inference_logprobs", 2, -0.8),
    ]:
        changed = copy.deepcopy(samples)
        changed[0][field][index] = value
        assert sample_digest(changed) != original


def test_swe_uv_bootstrap_upgrades_incapable_images_and_is_idempotent(monkeypatch):
    original = "command -v uv >/dev/null 2>&1 || install_uv"
    base = SimpleNamespace(_ENSURE_UV=original)
    monkeypatch.setitem(
        sys.modules, "verifiers.v1.runtimes", SimpleNamespace(base=base)
    )
    install()
    assert "uv sync --help" in base._ENSURE_UV
    assert base._ENSURE_UV.endswith(" || install_uv")
    fixed = base._ENSURE_UV
    install()
    assert base._ENSURE_UV == fixed


def test_swe_uv_bootstrap_rejects_unknown_native_bootstrap(monkeypatch):
    base = SimpleNamespace(_ENSURE_UV="changed upstream bootstrap")
    monkeypatch.setitem(
        sys.modules, "verifiers.v1.runtimes", SimpleNamespace(base=base)
    )
    with pytest.raises(RuntimeError, match="bootstrap changed"):
        install()


@pytest.mark.parametrize("training", [True, False])
def test_swe_bridge_pins_training_renderer_and_keeps_native_eval(monkeypatch, training):
    env = object.__new__(NativeSWEEnvironment)
    env.process = SimpleNamespace(poll=lambda: None)
    env.training = training
    env.base_model = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    env.renderer = {"name": "qwen3"}
    env.url = "http://bridge"
    output = trace_to_output(_trace(), reward=1)
    output["native_training_digest"] = sample_digest(_interleave_output(output, 1))
    requests = []

    class HTTPClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json):
            requests.append(json)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: output)

    monkeypatch.setattr("examples.qwen30b_swe.bridge_env.httpx.AsyncClient", HTTPClient)
    client = SimpleNamespace(
        api_base_url="http://inference:8000/v1",
        api_key_var="LOCAL_INFERENCE_KEY",
        extra_headers={},
    )
    asyncio.run(
        env.run_rollout(
            {"example_id": 0, "info": {"native_task": {"idx": 0}}},
            client=client,
            model="policy-17",
            sampling_args={"temperature": 1},
        )
    )
    request = requests[0]
    assert request["model"] == "policy-17"
    if training:
        assert request["client"]["renderer_model_name"] == env.base_model
        assert request["client"]["renderer"] == {"name": "qwen3"}
        assert request["client"]["base_url"] == "http://inference:8000"
    else:
        assert "renderer_model_name" not in request["client"]
        assert "renderer" not in request["client"]
        assert request["client"]["base_url"] == "http://inference:8000/v1"


def test_swe_bridge_reports_failed_episode_identifier(monkeypatch):
    env = object.__new__(NativeSWEEnvironment)
    env.process = SimpleNamespace(poll=lambda: None)
    env.training = False
    env.url = "http://bridge"

    class HTTPClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json):
            return httpx.Response(
                502,
                request=httpx.Request("POST", url),
                json={"detail": "Native SWE episode failed-123 failed; inspect trace."},
            )

    monkeypatch.setattr("examples.qwen30b_swe.bridge_env.httpx.AsyncClient", HTTPClient)
    client = SimpleNamespace(
        api_base_url="http://inference:8000/v1",
        api_key_var="LOCAL_INFERENCE_KEY",
        extra_headers={},
    )
    with pytest.raises(RuntimeError, match="HTTP 502.*failed-123"):
        asyncio.run(
            env.run_rollout(
                {"example_id": 0, "info": {"native_task": {"idx": 0}}},
                client=client,
                model="policy",
                sampling_args={"temperature": 1},
            )
        )


def test_swe_bridge_detects_dead_server_before_waiting_for_http(monkeypatch):
    env = object.__new__(NativeSWEEnvironment)
    env.training = False
    env.url = "http://bridge"
    env.log_path = "/run/native.log"
    env.process = SimpleNamespace(poll=lambda: -11)

    class HTTPClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            await asyncio.Event().wait()

    monkeypatch.setattr("examples.qwen30b_swe.bridge_env.httpx.AsyncClient", HTTPClient)
    client = SimpleNamespace(
        api_base_url="http://inference:8000/v1", api_key_var=None, extra_headers={}
    )

    async def check():
        with pytest.raises(RuntimeError, match="code -11.*native.log"):
            await asyncio.wait_for(
                env.run_rollout(
                    {"example_id": 0, "info": {"native_task": {"idx": 0}}},
                    client=client,
                    model="policy",
                    sampling_args={"temperature": 1},
                ),
                timeout=0.5,
            )

    asyncio.run(check())


def test_swe_bridge_retries_disconnected_startup_probe(monkeypatch, tmp_path):
    from unittest.mock import Mock

    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(
        "examples.qwen30b_swe.bridge_env.subprocess.Popen", Mock(return_value=process)
    )
    monkeypatch.setattr("examples.qwen30b_swe.bridge_env.time.sleep", lambda _: None)
    response = Mock()
    client = Mock()
    client.get.side_effect = [httpx.RemoteProtocolError("starting"), response]
    context = Mock()
    context.__enter__ = Mock(return_value=client)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(
        "examples.qwen30b_swe.bridge_env.httpx.Client", Mock(return_value=context)
    )
    env = NativeSWEEnvironment(
        native_python="python",
        env_config="env.json",
        dataset_path="data.jsonl",
        bridge_dir=str(tmp_path),
        base_model="model",
    )
    assert client.get.call_count == 2
    process.terminate.assert_not_called()
    assert env.process is process

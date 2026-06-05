from __future__ import annotations

import json

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.entrypoints.rl_debug import main as debug_main
from wavelet.inference import diagnostics
from wavelet.inference.diagnostics import (
    continuous_batch_probe,
    inference_debug_state,
    make_probe_examples,
    summarize_records,
)


def test_inference_debug_state_exposes_serving_fields() -> None:
    config = RLConfig(
        inference={
            "mode": "vllm_http",
            "http": {"host": "0.0.0.0", "ports": [8100, 8101]},
            "vllm": {
                "server_backend": "openai",
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.8,
            },
        },
        reward={"mode": "reference_match"},
    )

    state = inference_debug_state(config)

    assert state["inference"]["mode"] == "vllm_http"
    assert state["inference"]["http_base_urls"] == [
        "http://0.0.0.0:8100",
        "http://0.0.0.0:8101",
    ]
    assert state["inference"]["tensor_parallel_size"] == 2
    assert state["inference"]["gpu_memory_utilization"] == 0.8


def test_make_probe_examples_are_rl_records() -> None:
    records = make_probe_examples(count=2, prompt="Say ok")

    assert [record.metadata["probe_index"] for record in records] == [0, 1]
    assert records[0].prompt == [{"role": "user", "content": "Say ok #0"}]
    assert records[0].source == "inference_probe"


def test_summarize_records_reports_logprob_and_token_coverage() -> None:
    records = [
        RLExample(
            prompt=[{"role": "user", "content": "a"}],
            completion=[{"role": "assistant", "content": "ok"}],
            advantage=None,
            reward=None,
            input_ids=[1, 2, 3],
            loss_mask=[False, True, True],
            inference_logprobs=[-0.1, -0.2],
        )
    ]

    metrics = summarize_records(records, wall_seconds=2.0)

    assert metrics.records == 1
    assert metrics.model_input_tokens == 3
    assert metrics.completion_tokens == 2
    assert metrics.trainable_tokens == 2
    assert metrics.records_with_inference_logprobs == 1
    assert metrics.trainable_tokens_per_second == 1.0


def test_inference_debug_inspect_outputs_json(capsys) -> None:
    assert debug_main(["inference", "inspect", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["inference"]["mode"] == "vllm_http"
    assert report["sampling"]["temperature"] == 1.0


def test_continuous_batch_probe_routes_across_dp_ranks(monkeypatch) -> None:
    calls = []

    def fake_http_json(
        base_url,
        method,
        path,
        payload=None,
        *,
        headers=None,
        timeout=10.0,
    ):
        calls.append(
            {
                "base_url": base_url,
                "method": method,
                "path": path,
                "headers": headers or {},
                "payload": payload,
                "timeout": timeout,
            }
        )
        return {
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
            }
        }

    monkeypatch.setattr(diagnostics, "_http_json", fake_http_json)
    config = RLConfig(
        model={"name": "test-model"},
        inference={
            "mode": "vllm_http",
            "http": {"host": "127.0.0.1", "port": 8123},
            "vllm": {"data_parallel_size": 3},
        },
        reward={"mode": "reference_match"},
    )

    requests, metrics = continuous_batch_probe(
        config,
        count=6,
        concurrency=3,
        prompt="Say ok",
        max_completion_tokens=16,
        stagger_seconds=0.0,
    )

    assert metrics.requests == 6
    assert metrics.succeeded == 6
    assert metrics.completion_tokens == 30
    assert metrics.total_tokens == 48
    assert metrics.per_rank_requests == {"0": 2, "1": 2, "2": 2}
    assert [call["headers"]["X-data-parallel-rank"] for call in calls] == [
        "0",
        "1",
        "2",
        "0",
        "1",
        "2",
    ]
    assert {request.base_url for request in requests} == {"http://127.0.0.1:8123/v1"}
    assert {call["path"] for call in calls} == {"/chat/completions"}

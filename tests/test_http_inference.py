from dataclasses import replace
from pathlib import Path

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.inference.http import HTTPPolicyInferenceEngine


def _record(index: int) -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": str(index)}],
        completion=[],
        advantage=None,
        reward=None,
        source=str(index),
    )


def test_openai_rollout_rejects_missing_sampled_token_logprobs() -> None:
    engine = HTTPPolicyInferenceEngine(RLConfig())

    with pytest.raises(RuntimeError, match="do not align"):
        engine._openai_completion_logprobs(
            {"logprobs": {"content": [{"logprob": -0.1}]}},
            [10, 11],
        )

    with pytest.raises(RuntimeError, match="missing the sampled-token logprob"):
        engine._openai_completion_logprobs(
            {"logprobs": {"content": [{}]}},
            [10],
        )


def test_round_robin_annotation_restores_input_order() -> None:
    engine = HTTPPolicyInferenceEngine.__new__(HTTPPolicyInferenceEngine)
    engine.base_urls = ["server-0", "server-1", "server-2"]
    records = [_record(index) for index in range(7)]
    chunks: dict[str, list[str]] = {}

    def annotate_chunk(chunk: list[RLExample], base_url: str) -> list[RLExample]:
        chunks[base_url] = [record.source for record in chunk]
        return [
            replace(record, source=f"{record.source}@{base_url}") for record in chunk
        ]

    annotated = engine._annotate_round_robin(records, annotate_chunk)

    assert chunks == {
        "server-0": ["0", "3", "6"],
        "server-1": ["1", "4"],
        "server-2": ["2", "5"],
    }
    assert [record.source for record in annotated] == [
        "0@server-0",
        "1@server-1",
        "2@server-2",
        "3@server-0",
        "4@server-1",
        "5@server-2",
        "6@server-0",
    ]


def test_policy_load_uses_all_server_request_path(tmp_path: Path, monkeypatch) -> None:
    engine = HTTPPolicyInferenceEngine(RLConfig(output_dir=tmp_path))
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def request_all(
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        calls.append((method, path, payload))
        if payload is None:
            return [{"status": path.removeprefix("/")}]
        return [{"policy_step": payload["step"]}]

    monkeypatch.setattr(engine, "_request_all", request_all)

    engine.load_policy(tmp_path / "policy", step=3)

    assert calls == [
        (
            "POST",
            "/load_policy",
            {
                "policy_dir": str(tmp_path / "policy"),
                "step": 3,
                "adapter_name": "policy",
                "load_inplace": True,
            },
        ),
    ]
    assert engine.policy_step == 3


def test_paused_policy_load_resumes_servers_after_failure(
    tmp_path: Path, monkeypatch
) -> None:
    engine = HTTPPolicyInferenceEngine(RLConfig(output_dir=tmp_path))
    paths: list[str] = []

    def request_all(
        _method: str,
        path: str,
        _payload: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        paths.append(path)
        if path == "/load_policy":
            raise RuntimeError("load failed")
        return [{"status": "ok"}]

    monkeypatch.setattr(engine, "_request_all", request_all)

    with pytest.raises(RuntimeError, match="load failed"):
        engine._load_policy_while_generation_paused(
            {"policy_dir": str(tmp_path / "policy"), "step": 3}
        )

    assert paths == ["/pause", "/load_policy", "/resume"]

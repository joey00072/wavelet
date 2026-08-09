from __future__ import annotations

import pytest

from wavelet.configs.rl_config import FrozenModelConfig
from wavelet.orchestrator.reference import (
    VLLMReferenceScorer,
    extract_prompt_logprobs,
)


def test_extract_prompt_logprobs_preserves_token_alignment() -> None:
    values = extract_prompt_logprobs(
        {
            "prompt_logprobs": [
                None,
                {"11": {"logprob": -0.1}},
                {"12": {"logprob": -0.2}},
            ]
        },
        token_ids=[10, 11, 12],
    )

    assert values == pytest.approx([0.0, -0.1, -0.2])


def test_extract_prompt_logprobs_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must align"):
        extract_prompt_logprobs(
            {"prompt_logprobs": [None]},
            token_ids=[10, 11],
        )


def test_extract_prompt_logprobs_rejects_wrong_token() -> None:
    with pytest.raises(ValueError, match="omitted logprob for token 11"):
        extract_prompt_logprobs(
            {"prompt_logprobs": [None, {"99": {"logprob": -0.1}}]},
            token_ids=[10, 11],
        )


def test_extract_prompt_logprobs_rejects_null_after_first_token() -> None:
    with pytest.raises(ValueError, match="null logprob.*position 1"):
        extract_prompt_logprobs(
            {"prompt_logprobs": [None, None]},
            token_ids=[10, 11],
        )


def test_vllm_reference_scorer_uses_token_generate_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"prompt_logprobs":[null,{"11":{"logprob":-0.1}}]}'

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["authorization"] = request.headers.get("Authorization")
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("TEACHER_API_KEY", "secret")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    scorer = VLLMReferenceScorer(
        FrozenModelConfig(
            name="teacher",
            base_url="http://teacher:8001/v1",
            api_key_var="TEACHER_API_KEY",
            timeout_seconds=5.0,
        )
    )

    values = scorer.score([10, 11])

    assert values == pytest.approx([0.0, -0.1])
    assert captured["url"] == "http://teacher:8001/inference/v1/generate"
    assert captured["authorization"] == "Bearer secret"
    assert captured["timeout"] == 5.0
    assert b'"prompt_logprobs": 1' in captured["data"]

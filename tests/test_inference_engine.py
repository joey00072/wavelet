import pytest

from wavelet.configs.rl_config import RLSamplingConfig
from wavelet.inference.engine import (
    VLLMPolicyInferenceEngine,
    extract_vllm_generation_logprobs,
    extract_vllm_prompt_logprobs,
    fit_generation_context,
    openai_payload_to_vllm_kwargs,
    openai_sampling_payload,
    vllm_sampling_kwargs,
)


def test_embedded_vllm_server_rejects_missing_generation_logprobs() -> None:
    engine = VLLMPolicyInferenceEngine.__new__(VLLMPolicyInferenceEngine)
    engine.tokenizer = object()

    with pytest.raises(RuntimeError, match="did not return sampled-token logprobs"):
        engine._openai_logprob_content([1], None)  # noqa: SLF001


@pytest.mark.parametrize("token_key", [7, "7"])
def test_generation_logprobs_preserve_exact_zero(token_key: int | str) -> None:
    assert extract_vllm_generation_logprobs([{token_key: 0.0}], [7]) == [0.0]


@pytest.mark.parametrize("token_key", [7, "7"])
def test_prompt_logprobs_preserve_exact_zero(token_key: int | str) -> None:
    assert extract_vllm_prompt_logprobs(
        [None, {token_key: 0.0}],
        target_ids=[7],
        loss_mask=[True],
    ) == [0.0]


def test_sampling_payloads_preserve_backend_contracts() -> None:
    sampling = RLSamplingConfig(
        do_sample=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.05,
        min_tokens=2,
        max_completion_tokens=32,
        repetition_penalty=1.1,
        seed=17,
        extra_body={"stop": ["done"], "include_stop_str_in_output": True},
    )

    assert openai_sampling_payload(sampling) == {
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.05,
        "min_tokens": 2,
        "max_completion_tokens": 32,
        "repetition_penalty": 1.1,
        "seed": 17,
        "stop": ["done"],
        "include_stop_str_in_output": True,
    }
    assert vllm_sampling_kwargs(sampling, use_generation_logprobs=True) == {
        "n": sampling.num_generations,
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.05,
        "max_tokens": 32,
        "repetition_penalty": 1.1,
        "seed": 17,
        "stop": ["done"],
        "include_stop_str_in_output": True,
        "logprobs": 1,
    }


def test_openai_payload_conversion_preserves_zero_values() -> None:
    assert openai_payload_to_vllm_kwargs(
        {
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 0.0,
            "max_tokens": 8,
        },
        default_max_tokens=16,
    ) == {
        "n": 1,
        "temperature": 0.0,
        "top_p": 0.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 0.0,
        "max_tokens": 8,
    }


def test_context_fit_reserves_completion_room_for_all_backends() -> None:
    prompt_ids, max_completion_tokens = fit_generation_context(
        list(range(10)),
        max_prompt_tokens=8,
        max_model_len=6,
        max_completion_tokens=4,
    )

    assert prompt_ids == [5, 6, 7, 8, 9]
    assert max_completion_tokens == 1


def test_context_fit_uses_available_room_when_budget_is_unspecified() -> None:
    assert fit_generation_context(
        [1, 2, 3],
        max_prompt_tokens=None,
        max_model_len=8,
        max_completion_tokens=None,
    ) == ([1, 2, 3], 5)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.lora.request import LoRARequest

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl import RLExample
from wavelet.inference import engine as engine_module
from wavelet.inference.engine import VLLMPolicyInferenceEngine


class _FakeLLMEngine:
    def __init__(self) -> None:
        self.add_lora_calls: list[LoRARequest] = []

    def add_lora(self, request: LoRARequest) -> bool:
        self.add_lora_calls.append(request)
        return True


class _FakeLLM:
    def __init__(self, request_outputs: list[Any] | None = None) -> None:
        self.llm_engine = _FakeLLMEngine()
        self.generate_calls: list[dict[str, Any]] = []
        self._request_outputs = request_outputs or []

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        sampling_params: Any,
        *,
        use_tqdm: bool,
        lora_request: Any,
    ) -> list[Any]:
        del use_tqdm
        self.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "lora_request": lora_request,
            }
        )
        return self._request_outputs


class _FakeTokenizer:
    def apply_chat_template(self, messages: Any, **_kwargs: Any) -> list[int]:
        del messages
        return [11, 12, 13]


def _record() -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": "hi"}],
        completion=[],
        advantage=None,
        reward=None,
    )


def test_adapter_policy_load_forces_inplace_reload_for_each_snapshot() -> None:
    config = RLConfig(
        policy_transfer={"adapter_name": "policy", "adapter_id": 7},
        lora={"rank": 4, "target_modules": ["q_proj"]},
    )
    engine = VLLMPolicyInferenceEngine(config)
    llm = _FakeLLM()
    engine.llm = llm

    engine._load_adapter_policy(Path("policies/step_1/adapter"), step=1)
    engine._load_adapter_policy(Path("policies/step_2/adapter"), step=2)

    reloads = llm.llm_engine.add_lora_calls
    assert [request.lora_path for request in reloads] == [
        "policies/step_1/adapter",
        "policies/step_2/adapter",
    ]
    assert [request.lora_int_id for request in reloads] == [7, 7]
    assert all(request.load_inplace for request in reloads)
    assert engine._lora_request.lora_int_id == 7
    assert engine._lora_request.lora_path == "policies/step_2/adapter"
    assert engine._lora_request.load_inplace is False


def test_annotate_trains_on_sampled_token_ids_and_logprobs(monkeypatch) -> None:
    monkeypatch.setattr(
        engine_module,
        "_sampling_params_type",
        lambda: lambda **kwargs: kwargs,
    )
    config = RLConfig(inference={"sampling": {"temperature": 0.7}})
    outputs = [
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    text=" hello ",
                    token_ids=[21, 22],
                    logprobs=[{21: -0.5}, {22: -1.5}],
                )
            ]
        )
    ]
    engine = VLLMPolicyInferenceEngine(config)
    engine.llm = _FakeLLM(outputs)
    engine.tokenizer = _FakeTokenizer()

    [annotated] = engine.annotate([_record()])

    # One sampling pass only: no prompt-logprob rescoring fallback ran.
    assert len(engine.llm.generate_calls) == 1
    assert annotated.completion == [{"role": "assistant", "content": "hello"}]
    assert annotated.input_ids == [11, 12, 13, 21]
    assert annotated.target_ids == [12, 13, 21, 22]
    assert annotated.loss_mask == [False, False, True, True]
    assert annotated.inference_logprobs == [-0.5, -1.5]
    assert annotated.temperatures == [0.7, 0.7]


def test_prompt_logprob_scoring_rejects_non_unit_temperature() -> None:
    config = RLConfig(
        inference={
            "sampling": {"temperature": 0.7},
            "vllm": {"use_generation_logprobs": False},
        }
    )
    engine = VLLMPolicyInferenceEngine(config)
    engine.llm = _FakeLLM()
    engine.tokenizer = _FakeTokenizer()

    with pytest.raises(ValueError, match="temperature 1.0"):
        engine._attach_prompt_logprobs([_record()])

    assert engine.llm.generate_calls == []

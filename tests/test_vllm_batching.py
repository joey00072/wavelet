from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.vllm import VLLMPolicyInferenceEngine, _OpenAIBatchRequest


def _request(index: int) -> _OpenAIBatchRequest:
    return _OpenAIBatchRequest(payload={"index": index}, done=threading.Event())


def test_openai_batch_collector_respects_max_size() -> None:
    config = RLConfig(
        inference={
            "vllm": {
                "openai_batch_wait_seconds": 0.0,
                "openai_batch_min_size": 1,
                "openai_batch_max_wait_seconds": 0.0,
                "openai_batch_max_size": 2,
            }
        }
    )
    engine = VLLMPolicyInferenceEngine(config)
    engine._openai_batch = [_request(0), _request(1), _request(2)]

    with engine._openai_batch_condition:
        batch = engine._collect_openai_batch_locked()

    assert [request.payload["index"] for request in batch] == [0, 1]
    assert [request.payload["index"] for request in engine._openai_batch] == [2]


def test_openai_batch_collector_waits_for_min_size() -> None:
    config = RLConfig(
        inference={
            "vllm": {
                "openai_batch_wait_seconds": 0.0,
                "openai_batch_min_size": 2,
                "openai_batch_max_wait_seconds": 0.2,
            }
        }
    )
    engine = VLLMPolicyInferenceEngine(config)
    engine._openai_batch = [_request(0)]

    def append_request() -> None:
        time.sleep(0.02)
        with engine._openai_batch_condition:
            engine._openai_batch.append(_request(1))
            engine._openai_batch_condition.notify()

    thread = threading.Thread(target=append_request)
    thread.start()
    with engine._openai_batch_condition:
        batch = engine._collect_openai_batch_locked()
    thread.join()

    assert [request.payload["index"] for request in batch] == [0, 1]


def test_vllm_policy_load_reuses_stable_lora_request(monkeypatch) -> None:
    request_module = ModuleType("vllm.lora.request")

    class FakeLoRARequest:
        def __init__(
            self,
            lora_name: str,
            lora_int_id: int,
            lora_path: str,
            load_inplace: bool = False,
        ) -> None:
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path
            self.load_inplace = load_inplace

    request_module.LoRARequest = FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.lora", ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_module)

    config = RLConfig(
        policy_transfer={"adapter_name": "policy", "adapter_id": 10},
        lora={"rank": 4, "target_modules": ["q_proj"]},
    )
    engine = VLLMPolicyInferenceEngine(config)
    engine.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(add_lora=lambda _request: True),
    )

    engine._load_adapter_policy(Path("snapshot/adapter"), step=7)

    assert engine._lora_request.lora_name == "policy"
    assert engine._lora_request.lora_int_id == 10
    assert engine._lora_request.lora_path == "snapshot/adapter"
    assert engine._lora_request.load_inplace is False


def test_vllm_setup_passes_fully_sharded_loras(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}
    vllm_module = ModuleType("vllm")

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    vllm_module.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setattr("wavelet.inference.vllm.setup_tokenizer", lambda _: object())
    monkeypatch.setattr(VLLMPolicyInferenceEngine, "_openai_batch_loop", lambda _: None)

    config = RLConfig(
        inference={"vllm": {"fully_sharded_loras": True}},
        lora={"rank": 32, "target_modules": ["q_proj"]},
    )
    engine = VLLMPolicyInferenceEngine(config)

    engine.setup()

    assert captured_kwargs["enable_lora"] is True
    assert captured_kwargs["max_loras"] == 1
    assert captured_kwargs["max_cpu_loras"] == 1
    assert captured_kwargs["fully_sharded_loras"] is True
    assert captured_kwargs["max_lora_rank"] == 32
    assert captured_kwargs["logprobs_mode"] == "processed_logprobs"


def test_vllm_setup_passes_quantized_load_args(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}
    vllm_module = ModuleType("vllm")

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    vllm_module.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setattr("wavelet.inference.vllm.setup_tokenizer", lambda _: object())
    monkeypatch.setattr(VLLMPolicyInferenceEngine, "_openai_batch_loop", lambda _: None)

    config = RLConfig(
        inference={
            "vllm": {
                "quantization": "bitsandbytes",
                "load_format": "bitsandbytes",
            }
        },
    )
    engine = VLLMPolicyInferenceEngine(config)

    engine.setup()

    assert captured_kwargs["quantization"] == "bitsandbytes"
    assert captured_kwargs["load_format"] == "bitsandbytes"

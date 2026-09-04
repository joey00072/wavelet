from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.inference import engine as engine_module
from wavelet.inference.engine import (
    NCCL_READY_MARKER,
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

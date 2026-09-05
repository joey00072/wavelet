from __future__ import annotations

import inspect
from http import HTTPStatus
from logging import CRITICAL
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.exceptions import VLLMValidationError
from vllm.lora.lora_model import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import (
    LRUCacheWorkerLoRAManager,
    WorkerLoRAManager,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.engine.core import DPEngineCoreProc

from wavelet.configs.rl_config import RLConfig
from wavelet.inference import patches as inference_patches
from wavelet.inference import server


def test_load_lora_patch_still_addresses_upstream_request_replacement() -> None:
    source = inspect.getsource(OpenAIServingModels.load_lora_adapter)

    assert "lora_request = LoRARequest(" in source
    assert "self.lora_requests[lora_name].lora_int_id" in source
    assert "lora_request.lora_path = request.lora_path" not in source


class _NoopLock:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args) -> None:
        return None


def _serving_models(existing: LoRARequest, add_lora) -> SimpleNamespace:
    return SimpleNamespace(
        lora_resolver_lock={existing.lora_name: _NoopLock()},
        lora_requests={existing.lora_name: existing},
        lora_id_counter=SimpleNamespace(inc=lambda _amount: 2),
        engine_client=SimpleNamespace(add_lora=add_lora),
        is_base_model=lambda _name: False,
    )


@pytest.mark.anyio
async def test_load_lora_patch_reuses_adapter_id_and_publishes_fresh_request(
    monkeypatch,
) -> None:
    existing = LoRARequest("policy", 1, "/old")
    added: list[LoRARequest] = []

    async def add_lora(request: LoRARequest) -> None:
        added.append(request)

    serving = _serving_models(existing, add_lora)
    monkeypatch.setattr(
        OpenAIServingModels,
        "load_lora_adapter",
        OpenAIServingModels.load_lora_adapter,
    )
    server._patch_load_lora_adapter()

    result = await OpenAIServingModels.load_lora_adapter(
        serving,
        LoadLoRAAdapterRequest(
            lora_name="policy",
            lora_path="/new",
            load_inplace=True,
        ),
    )

    assert result == "Success: LoRA adapter 'policy' added successfully."
    published = serving.lora_requests["policy"]
    assert added == [published]
    assert published is not existing
    assert published.lora_int_id == 1
    assert published.lora_path == "/new"
    assert published.load_inplace is True
    assert existing.lora_path == "/old"
    assert existing.load_inplace is False


@pytest.mark.anyio
async def test_load_lora_patch_keeps_registered_request_when_engine_rejects_path(
    monkeypatch,
) -> None:
    existing = LoRARequest("policy", 1, "/old")

    async def add_lora(request: LoRARequest) -> None:
        raise ValueError(f"No adapter found for {request.lora_path}")

    serving = _serving_models(existing, add_lora)
    monkeypatch.setattr(
        OpenAIServingModels,
        "load_lora_adapter",
        OpenAIServingModels.load_lora_adapter,
    )
    server._patch_load_lora_adapter()

    result = await OpenAIServingModels.load_lora_adapter(
        serving,
        LoadLoRAAdapterRequest(
            lora_name="policy",
            lora_path="/missing",
            load_inplace=True,
        ),
    )

    assert result.error.code == HTTPStatus.NOT_FOUND
    assert result.error.type == "NotFoundError"
    assert serving.lora_requests["policy"] is existing
    assert existing.lora_path == "/old"
    assert existing.load_inplace is False


@pytest.mark.parametrize(
    "message",
    [
        (
            "This model's maximum context length is 4096 tokens. However, your "
            "prompt contains at least 5000 input tokens."
        ),
        "The decoder prompt is too long: request has 5000 input tokens.",
    ],
)
def test_prompt_tokens_from_validation_error_parses_token_count(
    message: str,
) -> None:
    error = VLLMValidationError(message)

    assert server._prompt_tokens_from_validation_error(error) == 5000


def test_lru_patch_still_addresses_path_changes_without_load_inplace() -> None:
    source = inspect.getsource(LRUCacheWorkerLoRAManager.add_adapter)

    assert "lora_request.load_inplace" in source
    assert "lora_request.lora_path" not in source
    assert "_wavelet_loaded_lora_paths" not in source


def test_skip_warning_patch_still_suppresses_upstream_module_warnings() -> None:
    source = inspect.getsource(WorkerLoRAManager._load_adapter)

    if "moe_ep_spec" in inspect.signature(LoRAModel.from_local_checkpoint).parameters:
        assert "expected_lora_modules" in source
        assert "logger.warning_once" not in source
    else:
        assert "logger.warning_once" in source
        assert "is_supported_lora_module" in source


def test_pin_memory_patch_controls_both_vllm_import_sites(monkeypatch) -> None:
    from vllm.lora import lora_model, lora_weights, model_manager

    modules = (lora_model, lora_weights, model_manager)
    for module in modules:
        if hasattr(module, "is_pin_memory_available"):
            original = module.is_pin_memory_available
            monkeypatch.setattr(module, "is_pin_memory_available", original)
        if hasattr(module, "PIN_MEMORY"):
            monkeypatch.setattr(module, "PIN_MEMORY", True)

    server._patch_lora_cpu_pin_memory()

    for module in modules:
        if hasattr(module, "is_pin_memory_available"):
            assert module.is_pin_memory_available() is False
        if hasattr(module, "PIN_MEMORY"):
            assert module.PIN_MEMORY is False


def test_tool_parser_patch_silences_upstream_parser_logger(monkeypatch) -> None:
    from vllm.tool_parsers import hermes_tool_parser

    original_level = hermes_tool_parser.logger.level
    monkeypatch.setattr(hermes_tool_parser.logger, "level", original_level)

    server._patch_noisy_tool_parser_errors()

    assert hermes_tool_parser.logger.level == CRITICAL


def test_vllm_028_owns_fp32_projection_and_dp_pause_protocol() -> None:
    projection_source = inspect.getsource(LogitsProcessor._apply_head)
    pause_source = inspect.getsource(DPEngineCoreProc._has_global_unfinished_reqs)
    resume = DPEngineCoreProc.resume_scheduler

    inference_patches.transformers_v5_compat()

    assert "self.head_dtype" in projection_source
    assert "pending_pause" in pause_source
    assert DPEngineCoreProc.resume_scheduler is resume


def test_chat_token_endpoint_builds_vllm_028_output_parser() -> None:
    calls = []

    class Parser:
        def __init__(self, tokenizer, tools, **kwargs) -> None:
            calls.append((tokenizer, tools, kwargs))

    serving = SimpleNamespace(
        parser_cls=Parser,
        model_config="model-config",
        _effective_chat_template_kwargs=lambda request: {
            "thinking": request.include_reasoning,
        },
    )
    request = SimpleNamespace(tools=["tool"], include_reasoning=True)

    parser, template_kwargs = server.OpenAIServingChatWithTokens._output_parser(
        serving,
        request,
        "tokenizer",
    )

    assert isinstance(parser, Parser)
    assert template_kwargs == {"thinking": True}
    assert calls == [
        (
            "tokenizer",
            ["tool"],
            {
                "chat_template_kwargs": {"thinking": True},
                "model_config": "model-config",
            },
        )
    ]


def test_chat_token_endpoint_tracks_vllm_028_reasoning_state() -> None:
    parser = SimpleNamespace(
        reasoning_parser=object(),
        is_reasoning_end=lambda token_ids: token_ids == [1, 2],
    )
    request = SimpleNamespace(include_reasoning=True, _grammar_from_parser=False)
    serving = SimpleNamespace(parser_cls=object())

    ended = server.OpenAIServingChatWithTokens._reasoning_ended(
        serving,
        request,
        parser,
        [1, 2],
    )

    assert ended is True
    request.include_reasoning = False
    assert (
        server.OpenAIServingChatWithTokens._reasoning_ended(
            serving,
            request,
            parser,
            [],
        )
        is True
    )


def test_build_app_patch_is_required_for_wavelet_router_and_state() -> None:
    source = inspect.getsource(api_server.build_app)
    init_source = inspect.getsource(api_server.init_app_state)

    assert "wavelet" not in source
    assert "openai_serving_chat_with_tokens" not in init_source
    assert "policy_step" not in init_source


@pytest.mark.anyio
async def test_load_policy_routes_lora_policy(monkeypatch) -> None:
    config = RLConfig()
    monkeypatch.setattr(server, "_CONFIG", config)
    captured = {}

    async def fake_load_adapter_policy(raw_request, **kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(server, "_load_adapter_policy", fake_load_adapter_policy)

    result = await server.load_policy(
        {
            "policy_dir": "/tmp/policy",
            "step": 4,
            "load_inplace": True,
        },
        SimpleNamespace(),
    )

    assert result == {"status": "ok"}
    assert captured == {
        "policy_dir": Path("/tmp/policy"),
        "step": 4,
        "load_inplace": True,
        "config": config,
    }


@pytest.mark.anyio
async def test_adapter_load_resets_sticky_load_inplace(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_dir = tmp_path / "policy" / "adapter"
    adapter_dir.mkdir(parents=True)
    registered: dict[str, LoRARequest] = {}

    async def load_lora_adapter(request: LoadLoRAAdapterRequest) -> str:
        registered[request.lora_name] = LoRARequest(
            request.lora_name,
            1,
            request.lora_path,
            load_inplace=request.load_inplace,
        )
        return "ok"

    models = SimpleNamespace(
        lora_requests=registered,
        load_lora_adapter=load_lora_adapter,
    )
    monkeypatch.setattr(server, "_models", lambda _request: models)
    state = SimpleNamespace(
        policy_step=None,
        policy_adapter_name=None,
        policy_adapter_path=None,
    )

    result = await server._load_adapter_policy(
        SimpleNamespace(app=SimpleNamespace(state=state)),
        policy_dir=tmp_path / "policy",
        step=3,
        load_inplace=True,
        config=RLConfig(),
    )

    assert result["policy_step"] == 3
    assert registered["policy"].load_inplace is False


@pytest.mark.anyio
async def test_policy_update_pause_drains_without_clearing_cache() -> None:
    calls = []

    async def pause_generation(*, mode: str, clear_cache: bool) -> None:
        calls.append(("pause", mode, clear_cache))

    state = SimpleNamespace(
        engine_client=SimpleNamespace(pause_generation=pause_generation),
        generation_paused=False,
    )

    result = await server.pause(SimpleNamespace(app=SimpleNamespace(state=state)))

    assert result == {"status": "paused"}
    assert calls == [("pause", "keep", False)]
    assert state.generation_paused is True


@pytest.mark.anyio
async def test_policy_update_resume_releases_generation() -> None:
    calls = []

    async def resume_generation() -> None:
        calls.append("resume")

    state = SimpleNamespace(
        engine_client=SimpleNamespace(resume_generation=resume_generation),
        generation_paused=True,
    )

    result = await server.resume(SimpleNamespace(app=SimpleNamespace(state=state)))

    assert result == {"status": "resumed"}
    assert calls == ["resume"]
    assert state.generation_paused is False

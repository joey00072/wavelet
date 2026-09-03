from __future__ import annotations

import inspect
from logging import CRITICAL
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import (
    LRUCacheWorkerLoRAManager,
    WorkerLoRAManager,
)

from wavelet.configs.rl_config import RLConfig
from wavelet.inference import server


def test_load_lora_patch_still_addresses_upstream_request_replacement() -> None:
    source = inspect.getsource(OpenAIServingModels.load_lora_adapter)

    assert "lora_request = LoRARequest(" in source
    assert "self.lora_requests[lora_name].lora_int_id" in source
    assert "lora_request.lora_path = request.lora_path" not in source


@pytest.mark.anyio
async def test_load_lora_patch_preserves_request_identity_on_path_update(
    monkeypatch,
) -> None:
    class _Lock:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args) -> None:
            return None

    existing = LoRARequest("policy", 1, "/old")
    added: list[LoRARequest] = []

    async def add_lora(request: LoRARequest) -> None:
        added.append(request)

    serving = SimpleNamespace(
        lora_resolver_lock={"policy": _Lock()},
        lora_requests={"policy": existing},
        lora_id_counter=SimpleNamespace(inc=lambda _amount: 2),
        engine_client=SimpleNamespace(add_lora=add_lora),
        is_base_model=lambda _name: False,
    )
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
    assert added == [existing]
    assert serving.lora_requests["policy"] is existing
    assert existing.lora_path == "/new"
    assert existing.load_inplace is True


def test_lru_patch_still_addresses_path_changes_without_load_inplace() -> None:
    source = inspect.getsource(LRUCacheWorkerLoRAManager.add_adapter)

    assert "lora_request.load_inplace" in source
    assert "lora_request.lora_path" not in source
    assert "_wavelet_loaded_lora_paths" not in source


def test_skip_warning_patch_still_suppresses_upstream_module_warnings() -> None:
    source = inspect.getsource(WorkerLoRAManager._load_adapter)

    assert "logger.warning_once" in source
    assert "is_supported_lora_module" in source


def test_pin_memory_patch_controls_both_vllm_import_sites(monkeypatch) -> None:
    from vllm.lora import lora_model, model_manager

    original_lora = lora_model.is_pin_memory_available
    original_manager = model_manager.is_pin_memory_available
    monkeypatch.setattr(lora_model, "is_pin_memory_available", original_lora)
    monkeypatch.setattr(model_manager, "is_pin_memory_available", original_manager)

    server._patch_lora_cpu_pin_memory()

    assert lora_model.is_pin_memory_available() is False
    assert model_manager.is_pin_memory_available() is False


def test_tool_parser_patch_silences_upstream_parser_logger(monkeypatch) -> None:
    from vllm.tool_parsers import hermes_tool_parser

    original_level = hermes_tool_parser.logger.level
    monkeypatch.setattr(hermes_tool_parser.logger, "level", original_level)

    server._patch_noisy_tool_parser_errors()

    assert hermes_tool_parser.logger.level == CRITICAL


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

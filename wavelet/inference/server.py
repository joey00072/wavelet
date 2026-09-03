from __future__ import annotations

import os
import re
import sys
import time
from argparse import Namespace
from http import HTTPStatus
from logging import CRITICAL
from pathlib import Path
from typing import Any

import uvloop
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import init_app_state
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.engine.serving import GenerationError, OpenAIServing
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels,
    create_error_response,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.entrypoints.utils import get_max_tokens, load_aware_call, with_cancellation
from vllm.exceptions import VLLMValidationError
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.diagnostics import inference_debug_state
from wavelet.utils.config import load_config
from wavelet.utils.monitoring import emit_perf


_CONFIG: RLConfig | None = None
router = APIRouter()
CONTEXT_FIT_SAFETY_TOKENS = 16

MODEL_TOOL_CALL_PARSER: dict[str, str] = {
    "zai-org/GLM-4.5": "glm45",
    "zai-org/GLM-4.5-FP8": "glm45",
    "zai-org/GLM-4.5-Base": "glm45",
    "zai-org/GLM-4.5-Air": "glm45",
    "zai-org/GLM-4.5-Air-FP8": "glm45",
    "zai-org/GLM-4.5-Air-Base": "glm45",
    "zai-org/GLM-4.5V": "glm45",
    "zai-org/GLM-4.5V-FP8": "glm45",
    "zai-org/GLM-4.7": "glm47",
    "zai-org/GLM-4.7-FP8": "glm47",
    "zai-org/GLM-4.7-Flash": "glm47",
    "zai-org/GLM-5": "glm47",
    "zai-org/GLM-5-FP8": "glm47",
    "zai-org/GLM-5.1": "glm47",
    "zai-org/GLM-5.1-FP8": "glm47",
    "MiniMaxAI/MiniMax-M2": "minimax_m2",
    "MiniMaxAI/MiniMax-M2.1": "minimax_m2",
    "MiniMaxAI/MiniMax-M2.5": "minimax_m2",
    "PrimeIntellect/INTELLECT-3": "hermes",
    "PrimeIntellect/INTELLECT-3-FP8": "hermes",
    "PrimeIntellect/INTELLECT-3.1": "hermes",
    "Qwen/Qwen3-0.6B": "hermes",
    "Qwen/Qwen3-0.6B-Base": "hermes",
    "Qwen/Qwen3-0.6B-FP8": "hermes",
    "Qwen/Qwen3-1.7B": "hermes",
    "Qwen/Qwen3-1.7B-Base": "hermes",
    "Qwen/Qwen3-1.7B-FP8": "hermes",
    "Qwen/Qwen3-4B": "hermes",
    "Qwen/Qwen3-4B-Base": "hermes",
    "Qwen/Qwen3-4B-FP8": "hermes",
    "Qwen/Qwen3-8B": "hermes",
    "Qwen/Qwen3-8B-Base": "hermes",
    "Qwen/Qwen3-8B-FP8": "hermes",
    "Qwen/Qwen3-14B": "hermes",
    "Qwen/Qwen3-14B-Base": "hermes",
    "Qwen/Qwen3-14B-FP8": "hermes",
    "Qwen/Qwen3-32B": "hermes",
    "Qwen/Qwen3-32B-FP8": "hermes",
    "Qwen/Qwen3-30B-A3B": "hermes",
    "Qwen/Qwen3-30B-A3B-Base": "hermes",
    "Qwen/Qwen3-30B-A3B-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B": "hermes",
    "Qwen/Qwen3-235B-A22B-FP8": "hermes",
    "Qwen/Qwen3-4B-Instruct-2507": "hermes",
    "Qwen/Qwen3-4B-Thinking-2507": "hermes",
    "Qwen/Qwen3-4B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-4B-Thinking-2507-FP8": "hermes",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "hermes",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "hermes",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "hermes",
    "Qwen/Qwen3-235B-A22B-Thinking-2507": "hermes",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Thinking": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8": "hermes",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": "hermes",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": "hermes",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "hermes",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": "hermes",
    "Qwen/Qwen3-Coder-Next": "hermes",
    "Qwen/Qwen3-Coder-Next-Base": "hermes",
    "Qwen/Qwen3-Coder-Next-FP8": "hermes",
    "Qwen/Qwen3.5-0.8B": "qwen3_coder",
    "Qwen/Qwen3.5-0.8B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-2B": "qwen3_coder",
    "Qwen/Qwen3.5-2B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-4B": "qwen3_coder",
    "Qwen/Qwen3.5-4B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-9B": "qwen3_coder",
    "Qwen/Qwen3.5-9B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-27B": "qwen3_coder",
    "Qwen/Qwen3.5-27B-FP8": "qwen3_coder",
    "Qwen/Qwen3.5-35B-A3B": "qwen3_coder",
    "Qwen/Qwen3.5-35B-A3B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-35B-A3B-FP8": "qwen3_coder",
    "Qwen/Qwen3.5-122B-A10B": "qwen3_coder",
    "Qwen/Qwen3.5-122B-A10B-FP8": "qwen3_coder",
    "Qwen/Qwen3.5-397B-A17B": "qwen3_coder",
    "Qwen/Qwen3.5-397B-A17B-FP8": "qwen3_coder",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "qwen3_coder",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "qwen3_coder",
}


def _resolve_tool_call_parser(
    model_name: str, tool_call_parser: str | None
) -> str | None:
    if tool_call_parser == "auto":
        return MODEL_TOOL_CALL_PARSER.get(model_name)
    return tool_call_parser


class ChatCompletionRequestWithTokens(ChatCompletionRequest):
    tokens: list[int] = Field(description="Prompt tokens to use for the request.")


class OpenAIServingChatWithTokens(OpenAIServingChat):
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ):
        fitted_request = request
        for _ in range(4):
            try:
                return await super().create_chat_completion(
                    fitted_request,
                    raw_request,
                )
            except VLLMValidationError as exc:
                next_request = _fit_chat_request_to_context(
                    fitted_request,
                    max_model_len=self.model_config.max_model_len,
                    error=exc,
                )
                if next_request is fitted_request:
                    raise
                fitted_request = next_request
        return await super().create_chat_completion(fitted_request, raw_request)

    async def _render_with_context_fit(
        self,
        request: ChatCompletionRequestWithTokens,
    ):
        for _ in range(4):
            try:
                return request, await self.render_chat_request(request)
            except VLLMValidationError as exc:
                fitted_request = _fit_chat_request_to_context(
                    request,
                    max_model_len=self.model_config.max_model_len,
                    error=exc,
                )
                if fitted_request is request:
                    raise
                request = fitted_request
        return request, await self.render_chat_request(request)

    async def _token_generators(
        self,
        request: ChatCompletionRequestWithTokens,
        engine_prompts,
        *,
        request_id: str,
        raw_request: Request | None,
        lora_request,
        reasoning_parser: ReasoningParser | None,
    ) -> list[Any]:
        data_parallel_rank = self._get_data_parallel_rank(raw_request)
        generators = []
        for index, engine_prompt in enumerate(engine_prompts):
            prompt_token_ids = self._extract_prompt_components(engine_prompt).token_ids
            sub_request_id = (
                request_id if len(engine_prompts) == 1 else f"{request_id}_{index}"
            )
            prompt_len = self._extract_prompt_len(engine_prompt)
            max_model_len = self.model_config.max_model_len
            if prompt_len >= max_model_len:
                raise VLLMValidationError(
                    f"This model's maximum context length is {max_model_len} tokens. "
                    f"However, your request has {prompt_len} input tokens.",
                    parameter="input_tokens",
                    value=prompt_len,
                )
            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                prompt_len,
                self.default_sampling_params,
                self.override_max_tokens,
            )
            if request.use_beam_search:
                sampling_params: SamplingParams | BeamSearchParams = (
                    request.to_beam_search_params(
                        max_tokens,
                        self.default_sampling_params,
                    )
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
            self._log_inputs(
                sub_request_id,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )
            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_prompt,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                reasoning_ended = (
                    reasoning_parser.is_reasoning_end(prompt_token_ids or [])
                    if reasoning_parser
                    else None
                )
                generator = self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=reasoning_ended,
                )
            generators.append(generator)
        return generators

    async def _finish_token_completion(
        self,
        request,
        result_generator,
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        reasoning_parser,
    ):
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )
        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )
        except GenerationError:
            raise
        except ValueError as exc:
            return self.create_error_response(exc)

    async def create_chat_completion_with_tokens(
        self,
        request: ChatCompletionRequestWithTokens,
        raw_request: Request | None = None,
    ):
        request = _fit_chat_request_to_prompt_tokens(
            request,
            max_model_len=self.model_config.max_model_len,
            prompt_tokens=len(request.tokens),
        )
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        reasoning_parser: ReasoningParser | None = None
        try:
            if self.reasoning_parser_cls:
                chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                    request.chat_template_kwargs,
                    self.default_chat_template_kwargs,
                )
                reasoning_parser = self.reasoning_parser_cls(
                    tokenizer,
                    chat_template_kwargs=chat_template_kwargs,
                )
        except RuntimeError as exc:
            return self.create_error_response(str(exc))

        request, rendered = await self._render_with_context_fit(request)
        if isinstance(rendered, ErrorResponse):
            return rendered
        conversation, engine_prompts = rendered
        engine_prompts[0]["prompt_token_ids"] = request.tokens
        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(
                request,
                supports_default_mm_loras=True,
            )
            model_name = self.models.model_name(lora_request)
        except (ValueError, TypeError, RuntimeError) as exc:
            return self.create_error_response(exc)

        try:
            generators = await self._token_generators(
                request,
                engine_prompts,
                request_id=request_id,
                raw_request=raw_request,
                lora_request=lora_request,
                reasoning_parser=reasoning_parser,
            )
        except ValueError as exc:
            return self.create_error_response(exc)

        assert len(generators) == 1
        (result_generator,) = generators
        return await self._finish_token_completion(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
        )


def _fit_chat_request_to_context(
    request: ChatCompletionRequest,
    *,
    max_model_len: int,
    error: VLLMValidationError,
) -> ChatCompletionRequest:
    prompt_tokens = _prompt_tokens_from_validation_error(error)
    if prompt_tokens is None or prompt_tokens >= max_model_len:
        return request
    return _fit_chat_request_to_prompt_tokens(
        request,
        max_model_len=max_model_len,
        prompt_tokens=prompt_tokens,
    )


def _fit_chat_request_to_prompt_tokens(
    request: ChatCompletionRequest,
    *,
    max_model_len: int,
    prompt_tokens: int,
) -> ChatCompletionRequest:
    remaining_tokens = max(
        max_model_len - prompt_tokens - CONTEXT_FIT_SAFETY_TOKENS,
        1,
    )
    requested_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )
    if requested_tokens is None or requested_tokens <= remaining_tokens:
        return request

    updates: dict[str, Any] = {}
    if request.max_completion_tokens is not None:
        updates["max_completion_tokens"] = remaining_tokens
    elif request.max_tokens is not None:
        updates["max_tokens"] = remaining_tokens
    emit_perf(
        "fit_chat_context",
        prompt_tokens=prompt_tokens,
        requested_tokens=requested_tokens,
        max_model_len=max_model_len,
        fitted_tokens=remaining_tokens,
    )
    return request.model_copy(update=updates)


def _prompt_tokens_from_validation_error(error: VLLMValidationError) -> int | None:
    value = getattr(error, "value", None)
    if isinstance(value, int):
        return value
    match = re.search(r"prompt contains at least (\\d+) input tokens", str(error))
    if match is None:
        match = re.search(r"request has (\\d+) input tokens", str(error))
    if match is None:
        return None
    return int(match.group(1))


def _base(request: Request) -> OpenAIServing:
    return request.app.state.openai_serving_tokenization


def _models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def _engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _chat_with_tokens(request: Request) -> OpenAIServingChatWithTokens | None:
    return request.app.state.openai_serving_chat_with_tokens


def _patch_load_lora_adapter() -> None:
    async def patched_load_lora_adapter(
        self: OpenAIServingModels,
        request: LoadLoRAAdapterRequest,
        base_model_name: str | None = None,
    ) -> ErrorResponse | str:
        lora_name = request.lora_name
        async with self.lora_resolver_lock[lora_name]:
            if lora_name in self.lora_requests:
                lora_request = self.lora_requests[lora_name]
                lora_request.lora_path = request.lora_path
            else:
                from vllm.lora.request import LoRARequest

                lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_int_id=self.lora_id_counter.inc(1),
                    lora_path=request.lora_path,
                    load_inplace=request.load_inplace,
                )
            lora_request.load_inplace = request.load_inplace
            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as exc:
                error_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                if "No adapter found" in str(exc):
                    error_type = "NotFoundError"
                    status_code = HTTPStatus.NOT_FOUND
                return create_error_response(
                    message=str(exc),
                    err_type=error_type,
                    status_code=status_code,
                )
            self.lora_requests[lora_name] = lora_request
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    OpenAIServingModels.load_lora_adapter = patched_load_lora_adapter


def _patch_lru_cache_worker_lora_manager() -> None:
    from vllm.lora.request import LoRARequest
    from vllm.lora.worker_manager import (
        LRUCacheLoRAModelManager,
        LRUCacheWorkerLoRAManager,
    )

    def patched_apply_adapters(
        self: LRUCacheWorkerLoRAManager,
        lora_requests: set[LoRARequest],
    ) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests
            if lora_request
        }
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots})."
            )
        for lora in loras_map.values():
            self.add_adapter(lora, force_load=False)

    def patched_add_adapter(
        self: LRUCacheWorkerLoRAManager,
        lora_request: LoRARequest,
        force_load: bool = True,
    ) -> bool:
        started_at = time.perf_counter()
        loaded_paths = getattr(self, "_wavelet_loaded_lora_paths", None)
        if loaded_paths is None:
            loaded_paths = {}
            self._wavelet_loaded_lora_paths = loaded_paths
        loaded_path = loaded_paths.get(lora_request.lora_int_id)
        should_load = (
            lora_request.lora_int_id not in self.list_adapters()
            or force_load
            or (loaded_path is not None and loaded_path != lora_request.lora_path)
        )
        if should_load:
            load_started_at = time.perf_counter()
            lora = self._load_adapter(lora_request)
            load_elapsed = time.perf_counter() - load_started_at

            self._adapter_manager.remove_adapter(lora.id)

            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            add_started_at = time.perf_counter()
            loaded = self._adapter_manager.add_adapter(lora)
            add_elapsed = time.perf_counter() - add_started_at
            if loaded:
                loaded_paths[lora_request.lora_int_id] = lora_request.lora_path
        else:
            load_elapsed = 0.0
            add_elapsed = 0.0
            loaded = (
                self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
            )
        activate_started_at = time.perf_counter()
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        _log_lora_add_adapter_perf(
            lora_request,
            lora_id=lora_request.lora_int_id,
            mode="load" if should_load else "touch",
            load_elapsed=load_elapsed,
            add_elapsed=add_elapsed,
            activate_elapsed=time.perf_counter() - activate_started_at,
            total_elapsed=time.perf_counter() - started_at,
        )
        return loaded

    LRUCacheWorkerLoRAManager._apply_adapters = patched_apply_adapters
    LRUCacheWorkerLoRAManager.add_adapter = patched_add_adapter


def _log_lora_add_adapter_perf(
    lora_request: Any,
    *,
    lora_id: int,
    mode: str,
    load_elapsed: float,
    add_elapsed: float,
    activate_elapsed: float,
    total_elapsed: float,
) -> None:
    if lora_request.lora_name != "policy" or mode == "touch":
        return
    emit_perf(
        "lora_add_adapter",
        force=True,
        name=lora_request.lora_name,
        id=lora_id,
        mode=mode,
        load=load_elapsed,
        add=add_elapsed,
        activate=activate_elapsed,
        total=total_elapsed,
    )


def _patch_lora_cpu_pin_memory() -> None:
    import vllm.lora.lora_model as lora_model
    import vllm.lora.model_manager as model_manager

    def pin_memory_unavailable() -> bool:
        return False

    lora_model.is_pin_memory_available = pin_memory_unavailable
    model_manager.is_pin_memory_available = pin_memory_unavailable


def _patch_noisy_tool_parser_errors() -> None:
    try:
        import vllm.tool_parsers.hermes_tool_parser as hermes_tool_parser
    except ImportError:
        return

    hermes_tool_parser.logger.setLevel(CRITICAL)


def _replace_active_adapter_inplace(adapter_manager: Any, lora: Any) -> bool:
    try:
        index = adapter_manager.lora_index_to_id.index(lora.id)
    except ValueError:
        return False

    adapter_manager._create_merged_loras_inplace(lora)
    adapter_manager._registered_adapters[lora.id] = lora
    adapter_manager._active_adapters[lora.id] = None
    for module_name, module in adapter_manager.modules.items():
        module_lora = adapter_manager._get_lora_layer_weights(lora, module_name)
        if not module_lora:
            module.reset_lora(index)
            continue
        module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
    return True


def _patch_skip_lora_module_warnings() -> None:
    from vllm.exceptions import LoRAAdapterNotFoundError
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.lora.request import LoRARequest
    from vllm.lora.utils import get_adapter_absolute_path
    from vllm.lora.worker_manager import WorkerLoRAManager

    def patched_load_adapter(
        self: WorkerLoRAManager,
        lora_request: LoRARequest,
    ) -> LoRAModel:
        try:
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_list: list[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_list.extend(packed_modules_mapping[module])
                else:
                    expected_lora_list.append(module)
                if module == "experts":
                    expected_lora_list.append(module)
            expected_lora_modules = set(expected_lora_list)
            lora_path = get_adapter_absolute_path(lora_request.lora_path)

            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                self.max_position_embeddings,
                lora_request.tensorizer_config_dict,
            )
            peft_helper.validate_legal(self.lora_config)

            model = self._adapter_manager.model
            weights_mapper = getattr(model, "hf_to_vllm_mapper", None)
            skip_prefixes = getattr(model, "lora_skip_prefixes", None)

            return self._lora_model_cls.from_local_checkpoint(
                lora_path,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                weights_mapper=weights_mapper,
                skip_prefixes=skip_prefixes,
            )
        except FileNotFoundError as exc:
            raise LoRAAdapterNotFoundError(
                lora_request.lora_name,
                lora_request.lora_path,
            ) from exc

    WorkerLoRAManager._load_adapter = patched_load_adapter


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "policy_step": getattr(request.app.state, "policy_step", None),
        "asleep": getattr(request.app.state, "asleep", False),
    }


@router.get("/debug/state")
async def debug_state(request: Request) -> dict[str, Any]:
    if _CONFIG is None:
        raise RuntimeError("Wavelet vLLM OpenAI server config was not initialized.")
    state = inference_debug_state(_CONFIG)
    state["runtime"] = {
        "policy_step": getattr(request.app.state, "policy_step", None),
        "policy_adapter_name": getattr(request.app.state, "policy_adapter_name", None),
        "policy_adapter_path": getattr(request.app.state, "policy_adapter_path", None),
        "policy_weight_path": getattr(request.app.state, "policy_weight_path", None),
        "generation_paused": getattr(
            request.app.state, "generation_paused", False
        ),
        "asleep": getattr(request.app.state, "asleep", False),
    }
    return state


@router.post("/pause")
async def pause(raw_request: Request) -> dict[str, str]:
    """Drain active requests and hold new generation during a policy update."""
    client = _engine_client(raw_request)
    if not hasattr(client, "pause_generation"):
        raise RuntimeError(
            "This vLLM engine does not support safe policy updates because "
            "pause_generation is unavailable."
        )
    await client.pause_generation(mode="keep", clear_cache=False)
    raw_request.app.state.generation_paused = True
    return {"status": "paused"}


@router.post("/resume")
async def resume(raw_request: Request) -> dict[str, str]:
    """Allow generation after a policy update transaction."""
    client = _engine_client(raw_request)
    if not hasattr(client, "resume_generation"):
        raise RuntimeError(
            "This vLLM engine does not support safe policy updates because "
            "resume_generation is unavailable."
        )
    await client.resume_generation()
    raw_request.app.state.generation_paused = False
    return {"status": "resumed"}


@router.post("/sleep")
async def sleep(payload: dict[str, Any], raw_request: Request) -> dict[str, Any]:
    level = int(payload.get("level", 1))
    client = _engine_client(raw_request)
    if hasattr(client, "reset_prefix_cache"):
        await client.reset_prefix_cache()
    if hasattr(client, "reset_mm_cache"):
        await client.reset_mm_cache()
    if hasattr(client, "sleep"):
        await client.sleep(level=level)
        status = "slept"
    elif hasattr(client, "pause_generation"):
        await client.pause_generation(mode="keep", clear_cache=True)
        status = "paused"
    else:
        try:
            await client.collective_rpc("sleep", kwargs={"level": level})
        except TypeError:
            await client.collective_rpc("sleep", args=())
        status = "slept"
    raw_request.app.state.asleep = True
    return {"status": status}


@router.post("/wake")
async def wake(payload: dict[str, Any], raw_request: Request) -> dict[str, Any]:
    tags = payload.get("tags")
    client = _engine_client(raw_request)
    if hasattr(client, "wake_up"):
        kwargs = {"tags": tags} if tags is not None else {}
        await client.wake_up(**kwargs)
        status = "woke"
    elif hasattr(client, "resume_generation"):
        await client.resume_generation()
        status = "resumed"
    else:
        kwargs = {"tags": tags} if tags is not None else {}
        try:
            await client.collective_rpc("wake_up", kwargs=kwargs)
        except TypeError:
            await client.collective_rpc("wake_up", args=())
        status = "woke"
    raw_request.app.state.asleep = False
    return {"status": status}


@router.post("/load_policy")
async def load_policy(payload: dict[str, Any], raw_request: Request):
    if _CONFIG is None:
        raise RuntimeError("Wavelet vLLM OpenAI server config was not initialized.")
    policy_dir = Path(payload["policy_dir"])
    step = int(payload["step"])
    if _CONFIG.lora is None:
        return await _load_full_model_policy(
            raw_request,
            policy_dir=policy_dir,
            step=step,
            config=_CONFIG,
        )
    return await _load_adapter_policy(
        raw_request,
        policy_dir=policy_dir,
        step=step,
        load_inplace=bool(payload.get("load_inplace", False)),
        config=_CONFIG,
    )


async def _load_full_model_policy(
    raw_request: Request,
    *,
    policy_dir: Path,
    step: int,
    config: RLConfig,
) -> dict[str, Any]:
    weight_dir = policy_dir
    if config.policy_transfer.type != "nccl" and (policy_dir / "model").exists():
        weight_dir = policy_dir / "model"
    weight_path = str(weight_dir.resolve())
    response = {"status": "ok", "policy_step": step, "weight_path": weight_path}
    unchanged = (
        getattr(raw_request.app.state, "policy_step", None) == step
        and getattr(raw_request.app.state, "policy_weight_path", None) == weight_path
    )
    if step == 0 or unchanged:
        raw_request.app.state.policy_step = step
        raw_request.app.state.policy_weight_path = weight_path
        return response

    client = _engine_client(raw_request)
    if config.policy_transfer.type == "nccl":
        await client.collective_rpc(
            "init_broadcaster",
            args=(
                config.policy_transfer.nccl_host,
                config.policy_transfer.nccl_port,
                config.policy_transfer.nccl_rank_offset,
                config.policy_transfer.nccl_inference_world_size,
                config.policy_transfer.nccl_timeout_seconds,
            ),
        )
    await client.collective_rpc("update_weights_from_path", args=(weight_path,))
    raw_request.app.state.policy_step = step
    raw_request.app.state.policy_weight_path = weight_path
    return response


async def _load_adapter_policy(
    raw_request: Request,
    *,
    policy_dir: Path,
    step: int,
    load_inplace: bool,
    config: RLConfig,
):
    adapter_name = config.policy_transfer.adapter_name
    adapter_dir = policy_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Policy adapter not found at {adapter_dir}.")

    adapter_path = str(adapter_dir.resolve())
    if (
        getattr(raw_request.app.state, "policy_step", None) == step
        and getattr(raw_request.app.state, "policy_adapter_name", None) == adapter_name
        and getattr(raw_request.app.state, "policy_adapter_path", None) == adapter_path
    ):
        return {
            "status": "ok",
            "policy_step": step,
            "adapter_name": adapter_name,
        }
    response = await _models(raw_request).load_lora_adapter(
        LoadLoRAAdapterRequest(
            lora_name=adapter_name,
            lora_path=adapter_path,
            load_inplace=load_inplace,
        )
    )
    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(),
            status_code=response.error.code,
        )
    raw_request.app.state.policy_step = step
    raw_request.app.state.policy_adapter_name = adapter_name
    raw_request.app.state.policy_adapter_path = adapter_path
    return {
        "status": "ok",
        "policy_step": step,
        "adapter_name": adapter_name,
    }


@router.post("/init_broadcaster")
async def init_broadcaster(payload: dict[str, Any], raw_request: Request):
    if _CONFIG is None:
        raise RuntimeError("Wavelet vLLM OpenAI server config was not initialized.")
    init_info = payload.get("init_info", payload)
    await _engine_client(raw_request).collective_rpc(
        "init_broadcaster",
        args=(
            init_info.get("host", _CONFIG.policy_transfer.nccl_host),
            int(init_info.get("port", _CONFIG.policy_transfer.nccl_port)),
            int(
                init_info.get(
                    "rank_offset",
                    _CONFIG.policy_transfer.nccl_rank_offset,
                )
            ),
            int(
                init_info.get(
                    "inference_world_size",
                    _CONFIG.policy_transfer.nccl_inference_world_size,
                )
            ),
            int(
                init_info.get(
                    "timeout",
                    _CONFIG.policy_transfer.nccl_timeout_seconds,
                )
            ),
        ),
    )
    return {"status": "ok"}


@router.post(
    "/v1/chat/completions/tokens",
    dependencies=[Depends(validate_json_request)],
)
@router.post(
    "/chat/completions/tokens",
    dependencies=[Depends(validate_json_request)],
)
@with_cancellation
@load_aware_call
async def chat_completions_tokens(
    request: ChatCompletionRequestWithTokens,
    raw_request: Request,
):
    handler = _chat_with_tokens(raw_request)
    if handler is None:
        return _base(raw_request).create_error_response(
            message="The model does not support Chat Completions API"
        )
    generator = await handler.create_chat_completion_with_tokens(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.error.code,
        )
    if isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())
    return StreamingResponse(content=generator, media_type="text/event-stream")


async def custom_init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple,
) -> None:
    await init_app_state(engine_client, state, args, supported_tasks)
    state.policy_step = None
    state.policy_adapter_name = None
    state.policy_adapter_path = None
    state.policy_weight_path = None
    state.generation_paused = False
    if "generate" in supported_tasks and state.openai_serving_chat is not None:
        serving_chat = object.__new__(OpenAIServingChatWithTokens)
        serving_chat.__dict__.update(state.openai_serving_chat.__dict__)
        state.openai_serving_chat = serving_chat
        state.openai_serving_chat_with_tokens = serving_chat
    else:
        state.openai_serving_chat_with_tokens = None


def _patch_build_app() -> None:
    import vllm.entrypoints.openai.api_server as api_server

    original_build_app = api_server.build_app

    def custom_build_app(args: Namespace, supported_tasks: tuple, model_config=None):
        app = original_build_app(args, supported_tasks, model_config)
        app.include_router(router)
        return app

    api_server.init_app_state = custom_init_app_state
    api_server.build_app = custom_build_app


def _base_serve_argv(config: RLConfig) -> list[str]:
    vllm_config = config.inference.vllm
    return [
        "--host",
        config.inference.http.host,
        "--port",
        str(config.inference.http.port),
        "--model",
        config.model.name,
        "--dtype",
        vllm_config.dtype or config.model.torch_dtype,
        "--tensor-parallel-size",
        str(vllm_config.tensor_parallel_size),
        "--data-parallel-size",
        str(vllm_config.data_parallel_size),
        "--gpu-memory-utilization",
        str(vllm_config.gpu_memory_utilization),
        "--max-model-len",
        str(vllm_config.max_model_len or config.data.seq_len + 1),
        "--logprobs-mode",
        "processed_logprobs",
        "--generation-config",
        "vllm",
        "--no-enable-log-requests",
    ]


def _append_optional_serve_args(argv: list[str], config: RLConfig) -> None:
    vllm_config = config.inference.vllm
    if vllm_config.quantization is not None:
        argv.extend(["--quantization", vllm_config.quantization])
    if vllm_config.load_format is not None:
        argv.extend(["--load-format", vllm_config.load_format])
    if vllm_config.data_parallel_size_local is not None:
        argv.extend(
            [
                "--data-parallel-size-local",
                str(vllm_config.data_parallel_size_local),
            ]
        )
    if vllm_config.data_parallel_rpc_port is not None:
        argv.extend(
            [
                "--data-parallel-rpc-port",
                str(vllm_config.data_parallel_rpc_port),
            ]
        )
    if config.model.trust_remote_code or vllm_config.trust_remote_code:
        argv.append("--trust-remote-code")
    if config.model.chat_template is not None:
        argv.extend(["--chat-template", config.model.chat_template])


def _append_parser_serve_args(argv: list[str], config: RLConfig) -> None:
    vllm_config = config.inference.vllm
    tool_call_parser = _resolve_tool_call_parser(
        config.model.name,
        vllm_config.tool_call_parser,
    )
    if tool_call_parser is not None:
        argv.extend(["--tool-call-parser", tool_call_parser])
        argv.append("--enable-auto-tool-choice")
    if vllm_config.reasoning_parser is not None:
        argv.extend(["--reasoning-parser", vllm_config.reasoning_parser])
    if vllm_config.enforce_eager:
        argv.append("--enforce-eager")
    if config.launcher.mode == "colocate_sleep":
        argv.append("--enable-sleep-mode")


def _append_lora_serve_args(argv: list[str], config: RLConfig) -> None:
    if config.lora is None:
        return
    vllm_config = config.inference.vllm
    max_lora_rank = vllm_config.max_lora_rank or config.lora.rank
    argv.extend(
        [
            "--enable-lora",
            "--max-loras",
            "1",
            "--max-cpu-loras",
            "1",
            "--max-lora-rank",
            str(max_lora_rank),
        ]
    )
    if vllm_config.fully_sharded_loras:
        argv.append("--fully-sharded-loras")


def _serve_argv(config: RLConfig) -> list[str]:
    """Build vLLM server arguments without initializing the GPU platform."""
    argv = _base_serve_argv(config)
    _append_optional_serve_args(argv, config)
    _append_parser_serve_args(argv, config)
    _append_lora_serve_args(argv, config)
    worker_extension_cls = (
        "wavelet.inference.vllm_weight_update.NCCLWeightUpdateWorker"
        if config.policy_transfer.type == "nccl"
        else "wavelet.inference.vllm_weight_update.FileSystemWeightUpdateWorker"
    )
    argv.extend(["--worker-extension-cls", worker_extension_cls])
    return argv


def _serve_args(config: RLConfig) -> Namespace:
    argv = _serve_argv(config)

    parser = FlexibleArgumentParser(
        description="Wavelet vLLM OpenAI-compatible RL server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(argv)
    assert args is not None
    validate_parsed_serve_args(args)
    return args


def main(argv: list[str] | None = None) -> int:
    global _CONFIG
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    _CONFIG = config
    from wavelet.inference.patches import transformers_v5_compat

    transformers_v5_compat()
    os.environ.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True")
    _patch_load_lora_adapter()
    _patch_lru_cache_worker_lora_manager()
    _patch_skip_lora_module_warnings()
    _patch_lora_cpu_pin_memory()
    _patch_noisy_tool_parser_errors()
    _patch_build_app()

    from vllm.entrypoints.openai.api_server import run_server

    uvloop.run(run_server(_serve_args(config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

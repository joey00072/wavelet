from __future__ import annotations

import os
import sys
from argparse import Namespace
from http import HTTPStatus
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
from wavelet.utils.config import load_config


_CONFIG: RLConfig | None = None
router = APIRouter()


class ChatCompletionRequestWithTokens(ChatCompletionRequest):
    tokens: list[int] = Field(description="Prompt tokens to use for the request.")


class OpenAIServingChatWithTokens(OpenAIServingChat):
    async def create_chat_completion_with_tokens(
        self,
        request: ChatCompletionRequestWithTokens,
        raw_request: Request | None = None,
    ):
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

        rendered = await self.render_chat_request(request)
        if isinstance(rendered, ErrorResponse):
            return rendered
        conversation, engine_prompts = rendered
        engine_prompts[0]["prompt_token_ids"] = request.tokens
        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
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

        data_parallel_rank = self._get_data_parallel_rank(raw_request)
        generators = []
        try:
            for index, engine_prompt in enumerate(engine_prompts):
                prompt_token_ids = self._extract_prompt_components(
                    engine_prompt
                ).token_ids
                sub_request_id = (
                    request_id if len(engine_prompts) == 1 else f"{request_id}_{index}"
                )
                prompt_len = self._extract_prompt_len(engine_prompt)
                max_model_len = self.model_config.max_model_len
                if prompt_len >= max_model_len:
                    raise VLLMValidationError(
                        f"This model's maximum context length is {max_model_len} "
                        f"tokens. However, your request has {prompt_len} input tokens.",
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
        except ValueError as exc:
            return self.create_error_response(exc)

        assert len(generators) == 1
        (result_generator,) = generators
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
                )
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
        if lora_request.lora_int_id not in self.list_adapters() or force_load:
            lora = self._load_adapter(lora_request)
            self._adapter_manager.remove_adapter(lora.id)

            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            loaded = (
                self._adapter_manager.get_adapter(lora_request.lora_int_id)
                is not None
            )
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded

    LRUCacheWorkerLoRAManager._apply_adapters = patched_apply_adapters
    LRUCacheWorkerLoRAManager.add_adapter = patched_add_adapter


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "policy_step": getattr(request.app.state, "policy_step", None),
        "asleep": getattr(request.app.state, "asleep", False),
    }


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
    adapter_name = str(
        payload.get("adapter_name") or _CONFIG.policy_transfer.adapter_name
    )
    adapter_dir = policy_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Policy adapter not found at {adapter_dir}.")
    response = await _models(raw_request).load_lora_adapter(
        LoadLoRAAdapterRequest(
            lora_name=adapter_name,
            lora_path=str(adapter_dir),
        )
    )
    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(),
            status_code=response.error.code,
        )
    raw_request.app.state.policy_step = step
    return {"status": "ok", "policy_step": step, "adapter_name": adapter_name}


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


def _serve_args(config: RLConfig) -> Namespace:
    vllm_config = config.inference.vllm
    max_lora_rank = vllm_config.max_lora_rank or (
        config.lora.rank if config.lora is not None else 1
    )
    argv = [
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
        "--max-loras",
        str(vllm_config.max_loras),
        "--max-cpu-loras",
        str(vllm_config.max_cpu_loras),
        "--max-lora-rank",
        str(max_lora_rank),
        "--logprobs-mode",
        "processed_logprobs",
        "--generation-config",
        "vllm",
        "--no-enable-log-requests",
    ]
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
    if vllm_config.enforce_eager:
        argv.append("--enforce-eager")
    if config.launcher.mode == "colocate_sleep":
        argv.append("--enable-sleep-mode")
    if config.lora is not None:
        argv.append("--enable-lora")

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
    os.environ.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True")
    _patch_load_lora_adapter()
    _patch_lru_cache_worker_lora_manager()
    _patch_build_app()

    from vllm.entrypoints.openai.api_server import run_server

    uvloop.run(run_server(_serve_args(config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

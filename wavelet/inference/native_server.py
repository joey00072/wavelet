from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl import (
    rl_examples_from_payload,
    rl_examples_to_payload,
)
from wavelet.debug import inference_debug_state
from wavelet.inference.engine import VLLMPolicyInferenceEngine
from wavelet.transport.policy import nccl_world_size
from wavelet.utils.config import load_config


class _GenerationGate:
    """Drain active generation and block new work during policy replacement."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paused = False
        self._active = 0

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    def pause(self) -> None:
        with self._condition:
            self._paused = True
            while self._active:
                self._condition.wait()

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    @contextmanager
    def generation(self):
        with self._condition:
            while self._paused:
                self._condition.wait()
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()


def _lifespan(config: RLConfig, engine: Any):
    @asynccontextmanager
    async def lifespan(_app: Any):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if config.lora is not None:
            os.environ.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True")
        engine.setup()
        if config.policy_transfer.type == "nccl":
            engine.init_weight_transfer(
                {
                    "master_address": config.policy_transfer.nccl_host,
                    "master_port": config.policy_transfer.nccl_port,
                    "rank_offset": config.policy_transfer.nccl_rank_offset,
                    "world_size": nccl_world_size(
                        config.policy_transfer.nccl_inference_world_size
                    ),
                }
            )
        try:
            yield
        finally:
            engine.close()

    return lifespan


def _register_status_routes(
    app: Any,
    config: RLConfig,
    engine: Any,
    generation_gate: _GenerationGate,
) -> None:
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "policy_step": engine.policy_step,
            "generation_paused": generation_gate.paused,
        }

    @app.get("/debug/state")
    def debug_state() -> dict[str, Any]:
        state = inference_debug_state(config)
        state["runtime"] = {
            "policy_step": engine.policy_step,
            "policy_adapter_loaded": getattr(engine, "_lora_request", None) is not None,
            "generation_paused": generation_gate.paused,
        }
        return state

    @app.post("/pause")
    def pause() -> dict[str, str]:
        generation_gate.pause()
        return {"status": "paused"}

    @app.post("/resume")
    def resume() -> dict[str, str]:
        generation_gate.resume()
        return {"status": "resumed"}


def _register_memory_routes(app: Any, engine: Any) -> None:
    @app.post("/sleep")
    def sleep(payload: dict[str, Any] | None = None) -> dict[str, str]:
        del payload
        engine.sleep()
        return {"status": "slept"}

    @app.post("/wake")
    def wake(payload: dict[str, Any] | None = None) -> dict[str, str]:
        tags = None if payload is None else payload.get("tags")
        engine.wake(tags=tags)
        return {"status": "woke"}


def _register_policy_routes(
    app: Any,
    engine: Any,
    http_exception: type[Exception],
) -> None:
    @app.post("/load_policy")
    def load_policy(payload: dict[str, Any]) -> dict[str, Any]:
        policy_dir = Path(payload["policy_dir"])
        step = int(payload["step"])
        engine.load_policy(policy_dir, step=step)
        return {"status": "ok", "policy_step": engine.policy_step}

    @app.post("/load_lora_adapter")
    def load_lora_adapter(payload: dict[str, Any]) -> dict[str, Any]:
        raw_adapter_path = payload.get("lora_path") or payload.get("adapter_path")
        if raw_adapter_path is None:
            raise http_exception(
                status_code=400,
                detail="Expected lora_path or adapter_path.",
            )
        adapter_path = Path(raw_adapter_path)
        step = int(payload.get("step", engine.policy_step or 0))
        engine._load_adapter_policy(adapter_path, step=step)
        engine.policy_step = step
        return {"status": "ok", "policy_step": engine.policy_step}

    @app.post("/update_weights")
    def update_weights(payload: dict[str, Any]) -> dict[str, Any]:
        update_info = payload.get("update_info")
        if update_info is not None:
            engine.update_weights(update_info)
            step = payload.get("step")
            if step is not None:
                engine.policy_step = int(step)
            return {"status": "ok", "policy_step": engine.policy_step}

        policy_dir = payload.get("policy_dir") or payload.get("weight_dir")
        if policy_dir is None:
            raise http_exception(
                status_code=400,
                detail="Expected update_info, policy_dir, or weight_dir.",
            )
        step = int(payload.get("step", engine.policy_step or 0))
        engine.load_policy(Path(policy_dir), step=step)
        return {"status": "ok", "policy_step": engine.policy_step}

    @app.post("/init_broadcaster")
    def init_broadcaster(payload: dict[str, Any]) -> dict[str, Any]:
        init_info = payload.get("init_info", payload)
        engine.init_weight_transfer(init_info)
        return {"status": "ok"}


def _register_inference_routes(
    app: Any,
    engine: Any,
    generation_gate: _GenerationGate,
) -> None:
    @app.post("/annotate")
    def annotate(payload: dict[str, Any]) -> dict[str, Any]:
        with generation_gate.generation():
            records = rl_examples_from_payload(payload["records"])
            annotated = engine.annotate(records)
        return {
            "records": rl_examples_to_payload(annotated),
            "policy_step": engine.policy_step,
        }

    @app.post("/v1/chat/completions")
    def openai_chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        with generation_gate.generation():
            return engine.openai_chat_completion(payload)

    @app.post("/v1/chat/completions/tokens")
    @app.post("/chat/completions/tokens")
    def openai_chat_completions_tokens(payload: dict[str, Any]) -> dict[str, Any]:
        with generation_gate.generation():
            return engine.openai_chat_completion(payload)

    @app.post("/v1/tokenize")
    @app.post("/tokenize")
    def tokenize(payload: dict[str, Any]) -> dict[str, Any]:
        return engine.tokenize_messages(payload)


def _build_app(config: RLConfig):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ImportError(
            "The vLLM HTTP server requires FastAPI. Install dependencies with "
            "`uv sync`."
        ) from exc

    engine = VLLMPolicyInferenceEngine(config)
    generation_gate = _GenerationGate()
    app = FastAPI(
        title="Wavelet vLLM RL Server",
        lifespan=_lifespan(config, engine),
    )
    _register_status_routes(app, config, engine, generation_gate)
    _register_memory_routes(app, engine)
    _register_policy_routes(app, engine, HTTPException)
    _register_inference_routes(app, engine, generation_gate)

    return app


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    from wavelet.monitor import setup_config_logger

    setup_config_logger(f"native_inference_{config.inference.http.port}", config)
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The vLLM HTTP server requires uvicorn. Install dependencies with "
            "`uv sync`."
        ) from exc

    app = _build_app(config)
    uvicorn.run(
        app,
        host=config.inference.http.host,
        port=config.inference.http.port,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

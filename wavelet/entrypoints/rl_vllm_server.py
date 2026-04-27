from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.serialization import (
    rl_examples_from_payload,
    rl_examples_to_payload,
)
from wavelet.inference.vllm import VLLMPolicyInferenceEngine
from wavelet.utils.config import load_config


def _server_engine_config(config: RLConfig) -> RLConfig:
    return config.model_copy(
        update={
            "inference": config.inference.model_copy(update={"mode": "vllm"}),
        }
    )


def _build_app(config: RLConfig):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ImportError(
            "The vLLM HTTP server requires FastAPI. Install dependencies with "
            "`uv sync`."
        ) from exc

    engine = VLLMPolicyInferenceEngine(_server_engine_config(config))
    app = FastAPI(title="Wavelet vLLM RL Server")

    @app.on_event("startup")
    def startup() -> None:
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
                    "world_size": (
                        config.policy_transfer.nccl_inference_world_size
                        + config.policy_transfer.nccl_rank_offset
                    ),
                }
            )

    @app.on_event("shutdown")
    def shutdown() -> None:
        engine.close()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "policy_step": engine.policy_step}

    @app.post("/pause")
    def pause() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/resume")
    def resume() -> dict[str, str]:
        return {"status": "ok"}

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
            raise HTTPException(
                status_code=400,
                detail="Expected lora_path or adapter_path.",
            )
        adapter_path = Path(raw_adapter_path)
        step = int(payload.get("step", engine.policy_step or 0))
        engine._load_adapter_policy(adapter_path, step=step)  # noqa: SLF001
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
            raise HTTPException(
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

    @app.post("/annotate")
    def annotate(payload: dict[str, Any]) -> dict[str, Any]:
        records = rl_examples_from_payload(payload["records"])
        annotated = engine.annotate(records)
        return {
            "records": rl_examples_to_payload(annotated),
            "policy_step": engine.policy_step,
        }

    @app.post("/v1/chat/completions")
    def openai_chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        return engine.openai_chat_completion(payload)

    @app.post("/v1/chat/completions/tokens")
    @app.post("/chat/completions/tokens")
    def openai_chat_completions_tokens(payload: dict[str, Any]) -> dict[str, Any]:
        return engine.openai_chat_completion(payload)

    @app.post("/v1/tokenize")
    @app.post("/tokenize")
    def tokenize(payload: dict[str, Any]) -> dict[str, Any]:
        return engine.tokenize_messages(payload)

    return app


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
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

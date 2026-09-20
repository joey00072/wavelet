"""Run the shared SWE harness in an environment with Verifiers v1 installed."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from fastapi import FastAPI, HTTPException, Request
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from verifiers.v1.configs.client import EvalClientConfig, TrainClientConfig

from examples.qwen30b_swe.runtime_compat import install
from examples.qwen30b_swe.trace_adapter import sample_digest, trace_to_output


def main() -> None:
    import uvicorn

    install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = vf.SingleAgentEnvConfig.model_validate(json.loads(args.config.read_text()))
    env = vf.load_environment(config)
    task_cls = type(env.taskset).task_type()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with env.serving():
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/rollout")
    async def rollout(request: Request) -> dict[str, Any]:
        payload = await request.json()
        task = task_cls(
            task_cls.data_type().model_validate(payload["task"]), config.taskset.task
        )
        if env.taskset.system_prompt is not None:
            task = task.with_system_prompt(env.taskset.system_prompt)
        client_type = TrainClientConfig if payload["training"] else EvalClientConfig
        client = client_type(**payload["client"])
        context = vf.ModelContext(
            client=client,
            model=payload["model"],
            sampling=vf.SamplingConfig(**payload["sampling"]),
        )
        (slot,) = env.slots(task)
        operation = asyncio.create_task(env.run_slot(slot, context))
        try:
            while not operation.done():
                if await request.is_disconnected():
                    raise asyncio.CancelledError
                await asyncio.wait({operation}, timeout=1)
            episode = operation.result()
        finally:
            if not operation.done():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{episode.id}.json").write_text(
            episode.model_dump_json(context={"float_decimals": None})
        )
        if not episode.ok or len(episode.traces) != 1:
            raise HTTPException(
                502, f"Native SWE episode {episode.id} failed; inspect its saved trace."
            )
        trace = episode.traces[0]
        output = trace_to_output(
            trace.model_dump(mode="json", context={"float_decimals": None}),
            reward=trace.reward,
        )
        samples = []
        for branch, mask in iter_trainable_branches(trace):
            ids = branch.token_ids
            samples.append(
                {
                    "input_ids": ids[:-1],
                    "target_ids": ids[1:],
                    "loss_mask": mask[1:],
                    "inference_logprobs": branch.logprobs[1:],
                }
            )
        output["native_training_digest"] = sample_digest(samples)
        output["is_truncated"] = trace.is_truncated
        output["native_info"] = trace.info
        return output

    uvicorn.run(
        app,
        fd=args.fd,
        loop="asyncio",
        log_level="warning",
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()

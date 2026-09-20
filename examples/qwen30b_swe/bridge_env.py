"""Legacy Wavelet verifier interface backed by the native SWE environment."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from examples.qwen30b_swe.trace_adapter import sample_digest

if TYPE_CHECKING:
    from datasets import Dataset
    from verifiers import ClientConfig


class NativeSWEEnvironment:
    requires_group_scoring = False
    env_id = "native-swe"
    env_args = None

    def __init__(
        self,
        *,
        native_python: str,
        env_config: str,
        dataset_path: str,
        bridge_dir: str,
        base_model: str,
        training: bool = True,
        renderer: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.base_model = base_model
        self.training = training
        self.renderer = renderer
        directory = Path(bridge_dir)
        directory.mkdir(parents=True, exist_ok=True)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        log_path = self.log_path = directory / f"{Path(env_config).stem}.log"
        with log_path.open("ab") as log:
            self.process = subprocess.Popen(
                [
                    native_python,
                    "-X",
                    "faulthandler",
                    "-m",
                    "examples.qwen30b_swe.native_server",
                    "--config",
                    env_config,
                    "--fd",
                    str(sock.fileno()),
                    "--output-dir",
                    str(directory / Path(env_config).stem),
                ],
                pass_fds=(sock.fileno(),),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        sock.close()
        deadline = time.monotonic() + 120
        try:
            with httpx.Client(timeout=1, trust_env=False) as client:
                while True:
                    if self.process.poll() is not None:
                        raise RuntimeError(
                            f"Native SWE server exited; inspect {log_path}."
                        )
                    try:
                        response = client.get(self.url + "/health")
                        response.raise_for_status()
                        break
                    except (
                        httpx.ConnectError,
                        httpx.ReadTimeout,
                        httpx.RemoteProtocolError,
                    ):
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                f"Native SWE server startup timed out: {log_path}"
                            )
                        time.sleep(0.2)
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=30)
            raise

    def get_eval_dataset(self, n: int) -> Dataset:
        from datasets import Dataset

        rows = []
        with self.dataset_path.open() as handle:
            for line in handle:
                rows.append(json.loads(line)["metadata"]["verifier_example"])
                if len(rows) >= n:
                    break
        return Dataset.from_list(rows)

    async def run_rollout(
        self,
        example: dict[str, Any],
        *,
        client: ClientConfig,
        model: str,
        sampling_args: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        from wavelet.orchestrator.envs import _interleave_output

        sampling = dict(sampling_args)
        sampling["extra_body"] = dict(sampling.get("extra_body") or {})
        sampling["extra_body"].pop("return_token_ids", None)
        sampling.pop("logprobs", None)
        base_url = client.api_base_url.rstrip("/")
        native_client = {
            "base_url": base_url.removesuffix("/v1") if self.training else base_url,
            "api_key_var": client.api_key_var,
            "headers": client.extra_headers or {},
        }
        if self.training:
            native_client["renderer_model_name"] = self.base_model
            if self.renderer is not None:
                native_client["renderer"] = self.renderer
        payload = {
            "task": example["info"]["native_task"],
            "client": native_client,
            "training": self.training,
            "model": model,
            "sampling": sampling,
        }
        async with httpx.AsyncClient(timeout=2400, trust_env=False) as http:
            pending = asyncio.create_task(
                http.post(self.url + "/rollout", json=payload)
            )
            try:
                while not pending.done():
                    code = self.process.poll()
                    if code is not None:
                        raise RuntimeError(
                            f"Native SWE server exited with code {code}; "
                            f"inspect {self.log_path}."
                        )
                    await asyncio.wait({pending}, timeout=1)
                response = pending.result()
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise RuntimeError(
                    f"Native SWE rollout failed (HTTP {response.status_code}): "
                    f"{response.text[:2000]}"
                ) from error
            output = response.json()
        output["example_id"] = example["example_id"]
        output["sampling_args"] = sampling_args
        actual = sample_digest(
            _interleave_output(output, sampling.get("temperature", 1))
        )
        if actual != output.pop("native_training_digest"):
            raise ValueError(
                "Wavelet token masks/logprobs differ from native SWE training samples."
            )
        return output

    async def teardown(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, timeout=40)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait)


def load_environment(**kwargs: Any) -> NativeSWEEnvironment:
    return NativeSWEEnvironment(**kwargs)

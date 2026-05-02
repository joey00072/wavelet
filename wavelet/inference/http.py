from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha1
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.inference.policy import PolicyInferenceEngine, RLInference
from wavelet.inference.serialization import (
    rl_examples_from_payload,
    rl_examples_to_payload,
)
from wavelet.trainer.model import setup_tokenizer


class HTTPPolicyInferenceEngine(PolicyInferenceEngine):
    """HTTP client for a persistent Wavelet vLLM rollout server."""

    def __init__(self, config: RLConfig) -> None:
        super().__init__(config)
        ports = config.inference.http.ports or [config.inference.http.port]
        self.base_urls = [
            f"http://{config.inference.http.host}:{port}"
            for port in ports
        ]
        self.base_url = self.base_urls[0]
        self.tokenizer = None
        self.policy_model_name = config.model.name
        self._policy_cache_root = self._default_policy_cache_root()

    def setup(self) -> None:
        deadline = time.monotonic() + self.config.inference.http.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                for base_url in self.base_urls:
                    self._request("GET", "/health", base_url=base_url)
                if self._uses_openai_rollouts():
                    self.tokenizer = setup_tokenizer(self.config.model)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(
            "Timed out waiting for vLLM HTTP server(s) at "
            f"{', '.join(self.base_urls)}."
        ) from last_error

    def load_policy(self, policy_dir: Path, *, step: int) -> None:
        if self._uses_openai_rollouts() and self.config.lora is not None:
            policy_dir = self._cache_lora_policy_dir(policy_dir, step=step)
        payload: dict[str, Any] = {"policy_dir": str(policy_dir), "step": step}
        if self._uses_openai_rollouts() and self.config.lora is not None:
            payload["adapter_name"] = self.config.policy_transfer.adapter_name
            payload["load_inplace"] = True
        if len(self.base_urls) == 1:
            self._request("POST", "/load_policy", payload, base_url=self.base_url)
        else:
            with ThreadPoolExecutor(max_workers=len(self.base_urls)) as executor:
                futures = [
                    executor.submit(
                        self._request,
                        "POST",
                        "/load_policy",
                        payload,
                        base_url=base_url,
                    )
                    for base_url in self.base_urls
                ]
                for future in futures:
                    future.result()
        self.policy_step = step
        if self.config.lora is not None:
            self.policy_model_name = self.config.policy_transfer.adapter_name

    def _default_policy_cache_root(self) -> Path:
        configured = os.environ.get("WAVELET_POLICY_CACHE_DIR")
        if configured:
            base_dir = Path(configured)
        else:
            shm_dir = Path("/dev/shm")
            base_dir = shm_dir if shm_dir.is_dir() else Path(tempfile.gettempdir())
        output_hash = sha1(
            str(self.config.output_dir.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        return base_dir / f"wavelet-policy-cache-{os.getuid()}" / f"{os.getpid()}-{output_hash}"

    def _cache_lora_policy_dir(self, policy_dir: Path, *, step: int) -> Path:
        adapter_dir = policy_dir / "adapter"
        tensor_path = adapter_dir / "adapter_model.safetensors"
        if not tensor_path.is_file():
            return policy_dir

        cached_policy_dir = self._policy_cache_root / f"step-{step:06d}"
        marker_path = cached_policy_dir / ".complete"
        if marker_path.is_file():
            return cached_policy_dir

        tmp_policy_dir = cached_policy_dir.with_name(
            f"{cached_policy_dir.name}.tmp-{time.monotonic_ns()}"
        )
        try:
            shutil.rmtree(tmp_policy_dir, ignore_errors=True)
            tmp_adapter_dir = tmp_policy_dir / "adapter"
            tmp_adapter_dir.mkdir(parents=True, exist_ok=True)
            for source in adapter_dir.iterdir():
                if source.is_file():
                    shutil.copy2(source, tmp_adapter_dir / source.name)
            (tmp_policy_dir / ".complete").write_text("ok\n", encoding="utf-8")
            if cached_policy_dir.exists():
                shutil.rmtree(cached_policy_dir)
            os.replace(tmp_policy_dir, cached_policy_dir)
            self._prune_policy_cache(keep_steps={step, step - 1})
        except OSError:
            shutil.rmtree(tmp_policy_dir, ignore_errors=True)
            return policy_dir
        return cached_policy_dir

    def _prune_policy_cache(self, *, keep_steps: set[int]) -> None:
        if not self._policy_cache_root.is_dir():
            return
        keep_names = {f"step-{step:06d}" for step in keep_steps if step >= 0}
        for child in self._policy_cache_root.iterdir():
            if child.is_dir() and child.name.startswith("step-") and child.name not in keep_names:
                shutil.rmtree(child, ignore_errors=True)

    def sleep(self) -> None:
        self._request_all("POST", "/sleep", {"level": 1})

    def wake(self, *, tags: list[str] | None = None) -> None:
        payload: dict[str, Any] = {}
        if tags is not None:
            payload["tags"] = tags
        self._request_all("POST", "/wake", payload)

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        if self._uses_openai_rollouts():
            return self._annotate_openai(records)
        if len(self.base_urls) == 1 or len(records) <= 1:
            response = self._request(
                "POST",
                "/annotate",
                {"records": rl_examples_to_payload(records)},
            )
            self.policy_step = response.get("policy_step")
            return rl_examples_from_payload(response["records"])

        chunks: list[tuple[int, list[RLExample]]] = [
            (index, []) for index in range(len(self.base_urls))
        ]
        for index, record in enumerate(records):
            chunks[index % len(chunks)][1].append(record)

        annotated_by_chunk: dict[int, list[RLExample]] = {}
        with ThreadPoolExecutor(max_workers=len(self.base_urls)) as executor:
            futures = {
                executor.submit(
                    self._request,
                    "POST",
                    "/annotate",
                    {"records": rl_examples_to_payload(chunk)},
                    base_url=self.base_urls[index],
                ): index
                for index, chunk in chunks
                if chunk
            }
            for future, index in futures.items():
                response = future.result()
                self.policy_step = response.get("policy_step")
                annotated_by_chunk[index] = rl_examples_from_payload(
                    response["records"]
                )

        positions = {index: 0 for index in annotated_by_chunk}
        merged: list[RLExample] = []
        for index in range(len(records)):
            chunk_index = index % len(self.base_urls)
            position = positions[chunk_index]
            merged.append(annotated_by_chunk[chunk_index][position])
            positions[chunk_index] = position + 1
        return merged

    def _annotate_single_server(self, records: list[RLExample]) -> list[RLExample]:
        response = self._request(
            "POST",
            "/annotate",
            {"records": rl_examples_to_payload(records)},
        )
        self.policy_step = response.get("policy_step")
        return rl_examples_from_payload(response["records"])

    def _annotate_openai(self, records: list[RLExample]) -> list[RLExample]:
        if self.tokenizer is None:
            self.tokenizer = setup_tokenizer(self.config.model)
        if len(records) == 0:
            return []

        chunks: list[tuple[int, list[RLExample]]] = [
            (index, []) for index in range(len(self.base_urls))
        ]
        for index, record in enumerate(records):
            chunks[index % len(chunks)][1].append(record)

        policy_model_name = self.policy_model_name
        annotated_by_chunk: dict[int, list[RLExample]] = {}
        with ThreadPoolExecutor(max_workers=len(self.base_urls)) as executor:
            futures = {
                executor.submit(
                    self._annotate_openai_chunk,
                    chunk,
                    base_url=self.base_urls[index],
                    policy_model_name=policy_model_name,
                ): index
                for index, chunk in chunks
                if chunk
            }
            for future, index in futures.items():
                annotated_by_chunk[index] = future.result()

        positions = {index: 0 for index in annotated_by_chunk}
        merged: list[RLExample] = []
        for index in range(len(records)):
            chunk_index = index % len(self.base_urls)
            position = positions[chunk_index]
            merged.append(annotated_by_chunk[chunk_index][position])
            positions[chunk_index] = position + 1
        return merged

    def _annotate_openai_chunk(
        self,
        records: list[RLExample],
        *,
        base_url: str,
        policy_model_name: str,
    ) -> list[RLExample]:
        annotated: list[RLExample] = []
        for record in records:
            prompt_ids = RLInference(self.config).prompt_token_ids(
                self.tokenizer,
                record,
            )
            max_prompt_tokens = self.config.inference.sampling.max_prompt_tokens
            if max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens:
                prompt_ids = prompt_ids[-max_prompt_tokens:]
            response = self._request(
                "POST",
                "/chat/completions/tokens",
                self._openai_payload(
                    record,
                    prompt_ids,
                    policy_model_name=policy_model_name,
                ),
                base_url=base_url,
            )
            annotated.append(
                self._record_from_openai_response(
                    record,
                    prompt_ids=prompt_ids,
                    response=response,
                )
            )
        return annotated

    def _openai_payload(
        self,
        record: RLExample,
        prompt_ids: list[int],
        *,
        policy_model_name: str | None = None,
    ) -> dict[str, Any]:
        sampling = self.config.inference.sampling
        payload: dict[str, Any] = {
            "model": policy_model_name or self.policy_model_name,
            "messages": record.prompt,
            "tokens": prompt_ids,
            "temperature": sampling.temperature if sampling.do_sample else 0.0,
            "top_p": sampling.top_p,
            "min_p": sampling.min_p,
            "max_completion_tokens": sampling.max_completion_tokens,
            "logprobs": True,
            "return_token_ids": True,
        }
        if sampling.top_k != 0:
            payload["top_k"] = sampling.top_k
        if sampling.seed is not None:
            payload["seed"] = sampling.seed
        if sampling.repetition_penalty != 1.0:
            payload["repetition_penalty"] = sampling.repetition_penalty
        if record.tools is not None:
            payload["tools"] = record.tools
        if record.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = record.chat_template_kwargs
        payload.update(sampling.extra_body)
        payload["return_token_ids"] = True
        return payload

    def _record_from_openai_response(
        self,
        record: RLExample,
        *,
        prompt_ids: list[int],
        response: dict[str, Any],
    ) -> RLExample:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI rollout response did not include choices.")
        choice = choices[0]
        message = choice.get("message") or {}
        completion_text = str(message.get("content") or "").strip()
        completion_ids = [int(token_id) for token_id in (choice.get("token_ids") or [])]
        if not completion_ids:
            raise RuntimeError(
                "OpenAI rollout response did not include completion token ids. "
                "Ensure the vLLM server supports return_token_ids."
            )
        completion_logprobs = self._openai_completion_logprobs(choice, completion_ids)
        sample = _shift_completion_sample(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=completion_logprobs,
            temperature=self.config.inference.sampling.temperature,
        )
        trainable_indexes = [
            index for index, trainable in enumerate(sample["loss_mask"]) if trainable
        ]
        if not trainable_indexes:
            raise RuntimeError("OpenAI rollout response produced no trainable tokens.")
        return replace(
            record,
            completion=[{"role": "assistant", "content": completion_text}],
            target_completion=record.target_completion or record.completion,
            input_ids=sample["input_ids"],
            target_ids=sample["target_ids"],
            loss_mask=sample["loss_mask"],
            inference_logprobs=[
                float(sample["inference_logprobs"][index])
                for index in trainable_indexes
            ],
            temperatures=[
                float(sample["temperatures"][index]) for index in trainable_indexes
            ],
        )

    def _openai_completion_logprobs(
        self,
        choice: dict[str, Any],
        completion_ids: list[int],
    ) -> list[float]:
        raw_logprobs = ((choice.get("logprobs") or {}).get("content") or [])
        values = [float(item.get("logprob", 0.0)) for item in raw_logprobs]
        if len(values) < len(completion_ids):
            values.extend([0.0] * (len(completion_ids) - len(values)))
        return values[: len(completion_ids)]

    def close(self) -> None:
        return None

    def _request_all(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if len(self.base_urls) == 1:
            self._request(method, path, payload, base_url=self.base_url)
            return
        with ThreadPoolExecutor(max_workers=len(self.base_urls)) as executor:
            futures = [
                executor.submit(
                    self._request,
                    method,
                    path,
                    payload,
                    base_url=base_url,
                )
                for base_url in self.base_urls
            ]
            for future in futures:
                future.result()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{base_url or self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.inference.http.request_timeout_seconds,
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP server returned {exc.code} for {path}: {detail}"
            ) from exc
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _uses_openai_rollouts(self) -> bool:
        return (
            self.config.inference.vllm.server_backend == "openai"
            and self.config.orchestrator.custom_rollout_function is None
        )


def _shift_completion_sample(
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    completion_logprobs: list[float],
    temperature: float,
) -> dict[str, list[Any]]:
    full_ids = prompt_ids + completion_ids
    full_mask = [False] * len(prompt_ids) + [True] * len(completion_ids)
    full_logprobs = [0.0] * len(prompt_ids) + completion_logprobs
    full_temperatures = [temperature] * len(full_ids)
    if len(full_ids) < 2:
        return {
            "input_ids": [],
            "target_ids": [],
            "loss_mask": [],
            "inference_logprobs": [],
            "temperatures": [],
        }
    return {
        "input_ids": full_ids[:-1],
        "target_ids": full_ids[1:],
        "loss_mask": full_mask[1:],
        "inference_logprobs": full_logprobs[1:],
        "temperatures": full_temperatures[1:],
    }

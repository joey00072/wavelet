# ruff: noqa: F811

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig, RLSamplingConfig
from wavelet.data.rl import (
    RLExample,
    rl_examples_from_payload,
    rl_examples_to_payload,
)


def openai_sampling_payload(sampling: RLSamplingConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "temperature": sampling.temperature if sampling.do_sample else 0.0,
        "top_p": sampling.top_p,
        "min_p": sampling.min_p,
    }
    if sampling.max_completion_tokens is not None:
        payload["max_completion_tokens"] = sampling.max_completion_tokens
    if sampling.top_k != 0:
        payload["top_k"] = sampling.top_k
    if sampling.min_tokens > 0:
        payload["min_tokens"] = sampling.min_tokens
    if sampling.seed is not None:
        payload["seed"] = sampling.seed
    if sampling.repetition_penalty != 1.0:
        payload["repetition_penalty"] = sampling.repetition_penalty
    payload.update(sampling.extra_body)
    return payload


def vllm_sampling_kwargs(
    sampling: RLSamplingConfig, *, use_generation_logprobs: bool
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "n": sampling.num_generations,
        "temperature": sampling.temperature if sampling.do_sample else 0.0,
        "top_p": sampling.top_p,
        "repetition_penalty": sampling.repetition_penalty,
        "min_p": sampling.min_p,
    }
    if sampling.max_completion_tokens is not None:
        kwargs["max_tokens"] = sampling.max_completion_tokens
    if use_generation_logprobs:
        kwargs["logprobs"] = 1
    if sampling.top_k != 0:
        kwargs["top_k"] = sampling.top_k
    if sampling.seed is not None:
        kwargs["seed"] = sampling.seed
    for key in ("stop", "include_stop_str_in_output"):
        if key in sampling.extra_body:
            kwargs[key] = sampling.extra_body[key]
    return kwargs


def openai_payload_to_vllm_kwargs(
    payload: dict[str, Any], *, default_max_tokens: int | None
) -> dict[str, Any]:
    extra_body = payload.get("extra_body") or {}
    max_tokens = (
        payload.get("max_completion_tokens")
        or payload.get("max_tokens")
        or default_max_tokens
    )
    kwargs: dict[str, Any] = {
        "n": 1,
        "temperature": float(payload.get("temperature", 1.0)),
        "top_p": float(payload.get("top_p", 1.0)),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    for key, converter in {
        "repetition_penalty": float,
        "top_k": int,
        "min_p": float,
    }.items():
        value = payload.get(key)
        if value is None:
            value = extra_body.get(key)
        if value is not None:
            kwargs[key] = converter(value)
    if payload.get("seed") is not None:
        kwargs["seed"] = int(payload["seed"])
    stop = payload.get("stop") or extra_body.get("stop")
    if stop is not None:
        kwargs["stop"] = stop
    include_stop = payload.get("include_stop_str_in_output")
    if include_stop is None:
        include_stop = extra_body.get("include_stop_str_in_output")
    if include_stop is not None:
        kwargs["include_stop_str_in_output"] = bool(include_stop)
    if (
        payload.get("logprobs")
        or payload.get("return_token_ids")
        or extra_body.get("return_token_ids")
    ):
        kwargs["logprobs"] = 1
    return kwargs


def fit_generation_context(
    prompt_ids: list[int],
    *,
    max_prompt_tokens: int | None,
    max_model_len: int | None,
    max_completion_tokens: int | None,
) -> tuple[list[int], int | None]:
    """Fit a tokenized prompt and completion budget into the model context."""
    fitted_prompt_ids = prompt_ids
    if max_prompt_tokens is not None and len(fitted_prompt_ids) > max_prompt_tokens:
        fitted_prompt_ids = fitted_prompt_ids[-max_prompt_tokens:]

    fitted_completion_tokens = max_completion_tokens
    if max_model_len is None or max_model_len <= 0:
        return fitted_prompt_ids, fitted_completion_tokens

    max_prompt_len = max(max_model_len - 1, 1)
    if len(fitted_prompt_ids) > max_prompt_len:
        fitted_prompt_ids = fitted_prompt_ids[-max_prompt_len:]
    available_completion_tokens = max(max_model_len - len(fitted_prompt_ids), 1)
    if fitted_completion_tokens is None:
        fitted_completion_tokens = available_completion_tokens
    else:
        fitted_completion_tokens = min(
            fitted_completion_tokens,
            available_completion_tokens,
        )
    return fitted_prompt_ids, fitted_completion_tokens


def logprob_value(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    raise TypeError(f"Unsupported vLLM logprob value: {type(value)!r}")


def _token_logprob(row: dict[object, object], token_id: int) -> object | None:
    if token_id in row:
        return row[token_id]
    string_token_id = str(token_id)
    if string_token_id in row:
        return row[string_token_id]
    return None


def extract_vllm_generation_logprobs(
    logprobs: object, token_ids: list[int]
) -> list[float]:
    if logprobs is None:
        raise ValueError("vLLM output did not include token logprobs.")
    rows = list(logprobs)
    if len(rows) != len(token_ids):
        raise ValueError(
            "vLLM returned a different number of logprob rows and generated "
            f"tokens ({len(rows)} != {len(token_ids)})."
        )
    values: list[float] = []
    for token_id, row in zip(token_ids, rows, strict=True):
        if isinstance(row, dict):
            candidate = _token_logprob(row, token_id)
            if candidate is None:
                raise ValueError(
                    f"vLLM logprobs are missing sampled token id {token_id}."
                )
            values.append(logprob_value(candidate))
        else:
            values.append(logprob_value(row))
    return values


def extract_vllm_prompt_logprobs(
    prompt_logprobs: object,
    *,
    target_ids: list[int],
    loss_mask: list[bool],
) -> list[float]:
    if prompt_logprobs is None:
        raise ValueError("vLLM scoring output did not include prompt_logprobs.")
    rows = list(prompt_logprobs)
    expected_rows = len(target_ids) + 1
    if len(rows) != expected_rows:
        raise ValueError(
            "vLLM returned a different number of prompt logprob rows and scored "
            f"tokens ({len(rows)} != {expected_rows})."
        )
    values: list[float] = []
    for index, (token_id, trainable) in enumerate(
        zip(target_ids, loss_mask, strict=True)
    ):
        if not trainable:
            continue
        row = rows[index + 1]
        if not isinstance(row, dict):
            values.append(logprob_value(row))
            continue
        candidate = _token_logprob(row, token_id)
        if candidate is None:
            raise ValueError(
                f"vLLM prompt_logprobs are missing target token id {token_id}."
            )
        values.append(logprob_value(candidate))
    return values


from wavelet.inference.policy import PolicyInferenceEngine, RLInference
from wavelet.trainer.model import setup_tokenizer
from wavelet.transport.policy import NCCL_READY_MARKER


class HTTPPolicyInferenceEngine(PolicyInferenceEngine):
    """HTTP client for a persistent Wavelet vLLM rollout server."""

    def __init__(self, config: RLConfig) -> None:
        super().__init__(config)
        ports = config.inference.http.ports or [config.inference.http.port]
        self.base_urls = [
            f"http://{config.inference.http.host}:{port}" for port in ports
        ]
        self.base_url = self.base_urls[0]
        self.tokenizer = None
        self.policy_model_name = config.model.name

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
            f"Timed out waiting for vLLM HTTP server(s) at {', '.join(self.base_urls)}."
        ) from last_error

    def load_policy(self, policy_dir: Path, *, step: int) -> None:
        if (
            self._uses_openai_rollouts()
            and self.config.lora is None
            and self.config.policy_transfer.type == "nccl"
            and step > 0
        ):
            (policy_dir / NCCL_READY_MARKER).touch()
        payload: dict[str, Any] = {"policy_dir": str(policy_dir), "step": step}
        if self._uses_openai_rollouts() and self.config.lora is not None:
            payload["adapter_name"] = self.config.policy_transfer.adapter_name
            payload["load_inplace"] = True
        if self.config.lora is not None:
            responses = self._request_all("POST", "/load_policy", payload)
        else:
            responses = self._load_policy_while_generation_paused(payload)
        for response in responses:
            if int(response.get("policy_step", -1)) != step:
                raise RuntimeError(
                    "vLLM server acknowledged the wrong policy step: "
                    f"expected {step}, received {response.get('policy_step')}."
                )
        self.policy_step = step
        if self.config.lora is not None:
            self.policy_model_name = self.config.policy_transfer.adapter_name

    def _load_policy_while_generation_paused(
        self,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Drain generation on every replica before replacing policy weights."""
        primary_error: Exception | None = None
        try:
            self._request_all("POST", "/pause")
            return self._request_all("POST", "/load_policy", payload)
        except Exception as exc:
            primary_error = exc
            raise
        finally:
            try:
                self._request_all("POST", "/resume")
            except Exception as exc:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "Inference policy loading also failed to resume every replica: "
                    f"{exc}"
                )

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
            return self._annotate_single_server(records)
        return self._annotate_round_robin(records, self._annotate_native_chunk)

    def _annotate_single_server(self, records: list[RLExample]) -> list[RLExample]:
        return self._annotate_native_chunk(records, self.base_url)

    def _annotate_native_chunk(
        self,
        records: list[RLExample],
        base_url: str,
    ) -> list[RLExample]:
        response = self._request(
            "POST",
            "/annotate",
            {"records": rl_examples_to_payload(records)},
            base_url=base_url,
        )
        self.policy_step = response.get("policy_step")
        return rl_examples_from_payload(response["records"])

    def _annotate_openai(self, records: list[RLExample]) -> list[RLExample]:
        if self.tokenizer is None:
            self.tokenizer = setup_tokenizer(self.config.model)
        if len(records) == 0:
            return []
        policy_model_name = self.policy_model_name
        return self._annotate_round_robin(
            records,
            lambda chunk, base_url: self._annotate_openai_chunk(
                chunk,
                base_url=base_url,
                policy_model_name=policy_model_name,
            ),
        )

    def _annotate_round_robin(
        self,
        records: list[RLExample],
        annotate_chunk: Callable[[list[RLExample], str], list[RLExample]],
    ) -> list[RLExample]:
        chunks = [
            records[index :: len(self.base_urls)]
            for index in range(len(self.base_urls))
        ]
        with ThreadPoolExecutor(max_workers=len(self.base_urls)) as executor:
            futures = {
                executor.submit(annotate_chunk, chunk, self.base_urls[index]): index
                for index, chunk in enumerate(chunks)
                if chunk
            }
            annotated = {index: future.result() for future, index in futures.items()}
        positions = [0] * len(chunks)
        merged = []
        for index in range(len(records)):
            chunk_index = index % len(self.base_urls)
            position = positions[chunk_index]
            merged.append(annotated[chunk_index][position])
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
            prompt_ids, max_completion_tokens = fit_generation_context(
                prompt_ids,
                max_prompt_tokens=(self.config.inference.sampling.max_prompt_tokens),
                max_model_len=self.config.inference.vllm.max_model_len,
                max_completion_tokens=(
                    self.config.inference.sampling.max_completion_tokens
                ),
            )
            response = self._request(
                "POST",
                "/chat/completions/tokens",
                self._openai_payload(
                    record,
                    prompt_ids,
                    policy_model_name=policy_model_name,
                    max_completion_tokens=max_completion_tokens,
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
        max_completion_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": policy_model_name or self.policy_model_name,
            "messages": record.prompt,
            "tokens": prompt_ids,
            "logprobs": True,
            **openai_sampling_payload(self.config.inference.sampling),
            "return_token_ids": True,
        }
        if record.tools is not None:
            payload["tools"] = record.tools
        if record.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = record.chat_template_kwargs
        if self.policy_step is not None:
            payload.setdefault("cache_salt", str(self.policy_step))
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
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
        raw_logprobs = (choice.get("logprobs") or {}).get("content") or []
        if len(raw_logprobs) != len(completion_ids):
            raise RuntimeError(
                "OpenAI rollout logprobs do not align with completion token ids "
                f"({len(raw_logprobs)} != {len(completion_ids)})."
            )
        values: list[float] = []
        for index, item in enumerate(raw_logprobs):
            if not isinstance(item, dict) or item.get("logprob") is None:
                raise RuntimeError(
                    "OpenAI rollout response is missing the sampled-token logprob "
                    f"at completion index {index}."
                )
            values.append(float(item["logprob"]))
        return values

    def close(self) -> None:
        return None

    def _request_all(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(self.base_urls) == 1:
            return [self._request(method, path, payload, base_url=self.base_url)]
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
            return [future.result() for future in futures]

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


import gc
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl import RLExample
from wavelet.data.sft import Example, build_sample
from wavelet.inference.engine import (
    extract_vllm_generation_logprobs,
    extract_vllm_prompt_logprobs,
    openai_payload_to_vllm_kwargs,
    vllm_sampling_kwargs,
)
from wavelet.inference.policy import PolicyInferenceEngine, RLInference, token_ids
from wavelet.trainer.model import setup_tokenizer
from wavelet.utils.monitoring import emit_perf, perf_enabled


@dataclass
class _OpenAIBatchRequest:
    payload: dict[str, Any]
    done: threading.Event
    result: dict[str, Any] | None = None
    error: BaseException | None = None


def _vllm_dtype(config: RLConfig) -> str:
    return config.inference.vllm.dtype or config.model.torch_dtype


def _sampling_params_type():
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM SamplingParams import failed.") from exc
    return SamplingParams


class VLLMPolicyInferenceEngine(PolicyInferenceEngine):
    """Persistent vLLM rollout engine with LoRA policy snapshot switching."""

    def __init__(self, config: RLConfig) -> None:
        super().__init__(config)
        self.llm = None
        self.tokenizer = None
        self._lora_request = None
        self._generate_lock = threading.Lock()
        self._openai_batch: list[_OpenAIBatchRequest] = []
        self._openai_batch_condition = threading.Condition()
        self._openai_batch_worker: threading.Thread | None = None
        self._tokenize_calls = 0
        self._tokenize_tokens = 0
        self._tokenize_seconds = 0.0

    def setup(self) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM inference requires the 'vllm' package. Install project "
                "dependencies with `uv sync`, then use inference.mode='vllm_http'."
            ) from exc

        vllm_config = self.config.inference.vllm
        max_lora_rank = vllm_config.max_lora_rank or (
            self.config.lora.rank if self.config.lora is not None else 1
        )
        kwargs: dict[str, Any] = {
            "model": self.config.model.name,
            "trust_remote_code": (
                self.config.model.trust_remote_code
                if vllm_config.trust_remote_code is None
                else vllm_config.trust_remote_code
            ),
            "dtype": _vllm_dtype(self.config),
            "max_model_len": vllm_config.max_model_len
            if vllm_config.max_model_len is not None
            else self.config.data.seq_len + 1,
            "tensor_parallel_size": vllm_config.tensor_parallel_size,
            "gpu_memory_utilization": vllm_config.gpu_memory_utilization,
            "enforce_eager": vllm_config.enforce_eager,
            "logprobs_mode": "processed_logprobs",
            "enable_lora": self.config.lora is not None,
            "max_loras": 1,
            "max_cpu_loras": 1,
            "max_lora_rank": max_lora_rank,
        }
        if vllm_config.quantization is not None:
            kwargs["quantization"] = vllm_config.quantization
        if vllm_config.load_format is not None:
            kwargs["load_format"] = vllm_config.load_format
        if self.config.lora is not None:
            kwargs["fully_sharded_loras"] = vllm_config.fully_sharded_loras
        if self.config.launcher.mode == "colocate_sleep":
            kwargs["enable_sleep_mode"] = True
        self.llm = LLM(**kwargs)
        self.tokenizer = setup_tokenizer(self.config.model)
        self._openai_batch_worker = threading.Thread(
            target=self._openai_batch_loop,
            name="wavelet-openai-batcher",
            daemon=True,
        )
        self._openai_batch_worker.start()

    def load_policy(self, policy_dir: Path, *, step: int) -> None:
        started_at = perf_counter()
        adapter_dir = policy_dir / "adapter"
        if adapter_dir.exists():
            self._load_adapter_policy(adapter_dir, step=step)
            self.policy_step = step
            emit_perf(
                "vllm_load_policy",
                step=step,
                kind="adapter",
                seconds=perf_counter() - started_at,
            )
            return
        model_dir = policy_dir / "model"
        if model_dir.exists():
            raise NotImplementedError(
                "vLLM full-model hot reload is not implemented. Use LoRA policy "
                "transfer for rollout inference."
            )
        raise FileNotFoundError(
            f"Policy snapshot '{policy_dir}' does not contain adapter/ or model/."
        )

    def init_weight_transfer(self, init_info: dict[str, Any]) -> None:
        if self.llm is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        self.llm.init_weight_transfer_engine({"init_info": init_info})

    def update_weights(self, update_info: dict[str, Any]) -> None:
        if self.llm is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        self.llm.update_weights({"update_info": update_info})

    def sleep(self) -> None:
        if self.llm is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        if hasattr(self.llm, "llm_engine") and hasattr(
            self.llm.llm_engine,
            "reset_prefix_cache",
        ):
            self.llm.llm_engine.reset_prefix_cache()
        if hasattr(self.llm, "reset_mm_cache"):
            self.llm.reset_mm_cache()
        self.llm.sleep(level=1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def wake(self, *, tags: list[str] | None = None) -> None:
        if self.llm is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        kwargs = {}
        if tags is not None:
            kwargs["tags"] = tags
        self.llm.wake_up(**kwargs)

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        if not self.config.inference.enabled:
            return records

        prompts = [
            {
                "prompt_token_ids": RLInference(self.config).prompt_token_ids(
                    self.tokenizer,
                    record,
                )
            }
            for record in records
        ]
        sampling_params = self._sampling_params()
        lora_request = self._lora_request
        with self._generate_lock:
            request_outputs = self.llm.generate(
                prompts,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
        self._mark_lora_loaded()

        generated_records: list[RLExample] = []
        generation_token_ids: list[list[int]] = []
        generation_logprobs: list[object] = []
        for record, request_output in zip(records, request_outputs, strict=True):
            for output in request_output.outputs:
                generated_records.append(
                    replace(
                        record,
                        completion=[
                            {
                                "role": "assistant",
                                "content": output.text.strip(),
                            }
                        ],
                        target_completion=record.target_completion or record.completion,
                    )
                )
                generation_token_ids.append(
                    [
                        int(token_id)
                        for token_id in (getattr(output, "token_ids", None) or [])
                    ]
                )
                generation_logprobs.append(getattr(output, "logprobs", None))
        return self._attach_generation_or_prompt_logprobs(
            generated_records,
            generation_token_ids=generation_token_ids,
            generation_logprobs=generation_logprobs,
        )

    def openai_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = _OpenAIBatchRequest(payload=payload, done=threading.Event())
        with self._openai_batch_condition:
            self._openai_batch.append(request)
            self._openai_batch_condition.notify()
        request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("OpenAI chat completion batcher returned no result.")
        return request.result

    def _openai_batch_loop(self) -> None:
        while True:
            with self._openai_batch_condition:
                while not self._openai_batch:
                    self._openai_batch_condition.wait()
                batch = self._collect_openai_batch_locked()

            try:
                results = self._openai_chat_completion_batch(
                    [request.payload for request in batch],
                )
            except BaseException as exc:  # noqa: BLE001 - unblock every waiter
                for request in batch:
                    request.error = exc
                    request.done.set()
                continue

            for request, result in zip(batch, results, strict=True):
                request.result = result
                request.done.set()

    def _collect_openai_batch_locked(self) -> list[_OpenAIBatchRequest]:
        vllm_config = self.config.inference.vllm
        min_size = vllm_config.openai_batch_min_size
        max_size = vllm_config.openai_batch_max_size
        first_wait = vllm_config.openai_batch_wait_seconds
        max_wait = max(first_wait, vllm_config.openai_batch_max_wait_seconds)
        started_at = perf_counter()
        if first_wait > 0:
            self._openai_batch_condition.wait(timeout=first_wait)

        while len(self._openai_batch) < min_size:
            remaining = (started_at + max_wait) - perf_counter()
            if remaining <= 0:
                break
            self._openai_batch_condition.wait(timeout=remaining)

        if max_size is not None and len(self._openai_batch) > max_size:
            batch = self._openai_batch[:max_size]
            self._openai_batch = self._openai_batch[max_size:]
            self._openai_batch_condition.notify()
            return batch

        batch = self._openai_batch
        self._openai_batch = []
        return batch

    def _openai_chat_completion_batch(
        self,
        payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        started_at = perf_counter()
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        sampling_params_type = _sampling_params_type()
        prompts: list[dict[str, list[int]]] = []
        prompt_id_rows: list[list[int]] = []
        sampling_params: list[Any] = []
        for payload in payloads:
            messages = payload["messages"]
            prompt_ids = payload.get("tokens")
            if prompt_ids is None:
                prompt_ids = token_ids(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        tools=payload.get("tools"),
                    )
                )
            else:
                prompt_ids = [int(token_id) for token_id in prompt_ids]
            sampling_kwargs = self._openai_sampling_kwargs(payload)
            prompt_ids, sampling_kwargs = self._fit_openai_context(
                prompt_ids,
                sampling_kwargs,
            )
            prompt_id_rows.append(prompt_ids)
            prompts.append({"prompt_token_ids": prompt_ids})
            sampling_params.append(sampling_params_type(**sampling_kwargs))

        prefill_tokens = sum(len(row) for row in prompt_id_rows)
        lora_request = self._lora_request
        with self._generate_lock:
            outputs = self.llm.generate(
                prompts,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
        self._mark_lora_loaded()
        results: list[dict[str, Any]] = []
        completion_tokens = 0
        for payload, prompt_ids, request_output in zip(
            payloads,
            prompt_id_rows,
            outputs,
            strict=True,
        ):
            output = request_output.outputs[0]
            completion_ids = [
                int(token_id) for token_id in (getattr(output, "token_ids", None) or [])
            ]
            completion_tokens += len(completion_ids)
            completion_logprobs = self._openai_logprob_content(
                completion_ids,
                getattr(output, "logprobs", None),
            )
            finish_reason = (
                "length"
                if getattr(output, "finish_reason", None) == "length"
                else "stop"
            )
            results.append(
                {
                    "id": f"wavelet-{self.policy_step or 0}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload.get("model") or self.config.model.name,
                    "prompt_token_ids": prompt_ids,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output.text,
                            },
                            "finish_reason": finish_reason,
                            "token_ids": completion_ids,
                            "logprobs": {"content": completion_logprobs},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(completion_ids),
                        "total_tokens": len(prompt_ids) + len(completion_ids),
                    },
                }
            )
        seconds = perf_counter() - started_at
        max_memory = 0
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_reserved()
        if perf_enabled():
            tokenize_calls = self._tokenize_calls
            tokenize_tokens = self._tokenize_tokens
            tokenize_seconds = self._tokenize_seconds
            self._tokenize_calls = 0
            self._tokenize_tokens = 0
            self._tokenize_seconds = 0.0
            emit_perf(
                "vllm_openai_batch",
                batch=len(payloads),
                prefill_tokens=prefill_tokens,
                completion_tokens=completion_tokens,
                seconds=seconds,
                tokens_per_s=(
                    f"{(prefill_tokens + completion_tokens) / max(seconds, 1e-9):.1f}"
                ),
                tokenize_calls=tokenize_calls,
                tokenize_tokens=tokenize_tokens,
                tokenize_seconds=f"{tokenize_seconds:.4f}",
                cuda_max_reserved=max_memory,
            )
        return results

    def _openai_sampling_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        return openai_payload_to_vllm_kwargs(
            payload,
            default_max_tokens=self.config.inference.sampling.max_completion_tokens,
        )

    def _fit_openai_context(
        self,
        prompt_ids: list[int],
        sampling_kwargs: dict[str, Any],
    ) -> tuple[list[int], dict[str, Any]]:
        fitted_kwargs = dict(sampling_kwargs)
        prompt_ids, max_tokens = fit_generation_context(
            prompt_ids,
            max_prompt_tokens=self.config.inference.sampling.max_prompt_tokens,
            max_model_len=self.config.inference.vllm.max_model_len,
            max_completion_tokens=fitted_kwargs.get("max_tokens"),
        )
        if max_tokens is not None:
            fitted_kwargs["max_tokens"] = max_tokens
        return prompt_ids, fitted_kwargs

    def tokenize_messages(self, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = perf_counter()
        if self.tokenizer is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        if "prompt" in payload:
            ids = token_ids(self.tokenizer(payload["prompt"], add_special_tokens=False))
        else:
            ids = token_ids(
                self.tokenizer.apply_chat_template(
                    payload["messages"],
                    tokenize=True,
                    add_generation_prompt=payload.get("add_generation_prompt", True),
                    tools=payload.get("tools"),
                )
            )
        seconds = perf_counter() - started_at
        self._tokenize_calls += 1
        self._tokenize_tokens += len(ids)
        self._tokenize_seconds += seconds
        return {
            "count": len(ids),
            "max_model_len": self.config.inference.vllm.max_model_len
            or self.config.data.seq_len + 1,
            "tokens": ids,
        }

    def _openai_logprob_content(
        self,
        token_ids: list[int],
        logprobs: object,
    ) -> list[dict[str, object]]:
        if self.tokenizer is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        if logprobs is None:
            raise RuntimeError(
                "vLLM generation did not return sampled-token logprobs. Enable "
                "generation logprobs for RL rollouts."
            )
        values = extract_vllm_generation_logprobs(logprobs, token_ids)
        return [
            {
                "token": "",
                "logprob": logprob,
                "bytes": None,
                "top_logprobs": [],
            }
            for token_id, logprob in zip(token_ids, values, strict=True)
        ]

    def close(self) -> None:
        self.llm = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sampling_params(self):
        kwargs = vllm_sampling_kwargs(
            self.config.inference.sampling,
            use_generation_logprobs=(
                self.config.inference.vllm.use_generation_logprobs
            ),
        )
        return _sampling_params_type()(**kwargs)

    def _load_adapter_policy(
        self, adapter_dir: Path, *, step: int | None = None
    ) -> None:
        if self.llm is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        if self.config.lora is None:
            raise ValueError("vLLM adapter policy transfer requires lora config.")
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError("vLLM LoRARequest import failed.") from exc

        adapter_id = self.config.policy_transfer.adapter_id
        adapter_name = self.config.policy_transfer.adapter_name
        self._lora_request = LoRARequest(
            adapter_name,
            adapter_id,
            str(adapter_dir),
            load_inplace=False,
        )

    def _attach_generation_or_prompt_logprobs(
        self,
        records: list[RLExample],
        *,
        generation_token_ids: list[list[int]],
        generation_logprobs: list[object],
    ) -> list[RLExample]:
        if not self.config.inference.vllm.use_generation_logprobs:
            return self._attach_prompt_logprobs(records)

        effective_temperature = max(
            self.config.inference.sampling.temperature,
            1e-6,
        )
        annotated: list[RLExample | None] = [None] * len(records)
        fallback_records: list[RLExample] = []
        fallback_indexes: list[int] = []
        for index, record in enumerate(records):
            sample = self._build_record_sample(record)
            trainable_ids = [
                int(token_id)
                for token_id, trainable in zip(
                    sample["target_ids"],
                    sample["loss_mask"],
                    strict=True,
                )
                if trainable
            ]
            token_ids = generation_token_ids[index]
            logprobs = generation_logprobs[index]
            if (
                logprobs is not None
                and len(token_ids) >= len(trainable_ids)
                and token_ids[: len(trainable_ids)] == trainable_ids
            ):
                try:
                    inference_logprobs = extract_vllm_generation_logprobs(
                        logprobs,
                        trainable_ids,
                    )
                except (TypeError, ValueError):
                    pass
                else:
                    annotated[index] = replace(
                        record,
                        inference_logprobs=inference_logprobs,
                        temperatures=[effective_temperature] * len(inference_logprobs),
                    )
                    continue
            fallback_indexes.append(index)
            fallback_records.append(record)

        if fallback_records:
            fallback_annotated = self._attach_prompt_logprobs(fallback_records)
            for index, record in zip(
                fallback_indexes,
                fallback_annotated,
                strict=True,
            ):
                annotated[index] = record

        return [record for record in annotated if record is not None]

    def _attach_prompt_logprobs(self, records: list[RLExample]) -> list[RLExample]:
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        if not records:
            return records

        samples: list[dict[str, Any]] = []
        prompts: list[dict[str, list[int]]] = []
        for record in records:
            sample = self._build_record_sample(record)
            samples.append(sample)
            full_token_ids = list(sample["input_ids"]) + [sample["target_ids"][-1]]
            prompts.append({"prompt_token_ids": full_token_ids})

        lora_request = self._lora_request
        with self._generate_lock:
            scoring_outputs = self.llm.generate(
                prompts,
                self._prompt_logprob_params(),
                use_tqdm=False,
                lora_request=lora_request,
            )
        self._mark_lora_loaded()

        effective_temperature = max(
            self.config.inference.sampling.temperature,
            1e-6,
        )
        annotated: list[RLExample] = []
        for record, sample, output in zip(
            records,
            samples,
            scoring_outputs,
            strict=True,
        ):
            inference_logprobs = extract_vllm_prompt_logprobs(
                output.prompt_logprobs,
                target_ids=[int(token_id) for token_id in sample["target_ids"]],
                loss_mask=[bool(value) for value in sample["loss_mask"]],
            )
            annotated.append(
                replace(
                    record,
                    inference_logprobs=inference_logprobs,
                    temperatures=[effective_temperature] * len(inference_logprobs),
                )
            )
        return annotated

    def _build_record_sample(self, record: RLExample) -> dict[str, Any]:
        sample = build_sample(
            Example(
                prompt=record.prompt,
                completion=record.completion,
                tools=record.tools,
                chat_template_kwargs=record.chat_template_kwargs,
                source=record.source,
            ),
            self.tokenizer,
            seq_len=self.config.data.seq_len,
            loss_mask_config=self.config.data.loss_mask,
        )
        if sample is None:
            raise ValueError("Generated rollout produced no trainable tokens.")
        return sample

    def _prompt_logprob_params(self):
        return _sampling_params_type()(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )

    def _mark_lora_loaded(self) -> None:
        if self._lora_request is None:
            return
        if not getattr(self._lora_request, "load_inplace", False):
            return
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError("vLLM LoRARequest import failed.") from exc

        self._lora_request = LoRARequest(
            self._lora_request.lora_name,
            self._lora_request.lora_int_id,
            str(self._lora_request.lora_path),
            load_inplace=False,
        )

from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.data.loading import Example
from wavelet.data.rl_dataset import RLExample
from wavelet.data.tokenization import build_sample
from wavelet.inference.policy import PolicyInferenceEngine, RLInference, token_ids
from wavelet.trainer.model import setup_tokenizer
from wavelet.utils.monitoring import emit_perf, perf_enabled


@dataclass
class _OpenAIBatchRequest:
    payload: dict[str, Any]
    done: threading.Event
    result: dict[str, Any] | None = None
    error: BaseException | None = None


def _logprob_value(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "logprob"):
        return float(getattr(value, "logprob"))
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    raise TypeError(f"Unsupported vLLM logprob value: {type(value)!r}")


def _extract_vllm_generation_logprobs(
    logprobs: object,
    token_ids: list[int],
) -> list[float]:
    if logprobs is None:
        raise ValueError("vLLM output did not include token logprobs.")
    rows = list(logprobs)
    if len(rows) < len(token_ids):
        raise ValueError(
            "vLLM returned fewer logprob rows than generated tokens "
            f"({len(rows)} < {len(token_ids)})."
        )
    values: list[float] = []
    for token_id, row in zip(token_ids, rows, strict=False):
        if isinstance(row, dict):
            candidate = row.get(token_id) or row.get(str(token_id))
            if candidate is None:
                raise ValueError(
                    f"vLLM logprobs are missing sampled token id {token_id}."
                )
            values.append(_logprob_value(candidate))
            continue
        values.append(_logprob_value(row))
    return values


def _extract_vllm_prompt_logprobs(
    prompt_logprobs: object,
    *,
    target_ids: list[int],
    loss_mask: list[bool],
) -> list[float]:
    if prompt_logprobs is None:
        raise ValueError("vLLM scoring output did not include prompt_logprobs.")
    rows = list(prompt_logprobs)
    if len(rows) < len(target_ids) + 1:
        raise ValueError(
            "vLLM returned fewer prompt logprob rows than scored tokens "
            f"({len(rows)} < {len(target_ids) + 1})."
        )

    values: list[float] = []
    for index, (token_id, trainable) in enumerate(
        zip(target_ids, loss_mask, strict=True),
    ):
        if not trainable:
            continue
        row = rows[index + 1]
        if not isinstance(row, dict):
            values.append(_logprob_value(row))
            continue
        candidate = row.get(token_id) or row.get(str(token_id))
        if candidate is None:
            raise ValueError(
                f"vLLM prompt_logprobs are missing target token id {token_id}."
            )
        values.append(_logprob_value(candidate))
    return values


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
            except BaseException as exc:
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
        extra_body = payload.get("extra_body") or {}
        max_tokens = (
            payload.get("max_completion_tokens")
            or payload.get("max_tokens")
            or self.config.inference.sampling.max_completion_tokens
        )
        kwargs: dict[str, Any] = {
            "n": 1,
            "temperature": float(payload.get("temperature", 1.0)),
            "top_p": float(payload.get("top_p", 1.0)),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        repetition_penalty = payload.get("repetition_penalty") or extra_body.get(
            "repetition_penalty"
        )
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(repetition_penalty)
        top_k = payload.get("top_k") or extra_body.get("top_k")
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
        min_p = payload.get("min_p") or extra_body.get("min_p")
        if min_p is not None:
            kwargs["min_p"] = float(min_p)
        seed = payload.get("seed")
        if seed is not None:
            kwargs["seed"] = int(seed)
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

    def _fit_openai_context(
        self,
        prompt_ids: list[int],
        sampling_kwargs: dict[str, Any],
    ) -> tuple[list[int], dict[str, Any]]:
        fitted_kwargs = dict(sampling_kwargs)
        max_prompt_tokens = self.config.inference.sampling.max_prompt_tokens
        if max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        max_model_len = self.config.inference.vllm.max_model_len
        if max_model_len is None or max_model_len <= 0:
            return prompt_ids, fitted_kwargs

        max_prompt_len = max(max_model_len - 1, 1)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        room = max(max_model_len - len(prompt_ids), 1)
        fitted_kwargs["max_tokens"] = min(int(fitted_kwargs["max_tokens"]), room)
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
            return [
                {
                    "token": "",
                    "logprob": 0.0,
                    "bytes": None,
                    "top_logprobs": [],
                }
                for token_id in token_ids
            ]
        values = _extract_vllm_generation_logprobs(logprobs, token_ids)
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
        sampling = self.config.inference.sampling
        kwargs: dict[str, Any] = {
            "n": sampling.num_generations,
            "temperature": sampling.temperature if sampling.do_sample else 0.0,
            "top_p": sampling.top_p,
            "repetition_penalty": sampling.repetition_penalty,
        }
        if sampling.max_completion_tokens is not None:
            kwargs["max_tokens"] = sampling.max_completion_tokens
        if self.config.inference.vllm.use_generation_logprobs:
            kwargs["logprobs"] = 1
        if sampling.top_k != 0:
            kwargs["top_k"] = sampling.top_k
        kwargs["min_p"] = sampling.min_p
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        extra_body = dict(sampling.extra_body)
        stop = extra_body.get("stop")
        if stop is not None:
            kwargs["stop"] = stop
        include_stop = extra_body.get("include_stop_str_in_output")
        if include_stop is not None:
            kwargs["include_stop_str_in_output"] = bool(include_stop)
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
                    inference_logprobs = _extract_vllm_generation_logprobs(
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
            inference_logprobs = _extract_vllm_prompt_logprobs(
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

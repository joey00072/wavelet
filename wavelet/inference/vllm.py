from __future__ import annotations

import gc
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.data.loading import Example
from wavelet.data.rl_dataset import RLExample
from wavelet.data.tokenization import build_sample
from wavelet.inference.policy import PolicyInferenceEngine, RLInference
from wavelet.trainer.model import setup_tokenizer


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


class VLLMPolicyInferenceEngine(PolicyInferenceEngine):
    """Persistent vLLM rollout engine with LoRA policy snapshot switching."""

    def __init__(self, config: RLConfig) -> None:
        super().__init__(config)
        self.llm = None
        self.tokenizer = None
        self._lora_request = None

    def setup(self) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM inference requires the 'vllm' package. Install project "
                "dependencies with `uv sync`, then use inference.mode='vllm'."
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
            "max_loras": vllm_config.max_loras,
            "max_cpu_loras": vllm_config.max_cpu_loras,
            "max_lora_rank": max_lora_rank,
        }
        self.llm = LLM(**kwargs)
        self.tokenizer = setup_tokenizer(self.config.model)

    def load_policy(self, policy_dir: Path, *, step: int) -> None:
        adapter_dir = policy_dir / "adapter"
        if adapter_dir.exists():
            self._load_adapter_policy(adapter_dir)
            self.policy_step = step
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

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        if not self.config.inference.enabled:
            return records
        if self.config.inference.mode != "vllm":
            raise ValueError(
                "VLLMPolicyInferenceEngine requires inference.mode='vllm'."
            )

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
        request_outputs = self.llm.generate(
            prompts,
            sampling_params,
            use_tqdm=False,
            lora_request=self._lora_request,
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

    def close(self) -> None:
        self.llm = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sampling_params(self):
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError("vLLM SamplingParams import failed.") from exc

        sampling = self.config.inference.sampling
        kwargs: dict[str, Any] = {
            "n": sampling.num_generations,
            "temperature": sampling.temperature if sampling.do_sample else 0.0,
            "top_p": sampling.top_p,
            "repetition_penalty": sampling.repetition_penalty,
            "max_tokens": sampling.max_completion_tokens,
        }
        if self.config.inference.vllm.use_generation_logprobs:
            kwargs["logprobs"] = 1
        if sampling.top_k > 0:
            kwargs["top_k"] = sampling.top_k
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        return SamplingParams(**kwargs)

    def _load_adapter_policy(self, adapter_dir: Path) -> None:
        if self.llm is None:
            raise RuntimeError("vLLM inference engine not set up. Call setup() first.")
        if self.config.lora is None:
            raise ValueError("vLLM adapter policy transfer requires lora config.")
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError("vLLM LoRARequest import failed.") from exc

        self._lora_request = LoRARequest(
            self.config.policy_transfer.adapter_name,
            self.config.policy_transfer.adapter_id,
            str(adapter_dir),
            load_inplace=True,
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
            samples.append(sample)
            full_token_ids = list(sample["input_ids"]) + [sample["target_ids"][-1]]
            prompts.append({"prompt_token_ids": full_token_ids})

        scoring_outputs = self.llm.generate(
            prompts,
            self._prompt_logprob_params(),
            use_tqdm=False,
            lora_request=self._lora_request,
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

    def _prompt_logprob_params(self):
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError("vLLM SamplingParams import failed.") from exc

        return SamplingParams(
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
            self.config.policy_transfer.adapter_name,
            self.config.policy_transfer.adapter_id,
            str(self._lora_request.lora_path),
            load_inplace=False,
        )

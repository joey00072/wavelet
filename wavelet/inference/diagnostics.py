from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample


@dataclass(frozen=True)
class InferenceProbeMetrics:
    records: int
    wall_seconds: float
    records_per_second: float
    model_input_tokens: int
    completion_tokens: int
    trainable_tokens: int
    model_input_tokens_per_second: float
    completion_tokens_per_second: float
    trainable_tokens_per_second: float
    records_with_completion: int
    records_with_inference_logprobs: int
    records_with_loss_mask: int
    min_completion_tokens: int
    max_completion_tokens: int
    mean_completion_tokens: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuousBatchRequest:
    index: int
    base_url: str
    data_parallel_rank: int | None
    ok: bool
    latency_seconds: float
    start_offset_seconds: float
    end_offset_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuousBatchMetrics:
    requests: int
    succeeded: int
    failed: int
    concurrency: int
    stagger_seconds: float
    wall_seconds: float
    requests_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_per_second: float
    total_tokens_per_second: float
    latency_p50_seconds: float
    latency_p90_seconds: float
    latency_max_seconds: float
    max_observed_concurrency: int
    overlapped: bool
    per_rank_requests: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def base_urls(config: RLConfig) -> list[str]:
    ports = config.inference.http.ports or [config.inference.http.port]
    return [f"http://{config.inference.http.host}:{port}" for port in ports]


def inference_debug_state(config: RLConfig) -> dict[str, Any]:
    return {
        "model": {
            "name": config.model.name,
            "torch_dtype": config.model.torch_dtype,
            "adapter_path": str(config.model.adapter_path)
            if config.model.adapter_path is not None
            else None,
            "load_in_4bit": config.model.load_in_4bit,
        },
        "inference": {
            "mode": config.inference.mode,
            "enabled": config.inference.enabled,
            "http_base_urls": base_urls(config)
            if config.inference.mode == "vllm_http"
            else [],
            "server_backend": config.inference.vllm.server_backend,
            "tensor_parallel_size": config.inference.vllm.tensor_parallel_size,
            "data_parallel_size": config.inference.vllm.data_parallel_size,
            "data_parallel_size_local": config.inference.vllm.data_parallel_size_local,
            "max_model_len": config.inference.vllm.max_model_len,
            "gpu_memory_utilization": config.inference.vllm.gpu_memory_utilization,
            "quantization": config.inference.vllm.quantization,
            "load_format": config.inference.vllm.load_format,
            "use_generation_logprobs": config.inference.vllm.use_generation_logprobs,
        },
        "sampling": config.inference.sampling.model_dump(mode="json"),
        "lora": None
        if config.lora is None
        else config.lora.model_dump(mode="json", exclude_none=True),
        "policy_transfer": config.policy_transfer.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "output_dir": str(config.output_dir),
    }


def make_probe_examples(*, count: int, prompt: str) -> list[RLExample]:
    return [
        RLExample(
            prompt=[{"role": "user", "content": f"{prompt} #{index}"}],
            completion=[{"role": "assistant", "content": ""}],
            advantage=None,
            reward=None,
            metadata={"probe_index": index},
            source="inference_probe",
        )
        for index in range(count)
    ]


def probe_engine(
    engine: Any,
    records: list[RLExample],
    *,
    warmup: int = 0,
    repeats: int = 1,
) -> tuple[list[RLExample], InferenceProbeMetrics]:
    if warmup > 0:
        engine.annotate(records[:warmup])
    started_at = time.perf_counter()
    annotated: list[RLExample] = []
    for _ in range(repeats):
        annotated.extend(engine.annotate(records))
    wall_seconds = time.perf_counter() - started_at
    return annotated, summarize_records(annotated, wall_seconds=wall_seconds)


def summarize_records(
    records: list[RLExample],
    *,
    wall_seconds: float,
) -> InferenceProbeMetrics:
    model_input_tokens = sum(len(record.input_ids or []) for record in records)
    trainable_tokens = sum(sum(record.loss_mask or []) for record in records)
    completion_lengths = [
        sum(record.loss_mask or [])
        for record in records
        if record.completion and record.completion[0].get("content")
    ]
    completion_tokens = sum(completion_lengths)
    records_with_completion = len(completion_lengths)
    records_with_inference_logprobs = sum(
        1 for record in records if record.inference_logprobs is not None
    )
    records_with_loss_mask = sum(1 for record in records if record.loss_mask is not None)
    wall = max(wall_seconds, 1e-9)
    return InferenceProbeMetrics(
        records=len(records),
        wall_seconds=wall_seconds,
        records_per_second=len(records) / wall,
        model_input_tokens=model_input_tokens,
        completion_tokens=completion_tokens,
        trainable_tokens=trainable_tokens,
        model_input_tokens_per_second=model_input_tokens / wall,
        completion_tokens_per_second=completion_tokens / wall,
        trainable_tokens_per_second=trainable_tokens / wall,
        records_with_completion=records_with_completion,
        records_with_inference_logprobs=records_with_inference_logprobs,
        records_with_loss_mask=records_with_loss_mask,
        min_completion_tokens=min(completion_lengths, default=0),
        max_completion_tokens=max(completion_lengths, default=0),
        mean_completion_tokens=(
            sum(completion_lengths) / len(completion_lengths)
            if completion_lengths
            else 0.0
        ),
    )


def http_health(config: RLConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for base_url in base_urls(config):
        started_at = time.perf_counter()
        try:
            payload = _http_json(base_url, "GET", "/health")
            debug_state = None
            try:
                debug_state = _http_json(base_url, "GET", "/debug/state")
            except (OSError, RuntimeError):
                debug_state = None
            results.append(
                {
                    "base_url": base_url,
                    "ok": True,
                    "latency_seconds": time.perf_counter() - started_at,
                    "response": payload,
                    "debug_state": debug_state,
                }
            )
        except (OSError, RuntimeError) as exc:
            results.append(
                {
                    "base_url": base_url,
                    "ok": False,
                    "latency_seconds": time.perf_counter() - started_at,
                    "error": str(exc),
                }
            )
    return results


def continuous_batch_probe(
    config: RLConfig,
    *,
    count: int,
    concurrency: int,
    prompt: str,
    max_completion_tokens: int,
    stagger_seconds: float = 0.0,
    data_parallel_size: int | None = None,
) -> tuple[list[ContinuousBatchRequest], ContinuousBatchMetrics]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be >= 1")
    if stagger_seconds < 0.0:
        raise ValueError("stagger_seconds must be >= 0")

    routes = _continuous_batch_routes(config, data_parallel_size=data_parallel_size)
    model_name = config.orchestrator.verifier_model or config.model.name
    sampling = config.inference.sampling
    started_at = time.perf_counter()

    def run_request(index: int) -> ContinuousBatchRequest:
        scheduled_at = started_at + index * stagger_seconds
        sleep_seconds = scheduled_at - time.perf_counter()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        base_url, headers = routes[index % len(routes)]
        dp_rank = headers.get("X-data-parallel-rank")
        request_started_at = time.perf_counter()
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt} #{index}",
                }
            ],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
        }
        try:
            response = _http_json(
                base_url,
                "POST",
                _chat_completions_path(base_url),
                payload,
                headers=headers,
                timeout=config.inference.http.request_timeout_seconds,
            )
            usage = response.get("usage") or {}
            ok = True
            error = None
        except (OSError, RuntimeError) as exc:
            usage = {}
            ok = False
            error = str(exc)
        request_ended_at = time.perf_counter()
        return ContinuousBatchRequest(
            index=index,
            base_url=base_url,
            data_parallel_rank=int(dp_rank) if dp_rank is not None else None,
            ok=ok,
            latency_seconds=request_ended_at - request_started_at,
            start_offset_seconds=request_started_at - started_at,
            end_offset_seconds=request_ended_at - started_at,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            error=error,
        )

    max_workers = min(count, concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_request, index) for index in range(count)]
        requests = [future.result() for future in as_completed(futures)]
    requests.sort(key=lambda item: item.index)
    wall_seconds = time.perf_counter() - started_at
    metrics = _continuous_batch_metrics(
        requests,
        concurrency=concurrency,
        stagger_seconds=stagger_seconds,
        wall_seconds=wall_seconds,
    )
    return requests, metrics


def _http_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if headers is not None:
        request_headers.update(headers)
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} returned {exc.code}: {detail}") from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _continuous_batch_routes(
    config: RLConfig,
    *,
    data_parallel_size: int | None,
) -> list[tuple[str, dict[str, str]]]:
    base_url_value = config.orchestrator.verifier_base_url
    if base_url_value is None:
        urls = [f"{url}/v1" for url in base_urls(config)]
    elif isinstance(base_url_value, str):
        urls = [base_url_value]
    else:
        urls = list(base_url_value)
    if not urls:
        raise ValueError("at least one inference base URL is required")

    dp_size = data_parallel_size or config.inference.vllm.data_parallel_size
    routes: list[tuple[str, dict[str, str]]] = []
    for base_url in urls:
        for dp_rank in range(dp_size):
            headers = (
                {"X-data-parallel-rank": str(dp_rank)} if dp_size > 1 else {}
            )
            routes.append((base_url.rstrip("/"), headers))
    return routes


def _chat_completions_path(base_url: str) -> str:
    return "/chat/completions" if base_url.endswith("/v1") else "/v1/chat/completions"


def _continuous_batch_metrics(
    requests: list[ContinuousBatchRequest],
    *,
    concurrency: int,
    stagger_seconds: float,
    wall_seconds: float,
) -> ContinuousBatchMetrics:
    succeeded = sum(1 for request in requests if request.ok)
    prompt_tokens = sum(request.prompt_tokens for request in requests)
    completion_tokens = sum(request.completion_tokens for request in requests)
    total_tokens = sum(request.total_tokens for request in requests)
    latencies = [request.latency_seconds for request in requests if request.ok]
    per_rank_requests: dict[str, int] = {}
    for request in requests:
        rank = "none" if request.data_parallel_rank is None else str(request.data_parallel_rank)
        per_rank_requests[rank] = per_rank_requests.get(rank, 0) + 1
    wall = max(wall_seconds, 1e-9)
    max_observed_concurrency = _max_observed_concurrency(requests)
    return ContinuousBatchMetrics(
        requests=len(requests),
        succeeded=succeeded,
        failed=len(requests) - succeeded,
        concurrency=concurrency,
        stagger_seconds=stagger_seconds,
        wall_seconds=wall_seconds,
        requests_per_second=len(requests) / wall,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_tokens_per_second=completion_tokens / wall,
        total_tokens_per_second=total_tokens / wall,
        latency_p50_seconds=_quantile(latencies, 0.50),
        latency_p90_seconds=_quantile(latencies, 0.90),
        latency_max_seconds=max(latencies, default=0.0),
        max_observed_concurrency=max_observed_concurrency,
        overlapped=max_observed_concurrency > 1,
        per_rank_requests=per_rank_requests,
    )


def _max_observed_concurrency(requests: list[ContinuousBatchRequest]) -> int:
    events: list[tuple[float, int]] = []
    for request in requests:
        events.append((request.start_offset_seconds, 1))
        events.append((request.end_offset_seconds, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
        active += delta
        max_active = max(max_active, active)
    return max_active


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]

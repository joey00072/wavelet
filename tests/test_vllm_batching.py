from __future__ import annotations

import threading
import time

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.vllm import VLLMPolicyInferenceEngine, _OpenAIBatchRequest


def _request(index: int) -> _OpenAIBatchRequest:
    return _OpenAIBatchRequest(payload={"index": index}, done=threading.Event())


def test_openai_batch_collector_respects_max_size() -> None:
    config = RLConfig(
        inference={
            "vllm": {
                "openai_batch_wait_seconds": 0.0,
                "openai_batch_min_size": 1,
                "openai_batch_max_wait_seconds": 0.0,
                "openai_batch_max_size": 2,
            }
        }
    )
    engine = VLLMPolicyInferenceEngine(config)
    engine._openai_batch = [_request(0), _request(1), _request(2)]

    with engine._openai_batch_condition:
        batch = engine._collect_openai_batch_locked()

    assert [request.payload["index"] for request in batch] == [0, 1]
    assert [request.payload["index"] for request in engine._openai_batch] == [2]


def test_openai_batch_collector_waits_for_min_size() -> None:
    config = RLConfig(
        inference={
            "vllm": {
                "openai_batch_wait_seconds": 0.0,
                "openai_batch_min_size": 2,
                "openai_batch_max_wait_seconds": 0.2,
            }
        }
    )
    engine = VLLMPolicyInferenceEngine(config)
    engine._openai_batch = [_request(0)]

    def append_request() -> None:
        time.sleep(0.02)
        with engine._openai_batch_condition:
            engine._openai_batch.append(_request(1))
            engine._openai_batch_condition.notify()

    thread = threading.Thread(target=append_request)
    thread.start()
    with engine._openai_batch_condition:
        batch = engine._collect_openai_batch_locked()
    thread.join()

    assert [request.payload["index"] for request in batch] == [0, 1]

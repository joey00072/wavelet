from __future__ import annotations

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.entrypoints.rl_launcher import _trainer_device_group
from wavelet.entrypoints.rl_vllm_openai_server import _serve_args
from wavelet.inference.http import HTTPPolicyInferenceEngine


def test_colocate_launcher_defaults_trainer_to_inference_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate",
            "inference_cuda_visible_devices": "0",
        }
    )

    assert _trainer_device_group(config) == "0"


def test_sleep_colocate_launcher_defaults_trainer_to_inference_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate_sleep",
            "inference_cuda_visible_devices": "0",
        }
    )

    assert _trainer_device_group(config) == "0"


def test_colocate_launcher_allows_explicit_trainer_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate",
            "inference_cuda_visible_devices": "0",
            "trainer_cuda_visible_devices": "1",
        }
    )

    assert _trainer_device_group(config) == "1"


def test_colocate_launcher_requires_trainer_devices_for_multi_replica() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate",
            "inference_num_replicas": 2,
            "inference_cuda_visible_devices": ["0", "1"],
        }
    )

    with pytest.raises(ValueError, match="multiple inference replicas"):
        _trainer_device_group(config)


def test_sleep_colocate_requires_synchronous_rollouts() -> None:
    with pytest.raises(ValueError, match="requires synchronous rollouts"):
        RLConfig(
            launcher={"mode": "colocate_sleep"},
            orchestrator={"max_async_level": 1},
        )


def test_sleep_colocate_rejects_multi_trainer() -> None:
    with pytest.raises(ValueError, match="one trainer process"):
        RLConfig(
            launcher={
                "mode": "colocate_sleep",
                "trainer_num_processes": 2,
            }
        )


def test_sleep_colocate_enables_vllm_sleep_allocator() -> None:
    config = RLConfig(launcher={"mode": "colocate_sleep"})

    args = _serve_args(config)

    assert args.enable_sleep_mode is True


def test_http_inference_sleep_discards_vllm_gpu_allocations() -> None:
    config = RLConfig()
    engine = HTTPPolicyInferenceEngine(config)
    calls = []

    def fake_request_all(method: str, path: str, payload: dict) -> None:
        calls.append((method, path, payload))

    engine._request_all = fake_request_all  # type: ignore[method-assign]

    engine.sleep()

    assert calls == [("POST", "/sleep", {"level": 1})]

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
import yaml

from wavelet import debug as debug_module
from wavelet.configs.rl_config import RLConfig
from wavelet.inference import native_server
from wavelet.orchestrator.placement import nccl_inference_ranks
from wavelet.orchestrator.runtime import (
    _config_path_for_role,
    _config_with_nccl_inference_world_size,
    _role_specs,
)
from wavelet.transport import policy as policy_module
from wavelet.transport.policy import NCCLWeightUpdateWorker, nccl_world_size


def _nccl_http_config(
    *,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    policy_transfer: dict[str, Any] | None = None,
    **overrides: Any,
) -> RLConfig:
    return RLConfig(
        inference={
            "mode": "vllm_http",
            "vllm": {
                "server_backend": "openai",
                "tensor_parallel_size": tensor_parallel_size,
                "data_parallel_size": data_parallel_size,
            },
        },
        lora=None,
        reward={"mode": "reference_match"},
        policy_transfer={"type": "nccl", **(policy_transfer or {})},
        **overrides,
    )


class _FakeProcessGroup:
    created: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def create(
        cls,
        *,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        store_timeout: int,
    ) -> _FakeProcessGroup:
        cls.created.append(
            {
                "host": host,
                "port": port,
                "rank": rank,
                "world_size": world_size,
                "store_timeout": store_timeout,
            }
        )
        return cls()


class _FakeCommunicator:
    def __init__(self, process_group: Any, *, device: Any) -> None:
        self.process_group = process_group
        self.device = device


def _init_worker(
    monkeypatch,
    *,
    local_rank: int,
    rank_offset: int,
    inference_world_size: int,
) -> dict[str, Any]:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        policy_module,
        "_require_vllm_receiver_nccl",
        lambda: (_FakeCommunicator, _FakeProcessGroup),
    )
    worker = NCCLWeightUpdateWorker()
    worker.device = torch.device("cuda", local_rank)
    worker.init_broadcaster("127.0.0.1", 29501, rank_offset, inference_world_size, 30)
    return _FakeProcessGroup.created[-1]


def test_replicas_agree_with_trainer_on_nccl_world_size(monkeypatch) -> None:
    # Two TP=1 replicas: the trainer is rank 0, the replicas are ranks 1 and 2,
    # and every participant must report the same three-member group.
    replica_0 = _init_worker(
        monkeypatch, local_rank=0, rank_offset=1, inference_world_size=2
    )
    replica_1 = _init_worker(
        monkeypatch, local_rank=0, rank_offset=2, inference_world_size=2
    )

    assert (replica_0["rank"], replica_0["world_size"]) == (1, 3)
    assert (replica_1["rank"], replica_1["world_size"]) == (2, 3)
    assert nccl_world_size(2) == 3


def test_native_server_lifespan_uses_trainer_world_size(monkeypatch) -> None:
    for key in (
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
    ):
        monkeypatch.setenv(key, "preset")
    config = _nccl_http_config(
        policy_transfer={"nccl_inference_world_size": 2, "nccl_rank_offset": 2},
    )

    class _Engine:
        init_info: dict[str, Any] | None = None

        def setup(self) -> None:
            return None

        def close(self) -> None:
            return None

        def init_weight_transfer(self, init_info: dict[str, Any]) -> None:
            self.init_info = init_info

    engine = _Engine()

    async def run() -> None:
        async with native_server._lifespan(config, engine)(None):
            pass

    asyncio.run(run())

    assert engine.init_info is not None
    assert engine.init_info["rank_offset"] == 2
    assert engine.init_info["world_size"] == 3


def test_nccl_world_size_counts_vllm_workers_per_replica() -> None:
    config = _nccl_http_config(
        tensor_parallel_size=2,
        launcher={
            "inference_num_replicas": 2,
            "inference_cuda_visible_devices": ["0,1", "2,3"],
        },
    )

    resolved = _config_with_nccl_inference_world_size(config, inference_replicas=2)

    assert resolved.policy_transfer.nccl_inference_world_size == 4
    assert resolved.policy_transfer.nccl_rank_offset == 1


def test_nccl_world_size_rejects_visible_devices_without_matching_workers() -> None:
    config = _nccl_http_config(
        tensor_parallel_size=1,
        launcher={
            "inference_num_replicas": 1,
            "inference_cuda_visible_devices": "0,1",
        },
    )

    with pytest.raises(ValueError, match="exactly 1 CUDA device"):
        _config_with_nccl_inference_world_size(config, inference_replicas=1)


def test_nccl_inference_ranks_use_tensor_times_data_parallel_size() -> None:
    config = _nccl_http_config(tensor_parallel_size=1, data_parallel_size=2)

    assert nccl_inference_ranks(config, None) == 2
    assert nccl_inference_ranks(config, "4,5") == 2
    with pytest.raises(ValueError, match="lists 3"):
        nccl_inference_ranks(config, "4,5,6")


def test_role_specs_offset_replicas_by_vllm_worker_count(tmp_path: Path) -> None:
    config = _nccl_http_config(
        tensor_parallel_size=2,
        output_dir=tmp_path,
        launcher={
            "mode": "process",
            "inference_num_replicas": 2,
            "inference_cuda_visible_devices": ["0,1", "2,3"],
        },
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
        },
    )
    config = _config_with_nccl_inference_world_size(config, inference_replicas=2)

    roles = _role_specs(
        config,
        trainer_config_path=_config_path_for_role(config, "trainer", config),
        inference_config_path=_config_path_for_role(config, "inference", config),
        inference_ports=[8100, 8101],
    )

    replica_transfer = {
        role.name: yaml.safe_load(role.config_path.read_text())["policy_transfer"]
        for role in roles
        if role.name.startswith("inference_server_")
    }
    assert {
        name: transfer["nccl_rank_offset"]
        for name, transfer in replica_transfer.items()
    } == {"inference_server_0": 1, "inference_server_1": 3}
    assert {
        transfer["nccl_inference_world_size"] for transfer in replica_transfer.values()
    } == {4}


@pytest.mark.parametrize(
    ("tensor_parallel_size", "expected_status"),
    [(1, "error"), (2, "ok")],
)
def test_preflight_flags_nccl_replica_device_count_mismatch(
    monkeypatch,
    tensor_parallel_size: int,
    expected_status: str,
) -> None:
    monkeypatch.setattr(debug_module, "_available_gpu_indices", lambda: None)
    config = _nccl_http_config(
        tensor_parallel_size=tensor_parallel_size,
        launcher={
            "inference_num_replicas": 1,
            "inference_cuda_visible_devices": "0,1",
        },
    )

    checks = debug_module._device_group_checks(config)

    check = next(check for check in checks if check.name == "inference_devices_0")
    assert check.status == expected_status
    if expected_status == "error":
        assert "NCCL policy transfer" in check.message

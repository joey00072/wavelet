from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import wavelet.inference.vllm_weight_update as weight_update
from wavelet.utils.policy_transfer import NCCL_UPDATE_INFO_FILENAME


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.loaded: list[tuple[str, torch.Tensor]] = []

    def load_weights(self, weights) -> None:  # type: ignore[no-untyped-def]
        self.loaded.extend((name, tensor.clone()) for name, tensor in weights)


class _DummyRunner:
    def __init__(self) -> None:
        self.model = _DummyModel()
        self.model_config = object()


class _DummyCommunicator:
    device = torch.device("cpu")

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None) -> None:
        del src, stream
        tensor.fill_(3)


def test_nccl_weight_update_worker_loads_broadcast_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        weight_update,
        "process_weights_after_loading",
        lambda *args, **kwargs: None,
    )
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / NCCL_UPDATE_INFO_FILENAME).write_text(
        json.dumps(
            {
                "names": ["model.layers.0.weight"],
                "dtype_names": ["float32"],
                "shapes": [[2]],
                "packed": False,
            }
        )
    )
    worker = weight_update.NCCLWeightUpdateWorker()
    worker.model_runner = _DummyRunner()
    worker._wavelet_nccl_communicator = _DummyCommunicator()

    worker.update_weights_from_path(str(policy_dir))

    assert worker.model_runner.model.loaded[0][0] == "model.layers.0.weight"
    assert worker.model_runner.model.loaded[0][1].tolist() == [3.0, 3.0]

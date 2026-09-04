from __future__ import annotations

import json
from contextlib import nullcontext
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


def test_filesystem_weight_update_worker_uses_layerwise_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_dir = tmp_path / "model"
    policy_dir.mkdir()
    runner = _DummyRunner()
    worker = weight_update.FileSystemWeightUpdateWorker()
    worker.model_runner = runner
    worker.load_config = object()
    worker.vllm_config = object()
    weights = [("model.layers.0.weight", torch.tensor([2.0]))]
    source_args: list[object] = []
    lifecycle: list[str] = []

    class _Loader:
        Source = staticmethod(lambda *args, **kwargs: (args, kwargs))

        def _get_weights_iterator(self, source):  # type: ignore[no-untyped-def]
            source_args.append(source)
            return iter(weights)

    monkeypatch.setattr(weight_update, "get_model_loader", lambda _config: _Loader())
    monkeypatch.setattr(weight_update, "DefaultModelLoader", _Loader)
    monkeypatch.setattr(
        weight_update,
        "set_current_vllm_config",
        lambda config: nullcontext(config),
    )
    monkeypatch.setattr(
        weight_update,
        "initialize_layerwise_reload",
        lambda model: lifecycle.append(f"initialize:{model is runner.model}"),
    )
    monkeypatch.setattr(
        weight_update,
        "finalize_layerwise_reload",
        lambda model, config: lifecycle.append(
            f"finalize:{model is runner.model}:{config is runner.model_config}"
        ),
    )

    worker.update_weights_from_path(str(policy_dir))

    assert len(source_args) == 1
    assert lifecycle == ["initialize:True", "finalize:True:True"]
    assert runner.model.loaded[0][0] == "model.layers.0.weight"
    assert runner.model.loaded[0][1].tolist() == [2.0]


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

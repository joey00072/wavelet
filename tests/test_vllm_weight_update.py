from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar

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


class _QueuedCommunicator:
    def __init__(self, values: list[torch.Tensor]) -> None:
        self.values = values

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None) -> None:
        del src, stream
        tensor.copy_(self.values.pop(0))


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


def test_partition_state_dict_keeps_non_layer_weights_separate() -> None:
    state = {
        "model.embed_tokens.weight": torch.zeros(2),
        "model.layers.0.self_attn.weight": torch.zeros(3),
        "model.layers.1.self_attn.weight": torch.zeros(4),
        "lm_head.weight": torch.zeros(5),
    }

    groups = weight_update._partition_state_dict(state)

    assert [list(group) for group in groups] == [
        ["model.embed_tokens.weight", "lm_head.weight"],
        ["model.layers.0.self_attn.weight"],
        ["model.layers.1.self_attn.weight"],
    ]


def test_convert_layer_to_hf_runs_once_for_a_layer() -> None:
    calls: list[int] = []

    class _Model(nn.Module):
        def convert_layer_to_hf(self, state, layer_index):  # type: ignore[no-untyped-def]
            calls.append(layer_index)
            state["converted.weight"] = state.pop("trainer.weight")
            return state

    converted = weight_update._convert_layer_to_hf(
        _Model(),
        {"trainer.weight": torch.ones(2)},
        3,
    )

    assert calls == [3]
    assert list(converted) == ["converted.weight"]


def test_iter_layer_state_dicts_converts_each_partition() -> None:
    calls: list[int] = []

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Parameter(torch.zeros(1))
            self.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])

        def convert_layer_to_hf(self, state, layer_index):  # type: ignore[no-untyped-def]
            calls.append(layer_index)
            return state

    groups = list(weight_update._iter_layer_state_dicts(_Model()))

    assert len(groups) == 3
    assert calls == [-1, 0, 1]


def test_materialize_wire_tensors_limits_root_fsdp_summon_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeFSDP(nn.Module):
        summon_calls: ClassVar[list[dict[str, object]]] = []

        @classmethod
        def summon_full_params(cls, owner, **kwargs):  # type: ignore[no-untyped-def]
            del owner
            cls.summon_calls.append(kwargs)
            return nullcontext()

    class _Model(_FakeFSDP):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_FakeFSDP()])

    monkeypatch.setattr(
        torch.distributed.fsdp,
        "FullyShardedDataParallel",
        _FakeFSDP,
    )
    model = _Model()

    weight_update._materialize_wire_tensors(
        model,
        {"embed.weight": torch.ones(1)},
        -1,
    )
    weight_update._materialize_wire_tensors(
        model,
        {"layers.0.weight": torch.ones(1)},
        0,
    )

    assert [call["recurse"] for call in _FakeFSDP.summon_calls] == [False, True]


def test_nccl_weight_update_worker_loads_each_layer_with_mixed_dtypes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    metadata = [
        {"float32": [{"name": "model.a", "shape": [2], "numel": 2}]},
        {
            "float32": [{"name": "model.b", "shape": [1], "numel": 1}],
            "bfloat16": [{"name": "model.c", "shape": [2], "numel": 2}],
        },
    ]
    (policy_dir / NCCL_UPDATE_INFO_FILENAME).write_text(
        json.dumps({"protocol": "layerwise_v1"})
    )
    values: list[torch.Tensor] = [torch.tensor([2])]
    for layer in metadata:
        payload = json.dumps(layer).encode()
        values.extend([torch.tensor([len(payload)]), torch.tensor(list(payload))])
        for dtype_name, entries in layer.items():
            dtype = getattr(torch, dtype_name)
            values.append(
                torch.tensor(
                    [float(i) for i in range(sum(e["numel"] for e in entries))],
                    dtype=dtype,
                )
            )
    worker = weight_update.NCCLWeightUpdateWorker()
    runner = _DummyRunner()
    worker.model_runner = runner
    worker.vllm_config = object()
    worker._wavelet_nccl_communicator = _QueuedCommunicator(values)
    lifecycle: list[str] = []
    monkeypatch.setattr(
        weight_update,
        "set_current_vllm_config",
        lambda config: nullcontext(config),
    )
    monkeypatch.setattr(
        weight_update,
        "initialize_layerwise_reload",
        lambda model: lifecycle.append("initialize"),
    )
    monkeypatch.setattr(
        weight_update,
        "finalize_layerwise_reload",
        lambda model, config: lifecycle.append("finalize"),
    )

    worker.update_weights_from_path(str(policy_dir))

    assert lifecycle == ["initialize", "finalize"]
    assert [name for name, _ in runner.model.loaded] == [
        "model.a",
        "model.b",
        "model.c",
    ]
    assert runner.model.loaded[0][1].tolist() == [0.0, 1.0]
    assert runner.model.loaded[2][1].dtype == torch.bfloat16


def test_nccl_broadcaster_groups_each_layer_by_dtype() -> None:
    communicator = _QueuedCommunicator([])
    broadcaster = object.__new__(weight_update.NCCLWeightBroadcaster)
    broadcaster.rank = 0
    broadcaster.source_rank = 0
    broadcaster._device = torch.device("cpu")
    broadcaster._communicator = communicator
    calls: list[torch.Tensor] = []

    def broadcast(tensor: torch.Tensor, src: int, stream=None) -> None:
        del src, stream
        calls.append(tensor.clone())

    communicator.broadcast = broadcast  # type: ignore[method-assign]
    broadcaster.broadcast_layers(
        [
            {
                "model.a": torch.tensor([1.0, 2.0]),
                "model.b": torch.tensor([3.0], dtype=torch.bfloat16),
            }
        ],
        layer_count=1,
    )

    assert len(calls) == 5  # count, metadata size, metadata, and two dtype buffers
    assert calls[-2].dtype == torch.float32
    assert calls[-2].tolist() == [1.0, 2.0]
    assert calls[-1].dtype == torch.bfloat16
    assert calls[-1].tolist() == [3.0]


def test_nccl_broadcaster_resolves_unindexed_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    assert weight_update._indexed_cuda_device("cuda") == torch.device("cuda:3")
    assert weight_update._indexed_cuda_device("cuda:1") == torch.device("cuda:1")

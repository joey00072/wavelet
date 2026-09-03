from pathlib import Path
from typing import ClassVar

import pytest
import torch

from wavelet.kernels.utils import get_lora_parameters
from wavelet.trainer import lora as lora_utils


class _FakePeftModel:
    pass


def test_fsdp_lora_snapshot_uses_sharded_lora_gather(
    monkeypatch, tmp_path: Path
) -> None:
    wrapped = object()
    unwrapped = _FakePeftModel()
    gathered_state = {"base_model.model.layer.lora_A.weight": torch.ones(1)}
    saved_path = tmp_path / "adapter"
    calls: dict[str, object] = {}

    monkeypatch.setattr(lora_utils, "PeftModel", _FakePeftModel)
    monkeypatch.setattr(lora_utils, "unwrap_model", lambda model: unwrapped)

    def fake_gather(
        model: object,
        peft_model: _FakePeftModel,
        *,
        parallel_dims=None,
    ) -> dict[str, torch.Tensor]:
        calls["gather_model"] = model
        calls["gather_peft_model"] = peft_model
        calls["gather_parallel_dims"] = parallel_dims
        return gathered_state

    def fake_save(
        model: _FakePeftModel,
        output_dir: Path,
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
        is_main_process: bool = True,
        parallel_dims=None,
    ) -> Path:
        calls["save_model"] = model
        calls["save_output_dir"] = output_dir
        calls["save_state_dict"] = state_dict
        calls["save_is_main_process"] = is_main_process
        calls["save_parallel_dims"] = parallel_dims
        return saved_path

    monkeypatch.setattr(lora_utils, "_gather_fsdp_lora_state_dict", fake_gather)
    monkeypatch.setattr(lora_utils, "save_lora_adapter_snapshot", fake_save)

    result = lora_utils.save_lora_adapter_snapshot_from_fsdp(
        wrapped, tmp_path, is_main_process=True
    )

    assert result == saved_path
    assert calls == {
        "gather_model": wrapped,
        "gather_peft_model": unwrapped,
        "gather_parallel_dims": None,
        "save_model": unwrapped,
        "save_output_dir": tmp_path,
        "save_state_dict": gathered_state,
        "save_is_main_process": True,
        "save_parallel_dims": None,
    }


def test_fsdp_lora_gather_preserves_adapter_name_before_peft_filter(
    monkeypatch,
) -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))

    class FakeWrapped:
        def named_parameters(self):
            yield "_fsdp_wrapped_module.base.q_proj.lora_A.default.weight", parameter

    monkeypatch.setattr(lora_utils.torch.distributed, "is_initialized", lambda: False)

    state = lora_utils._gather_fsdp_lora_state_dict(FakeWrapped(), _FakePeftModel())

    assert list(state) == ["base.q_proj.lora_A.default.weight"]
    assert torch.equal(state["base.q_proj.lora_A.default.weight"], parameter)


def test_single_lora_guard_rejects_multiple_peft_adapters(monkeypatch) -> None:
    class FakePeftModel:
        peft_config: ClassVar = {"default": object(), "policy_old": object()}

        def active_adapters(self):
            return ["default"]

    monkeypatch.setattr(lora_utils, "PeftModel", FakePeftModel)

    with pytest.raises(RuntimeError, match="exactly one PEFT LoRA adapter"):
        lora_utils.enforce_single_lora_adapter(FakePeftModel())


def test_fsdp_lora_gather_filters_to_single_adapter(monkeypatch) -> None:
    default = torch.nn.Parameter(torch.ones(2, 2))
    old = torch.nn.Parameter(torch.full((2, 2), 9.0))

    class FakePeftModel:
        peft_config: ClassVar = {"default": object()}

        def active_adapters(self):
            return ["default"]

    class FakeWrapped:
        def named_parameters(self):
            yield "_fsdp_wrapped_module.base.q_proj.lora_A.default.weight", default
            yield "_fsdp_wrapped_module.base.q_proj.lora_A.policy_old.weight", old

    monkeypatch.setattr(lora_utils, "PeftModel", FakePeftModel)
    monkeypatch.setattr(lora_utils.torch.distributed, "is_initialized", lambda: False)

    state = lora_utils._gather_fsdp_lora_state_dict(FakeWrapped(), FakePeftModel())

    assert list(state) == ["base.q_proj.lora_A.default.weight"]
    assert torch.equal(state["base.q_proj.lora_A.default.weight"], default)


def test_fused_lora_parameters_reject_multiple_active_adapters() -> None:
    class FakeProj:
        disable_adapters = False
        merged = False
        active_adapters = ("default", "policy_old")
        base_layer = torch.nn.Linear(2, 2, bias=False)

    with pytest.raises(RuntimeError, match="exactly one active adapter"):
        get_lora_parameters(FakeProj())


def test_saved_lora_keys_strip_nested_fsdp_wrapped_module_segments() -> None:
    assert (
        lora_utils._strip_fsdp_wrapped_module_segments(
            "base_model.model.layers.0._fsdp_wrapped_module.mlp.up_proj.lora_A.weight"
        )
        == "base_model.model.layers.0.mlp.up_proj.lora_A.weight"
    )
    assert (
        lora_utils._strip_fsdp_wrapped_module_segments(
            "_fsdp_wrapped_module.base_model.model.layers.0.self_attn.q_proj."
            "lora_B.weight"
        )
        == "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
    )


def test_hf_tp_lora_gather_dim_follows_parallel_plan() -> None:
    class FakeBase:
        def __init__(self, plan: str) -> None:
            self._hf_tp_plan = plan

    class FakeModule:
        def __init__(self, plan: str) -> None:
            self.base_layer = FakeBase(plan)

    class FakeModel:
        def named_modules(self):
            yield "model.q_proj", FakeModule("colwise")
            yield "model.o_proj", FakeModule("rowwise")

    model = FakeModel()

    assert (
        lora_utils._hf_tp_lora_gather_dim(model, "model.q_proj.lora_B.default.weight")
        == 0
    )
    assert (
        lora_utils._hf_tp_lora_gather_dim(model, "model.o_proj.lora_A.default.weight")
        == 1
    )
    assert (
        lora_utils._hf_tp_lora_gather_dim(model, "model.q_proj.lora_A.default.weight")
        is None
    )


def test_hf_tp_lora_gather_uses_tp_mesh_group(monkeypatch) -> None:
    tp_group = object()
    calls: dict[str, object] = {}

    class FakeMesh:
        def get_group(self):
            return tp_group

    class FakeParallelDims:
        tp_enabled = True
        fsdp_enabled = False

        def get_mesh(self, name: str):
            assert name == "tp"
            return FakeMesh()

    class FakeBase:
        _hf_tp_plan = "colwise"

    class FakeModule:
        base_layer = FakeBase()

    class FakeModel:
        def named_modules(self):
            yield "model.q_proj", FakeModule()

    monkeypatch.setattr(lora_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        lora_utils.torch.distributed,
        "get_world_size",
        lambda group=None: 2,
    )
    monkeypatch.setattr(lora_utils.torch.distributed, "get_rank", lambda: 0)

    def fake_all_gather_object(gathered, local_state, group=None):
        calls["group"] = group
        gathered[0] = local_state
        gathered[1] = {"model.q_proj.lora_B.default.weight": torch.full((1, 2), 2.0)}

    monkeypatch.setattr(
        lora_utils.torch.distributed,
        "all_gather_object",
        fake_all_gather_object,
    )

    state = lora_utils._gather_hf_tp_lora_state_dict(
        FakeModel(),
        state_dict={"model.q_proj.lora_B.default.weight": torch.ones(1, 2)},
        parallel_dims=FakeParallelDims(),
    )

    assert calls == {"group": tp_group}
    assert torch.equal(
        state["model.q_proj.lora_B.default.weight"],
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )


def test_hf_tp_lora_syncs_only_replicated_grads(monkeypatch) -> None:
    calls: list[torch.Tensor] = []
    tp_group = object()

    class FakeMesh:
        def get_group(self):
            return tp_group

    class FakeParallelDims:
        tp_enabled = True
        fsdp_enabled = False

        def get_mesh(self, name: str):
            assert name == "tp"
            return FakeMesh()

    class FakeBase:
        def __init__(self, plan: str) -> None:
            self._hf_tp_plan = plan

    class FakeLora(torch.nn.Module):
        def __init__(self, plan: str) -> None:
            super().__init__()
            self.base_layer = FakeBase(plan)
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 2, bias=False)}
            )

    model = torch.nn.Sequential(FakeLora("colwise"), FakeLora("rowwise"))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    monkeypatch.setattr(lora_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        lora_utils.torch.distributed,
        "get_world_size",
        lambda group=None: 2,
    )

    def fake_all_reduce(tensor, *, op=None, group=None):
        assert group is tp_group
        calls.append(tensor)

    monkeypatch.setattr(lora_utils.torch.distributed, "all_reduce", fake_all_reduce)

    lora_utils.sync_hf_tp_lora_replicated_grads(model, FakeParallelDims())

    assert calls == [
        model[0].lora_A["default"].weight.grad,
        model[1].lora_B["default"].weight.grad,
    ]


def test_hf_tp_lora_prepare_reduces_rowwise_lora_output(monkeypatch) -> None:
    tp_mesh = object()
    calls: list[object] = []

    class FakeParallelDims:
        tp_enabled = True

        def get_mesh(self, name: str):
            assert name == "tp"
            return tp_mesh

    class FakeBase:
        _hf_tp_plan = "rowwise"

    class FakeLora(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = FakeBase()
            self.lora_A = torch.nn.ModuleDict()
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 2, bias=False)}
            )

    import transformers.integrations.tensor_parallel as tp_utils

    monkeypatch.setattr(lora_utils.torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce_forward(output, mesh):
        calls.append(mesh)
        return output + 1

    monkeypatch.setattr(tp_utils, "all_reduce_forward", fake_all_reduce_forward)

    model = FakeLora()
    x = torch.zeros(1, 2)
    base_output = model.lora_B["default"](x)
    lora_utils.prepare_hf_tp_lora_for_training(model, FakeParallelDims())
    reduced_output = model.lora_B["default"](x)

    assert calls == [tp_mesh]
    assert torch.equal(reduced_output, base_output + 1)


def test_fsdp_lora_gather_uses_hsdp_mesh_group(monkeypatch) -> None:
    hsdp_group = object()
    calls: dict[str, object] = {}

    class FakeMesh:
        def get_group(self):
            return hsdp_group

    class FakeParallelDims:
        tp_enabled = True
        fsdp_enabled = True

        def get_mesh(self, name: str):
            assert name == "hsdp"
            return FakeMesh()

    class FakeModule:
        def __init__(self) -> None:
            self.lora_A = {"default": torch.nn.Linear(2, 2, bias=False)}

    class FakeUnwrapped:
        def named_modules(self):
            yield "layer", FakeModule()

    class FakeWrapped:
        def named_parameters(self):
            yield (
                "_fsdp_wrapped_module.layer.lora_A.default.weight",
                torch.nn.Parameter(torch.ones(2)),
            )

    monkeypatch.setattr(lora_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        lora_utils.torch.distributed,
        "get_rank",
        lambda group=None: 0,
    )
    monkeypatch.setattr(
        lora_utils.torch.distributed,
        "get_world_size",
        lambda group=None: 2,
    )

    def fake_gather_object(
        local_state,
        gathered,
        *,
        group=None,
        group_dst=None,
        dst=None,
    ):
        calls["group"] = group
        calls["group_dst"] = group_dst
        calls["dst"] = dst
        gathered[0] = local_state
        gathered[1] = {"layer.lora_A.default.weight": torch.full((2,), 2.0)}

    monkeypatch.setattr(
        lora_utils.torch.distributed,
        "gather_object",
        fake_gather_object,
    )

    state = lora_utils._gather_fsdp_lora_state_dict(
        FakeWrapped(),
        FakeUnwrapped(),
        parallel_dims=FakeParallelDims(),
    )

    assert calls == {"group": hsdp_group, "group_dst": 0, "dst": None}
    assert torch.equal(
        state["layer.lora_A.default.weight"],
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )

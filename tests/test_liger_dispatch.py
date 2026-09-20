import sys
from types import ModuleType, SimpleNamespace

import pytest

from wavelet.trainer import model as model_utils


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_vl", "qwen3_moe"])
def test_liger_rejects_unsupported_architecture_despite_qwen3_filename(
    monkeypatch, model_type: str
) -> None:
    monkeypatch.setattr(
        model_utils.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type=model_type),
    )
    # Reject before importing optional kernels, independently of their availability.
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", None)
    with pytest.raises(ValueError, match="unsupported.*model_type"):
        model_utils.apply_liger_kernel("liger_fused", "/checkpoints/qwen3-latest")


@pytest.mark.parametrize(
    ("model_type", "patch_name"),
    [
        ("qwen3", "apply_liger_kernel_to_qwen3"),
        ("qwen2", "apply_liger_kernel_to_qwen2"),
        ("llama", "apply_liger_kernel_to_llama"),
        ("mistral", "apply_liger_kernel_to_mistral"),
    ],
)
@pytest.mark.parametrize("loss_impl", ["liger", "liger_fused"])
def test_liger_dispatches_actual_architecture_from_neutral_path(
    monkeypatch, model_type: str, patch_name: str, loss_impl: str
) -> None:
    config_calls = []
    patch_calls = []

    def load_config(source, **kwargs):
        config_calls.append((source, kwargs))
        return SimpleNamespace(model_type=model_type)

    monkeypatch.setattr(model_utils.AutoConfig, "from_pretrained", load_config)
    module = ModuleType("liger_kernel.transformers")
    module.monkey_patch = SimpleNamespace(
        **{patch_name: lambda **kwargs: patch_calls.append(kwargs)}
    )
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", module)

    model_utils.apply_liger_kernel(
        loss_impl, "/checkpoints/run-42", trust_remote_code=True
    )

    assert config_calls == [("/checkpoints/run-42", {"trust_remote_code": True})]
    assert patch_calls == [
        {
            "rope": True,
            "rms_norm": True,
            "swiglu": True,
            "cross_entropy": False,
            "fused_linear_cross_entropy": loss_impl == "liger_fused",
        }
    ]

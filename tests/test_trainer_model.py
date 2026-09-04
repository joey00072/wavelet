from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from wavelet.configs.sft import ModelConfig
from wavelet.trainer import model as model_utils
from wavelet.trainer.model import setup_tokenizer


class _Tokenizer:
    pad_token = None
    eos_token = "<eos>"
    chat_template = "original"
    padding_side = "right"


def test_setup_tokenizer_falls_back_from_adapter_to_base_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path | str] = []
    tokenizer = _Tokenizer()

    def from_pretrained(source: Path | str, **_: object) -> _Tokenizer:
        calls.append(source)
        if source == Path("outputs/policy/adapter"):
            raise ValueError("adapter does not contain tokenizer artifacts")
        return tokenizer

    monkeypatch.setattr(
        "wavelet.trainer.model.AutoTokenizer.from_pretrained", from_pretrained
    )

    result = setup_tokenizer(
        ModelConfig(
            name="org/base-model",
            adapter_path=Path("outputs/policy/adapter"),
        )
    )

    assert result is tokenizer
    assert calls == [Path("outputs/policy/adapter"), "org/base-model"]
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "left"


def test_setup_tokenizer_does_not_hide_base_model_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def from_pretrained(source: Path | str, **_: object) -> _Tokenizer:
        raise ValueError(f"cannot load {source}")

    monkeypatch.setattr(
        "wavelet.trainer.model.AutoTokenizer.from_pretrained", from_pretrained
    )

    with pytest.raises(ValueError, match="cannot load org/base-model"):
        setup_tokenizer(ModelConfig(name="org/base-model"))


def test_auto_attention_falls_back_to_sdpa_without_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_utils, "_flash_attention_available", lambda: False)

    assert model_utils._best_attn_implementation() == "sdpa"


def test_explicit_flash_attention_requires_importable_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_utils, "_flash_attention_available", lambda: False)

    with pytest.raises(ImportError, match="uv sync --extra flash-attn"):
        model_utils._model_load_kwargs(
            ModelConfig(attn_implementation="flash_attention_2"),
            distributed=False,
            parallel_dims=None,
        )


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_setup_runtime_applies_configured_matmul_precision(
    monkeypatch: pytest.MonkeyPatch,
    precision: Literal["highest", "high", "medium"],
) -> None:
    applied: list[str] = []
    monkeypatch.setattr(model_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        model_utils.torch,
        "set_float32_matmul_precision",
        applied.append,
    )

    model_utils.setup_runtime(ModelConfig(matmul_precision=precision))

    assert applied == [precision]


def test_removed_allow_tf32_model_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="allow_tf32"):
        ModelConfig.model_validate({"allow_tf32": True})

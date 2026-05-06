from __future__ import annotations

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import ModelConfig
from wavelet.trainer.model import _fsdp_mixed_precision
from wavelet.trainer.rl_trainer import (
    RLTrainer,
    _packed_causal_attention_mask,
    _packed_training_attention_mask,
)


class _Config:
    def __init__(self, attn_implementation: str) -> None:
        self._attn_implementation = attn_implementation


class _Model:
    def __init__(self, attn_implementation: str) -> None:
        self.config = _Config(attn_implementation)


class _Wrapper:
    def __init__(self, module: object) -> None:
        self._fsdp_wrapped_module = module


class _LogitModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"logits": self.logits}


class _LogprobModel:
    def __init__(self, logprobs: torch.Tensor) -> None:
        self.logprobs = logprobs

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"logprobs": self.logprobs}


def test_packed_causal_attention_mask_blocks_cross_sample_attention() -> None:
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    mask = _packed_causal_attention_mask(attention_mask, position_ids)

    assert mask is not None
    assert mask.shape == (1, 1, 5, 5)
    allowed = mask[0, 0] == 0
    assert allowed.tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [False, False, False, True, False],
        [False, False, False, True, True],
    ]


def test_unpacked_full_attention_mask_is_dropped() -> None:
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    position_ids = torch.arange(5).unsqueeze(0)

    mask = _packed_causal_attention_mask(attention_mask, position_ids)

    assert mask is None


def test_packed_reward_mean_is_weighted_by_rollout_count() -> None:
    trainer = RLTrainer(RLConfig())
    rewards = torch.tensor([0.0, 1.0])
    sample_counts = torch.tensor([1, 3])

    assert trainer._reward_mean(rewards, sample_counts=sample_counts) == 0.75  # noqa: SLF001


def test_rollout_reward_metric_is_weighted_by_rollout_count() -> None:
    trainer = RLTrainer(RLConfig())

    metrics = trainer._aggregate_rollout_metrics(  # noqa: SLF001
        [
            {"reward/all/mean": 0.0, "rollout/count": 1.0},
            {"reward/all/mean": 1.0, "rollout/count": 3.0},
        ]
    )
    metrics = trainer._finalize_synced_metrics(metrics)  # noqa: SLF001

    assert metrics["reward/all/mean"] == 0.75
    assert metrics["reward_mean"] == 0.75
    assert "_reward_weighted_sum" not in metrics
    assert "_reward_weight" not in metrics


def test_gradient_accumulation_loss_scale_divides_grads_once() -> None:
    trainer = RLTrainer(RLConfig())
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[6.0, 12.0]])
    trainer.model = model
    trainer._gradient_accumulation_loss_scale = 6.0  # noqa: SLF001

    trainer._apply_gradient_accumulation_loss_scale()  # noqa: SLF001

    assert torch.allclose(model.weight.grad, torch.tensor([[1.0, 2.0]]))


def test_packed_flash_attention_uses_varlen_position_ids() -> None:
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    mask = _packed_training_attention_mask(
        _Wrapper(_Model("flash_attention_2")),
        attention_mask,
        position_ids,
    )

    assert mask is None


def test_packed_fa4_attention_uses_varlen_position_ids() -> None:
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    mask = _packed_training_attention_mask(
        _Wrapper(_Model("fa4")),
        attention_mask,
        position_ids,
    )

    assert mask is None


def test_packed_flash_attention_rejects_padded_rows() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.long)

    try:
        _packed_training_attention_mask(
            _Model("flash_attention_2"),
            attention_mask,
            position_ids,
        )
    except ValueError as exc:
        assert "pad-free packed rows" in str(exc)
    else:
        raise AssertionError("expected padded packed FlashAttention rows to fail")


def test_model_logprobs_uses_fp32_for_bfloat16_logits() -> None:
    logits = torch.randn(2, 3, 7, dtype=torch.bfloat16)
    targets = torch.tensor([[1, 2, 3], [3, 4, 5]])
    temperatures = torch.ones((2, 3), dtype=torch.float32)
    trainer = RLTrainer(RLConfig())
    trainer.model = _LogitModel(logits)  # type: ignore[assignment]

    actual = trainer._model_logprobs(  # noqa: SLF001
        {
            "input_ids": targets,
            "position_ids": targets,
            "target_ids": targets,
            "temperatures": temperatures,
        },
        attention_mask=None,
    )
    expected = logits.float().log_softmax(dim=-1).gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected)


def test_model_logprobs_casts_chunked_output_to_fp32() -> None:
    logprobs = torch.randn(2, 3, dtype=torch.bfloat16)
    trainer = RLTrainer(RLConfig())
    trainer.model = _LogprobModel(logprobs)  # type: ignore[assignment]

    actual = trainer._model_logprobs(  # noqa: SLF001
        {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "position_ids": torch.ones((2, 3), dtype=torch.long),
            "target_ids": torch.ones((2, 3), dtype=torch.long),
            "temperatures": torch.ones((2, 3), dtype=torch.float32),
        },
        attention_mask=None,
    )

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, logprobs.float())


def test_float32_fsdp_config_uses_prime_style_mixed_precision(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    policy = _fsdp_mixed_precision(ModelConfig(torch_dtype="float32"))  # noqa: SLF001

    assert policy is not None
    assert policy.param_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.float32
    assert policy.buffer_dtype is torch.bfloat16

from __future__ import annotations

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import ModelConfig
from wavelet.distributed.world import World
from wavelet.trainer import model as model_utils
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
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        self.kwargs = kwargs
        return {"logits": self.logits}


class _LogprobModel:
    def __init__(self, logprobs: torch.Tensor) -> None:
        self.logprobs = logprobs
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        self.kwargs = kwargs
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
    model = _LogitModel(logits)
    trainer.model = model  # type: ignore[assignment]

    actual = trainer._model_logprobs(  # noqa: SLF001
        {
            "input_ids": targets,
            "position_ids": targets,
            "target_ids": targets,
            "temperatures": temperatures,
        },
        attention_mask=None,
    )
    expected = (
        logits.float()
        .log_softmax(dim=-1)
        .gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected)
    assert model.kwargs is not None
    assert "labels" not in model.kwargs
    assert "temperature" not in model.kwargs


def test_model_logprobs_casts_chunked_output_to_fp32() -> None:
    logprobs = torch.randn(2, 3, dtype=torch.bfloat16)
    config = RLConfig()
    config.model.fused_lm_head_token_chunk_size = 1024
    trainer = RLTrainer(config)
    model = _LogprobModel(logprobs)
    trainer.model = model  # type: ignore[assignment]

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
    assert model.kwargs is not None
    assert "labels" in model.kwargs
    assert "temperature" in model.kwargs


def test_float32_fsdp_config_uses_bfloat16_params_and_float32_reduce(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    policy = _fsdp_mixed_precision(ModelConfig(torch_dtype="float32"))  # noqa: SLF001

    assert policy is not None
    assert policy.param_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.float32
    assert policy.buffer_dtype is torch.bfloat16


def test_distributed_qlora_uses_current_cuda_device_map(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeConfig:
        quantization_config = None

    class FakeModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"use_cache": True})()

        def named_modules(self):
            return iter(())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        model_utils.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: FakeConfig(),
    )
    monkeypatch.setattr(
        model_utils,
        "BitsAndBytesConfig",
        lambda **kwargs: {"bitsandbytes": kwargs},
    )
    monkeypatch.setattr(
        model_utils,
        "prepare_model_for_kbit_training",
        lambda model, **kwargs: model,
    )

    def fake_from_pretrained(*args, **kwargs):
        del args
        captured_kwargs.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(
        model_utils.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )

    model_utils.setup_model(
        ModelConfig(
            name="fake/model",
            attn_implementation="sdpa",
            load_in_4bit=True,
            gradient_checkpointing=False,
        ),
        distributed=True,
    )

    assert captured_kwargs["device_map"] == {"": 2}
    assert "quantization_config" in captured_kwargs


def test_prepare_kbit_model_can_skip_float32_cast() -> None:
    class FakeKbitModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
            self.input_grads_enabled = False

        def enable_input_require_grads(self) -> None:
            self.input_grads_enabled = True

    model = FakeKbitModel()

    prepared = model_utils.prepare_kbit_model(
        model,  # type: ignore[arg-type]
        gradient_checkpointing=True,
        cast_non_quantized_to_float32=False,
    )

    assert prepared is model
    assert model.linear.weight.dtype is torch.bfloat16
    assert model.linear.weight.requires_grad is False
    assert model.input_grads_enabled is True


def test_qlora_ddp_does_not_move_quantized_model(monkeypatch) -> None:
    class FakeModel:
        def to(self, device: torch.device):
            raise AssertionError(f"QLoRA model should not be moved with .to({device})")

    calls: dict[str, object] = {}

    def fake_ddp(model, **kwargs):
        calls["model"] = model
        calls["kwargs"] = kwargs
        return model

    monkeypatch.setattr(model_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(model_utils, "DDP", fake_ddp)

    model = FakeModel()
    wrapped = model_utils.maybe_wrap_ddp(
        model,  # type: ignore[arg-type]
        model_config=ModelConfig(load_in_4bit=True),
        world=World(
            rank=0,
            local_rank=1,
            world_size=2,
            local_world_size=2,
            device=torch.device("cuda"),
        ),
    )

    assert wrapped is model
    assert calls["model"] is model
    assert calls["kwargs"] == {"device_ids": [1], "output_device": 1}

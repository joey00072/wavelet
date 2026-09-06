from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import ModelConfig
from wavelet.data.rl import (
    PackedRLDataset,
    RLDataset,
    RLExample,
    setup_rl_dataloader,
)
from wavelet.trainer import model as model_utils
from wavelet.trainer.ckpt import TrainerState
from wavelet.trainer.distributed import World
from wavelet.trainer.model import _fsdp_mixed_precision
from wavelet.trainer.rl import (
    RLTrainer,
    _packed_causal_attention_mask,
    _packed_training_attention_mask,
)
from wavelet.trainer.types import LossOutput, TrainOutput


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

    assert trainer._reward_mean(rewards, sample_counts=sample_counts) == 0.75


def test_rollout_reward_metric_is_weighted_by_rollout_count() -> None:
    trainer = RLTrainer(RLConfig())

    metrics = trainer._aggregate_rollout_metrics(
        [
            {"reward/all/mean": 0.0, "rollout/count": 1.0},
            {"reward/all/mean": 1.0, "rollout/count": 3.0},
        ]
    )
    metrics = trainer._finalize_synced_metrics(metrics)

    assert metrics["reward/all/mean"] == 0.75
    assert metrics["reward_mean"] == 0.75
    assert "_reward_weighted_sum" not in metrics
    assert "_reward_weight" not in metrics


def test_ref_kl_only_train_output_logs_without_rl_metrics() -> None:
    trainer = RLTrainer(RLConfig())
    trainer.monitor = Mock()
    trainer.step = 1
    progress = Mock()
    output = TrainOutput(
        loss=LossOutput(loss=torch.tensor(0.3)),
        stepped=True,
        metrics={
            "loss": 0.3,
            "ref_kl/unmasked_mismatch_kl": 0.002,
        },
    )

    trainer._log_train_output(output, progress)

    progress.set_postfix.assert_called_once_with(
        loss="0.3000",
        kl="0.0020",
        lr="0.00e+00",
    )


def test_gradient_accumulation_loss_scale_divides_grads_once() -> None:
    trainer = RLTrainer(RLConfig())
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[6.0, 12.0]])
    trainer.model = model
    trainer._gradient_accumulation_loss_scale = 6.0

    trainer._apply_gradient_accumulation_loss_scale()

    assert torch.allclose(model.weight.grad, torch.tensor([[1.0, 2.0]]))


def test_dynamic_loss_scale_normalizes_accumulated_raw_gradients() -> None:
    trainer = RLTrainer(RLConfig(data={"num_workers": 1}, max_grad_norm=0.0))
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    trainer._optimizer_batch_loss_scale = None
    trainer._dynamic_loss_scale_local = 6.0
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[6.0, 12.0]])
    trainer.model = model
    trainer.optimizer = Mock()
    trainer.scheduler = Mock()

    trainer._apply_optimizer_step()

    assert torch.allclose(model.weight.grad, torch.tensor([[1.0, 2.0]]))
    assert trainer._dynamic_loss_scale_local == 0.0


def test_dynamic_loss_backward_accumulates_sums_before_final_scaling() -> None:
    trainer = RLTrainer(RLConfig(data={"num_workers": 1}))
    trainer.accumulation_steps = 2
    trainer._optimizer_batch_loss_scale = None
    loss = torch.tensor(6.0, requires_grad=True)

    trainer._backward_rl_loss(loss)

    assert loss.grad is not None
    assert loss.grad.item() == pytest.approx(1.0)


def test_standalone_rl_train_honors_zero_max_steps() -> None:
    trainer = RLTrainer(RLConfig(max_steps=0))
    trainer.train_until = Mock()

    trainer.train()

    trainer.train_until.assert_called_once_with(0, finish_run=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_loss_aborts_and_clears_accumulated_gradients(value) -> None:
    trainer = RLTrainer(RLConfig())
    trainer.optimizer = Mock()

    with pytest.raises(FloatingPointError, match="Non-finite RL loss"):
        trainer._require_finite_loss(
            torch.tensor(value),
            label="RL loss",
        )

    trainer.optimizer.zero_grad.assert_called_once_with(set_to_none=True)


def test_remote_nonfinite_loss_aborts_before_backward(monkeypatch) -> None:
    trainer = RLTrainer(RLConfig())
    trainer.optimizer = Mock()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def mark_remote_failure(flag, *, op) -> None:
        assert op == torch.distributed.ReduceOp.MIN
        flag.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", mark_remote_failure)

    with pytest.raises(FloatingPointError, match="another rank"):
        trainer._require_finite_loss(
            torch.tensor(1.0),
            label="RL loss",
        )


def test_finalize_waits_for_pending_async_checkpoint(monkeypatch) -> None:
    trainer = RLTrainer(RLConfig())
    trainer.monitor = Mock()
    trainer.ckpt_manager = Mock()
    monkeypatch.setattr(trainer, "_save_model", Mock())

    trainer.finalize(status="completed")

    trainer.ckpt_manager.save.assert_called_once_with(
        TrainerState(step=0, micro_step=0),
        dataloader=None,
        force=True,
    )
    trainer.ckpt_manager.wait_for_pending_save.assert_called_once_with()


def test_orchestrated_rl_resume_accepts_dynamic_micro_step_count() -> None:
    trainer = RLTrainer(RLConfig())
    trainer.accumulation_steps = 2

    trainer._validate_resume_state(TrainerState(step=100, micro_step=1600))


def test_orchestrated_rl_checkpoint_excludes_transient_dataloader() -> None:
    trainer = RLTrainer(RLConfig())
    trainer.dataloader = object()

    assert trainer._checkpoint_dataloader() is None


def test_unpacked_loss_scale_counts_every_example_in_optimizer_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RLConfig(data={"batch_size": 32, "micro_batch_size": 16, "seq_len": 8})
    trainer = RLTrainer(config)
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    trainer.accumulation_steps = 2
    trainer.dataset = RLDataset(
        records=[
            RLExample(
                prompt=[],
                completion=[],
                advantage=1.0,
                reward=1.0,
                input_ids=[1, 2],
                target_ids=[2, 3],
                loss_mask=[True, True],
                inference_logprobs=[-1.0, -1.0],
                temperatures=[1.0, 1.0],
            )
            for _ in range(32)
        ],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=config.data,
    )

    captured: dict[str, bool] = {}

    def fake_average(
        scales: dict[str, float | int],
        *,
        scales_are_cp_replicated: bool = False,
    ) -> dict[str, float]:
        captured["scales_are_cp_replicated"] = scales_are_cp_replicated
        return {name: float(value) for name, value in scales.items()}

    monkeypatch.setattr(trainer, "_average_data_parallel_loss_scales", fake_average)

    assert trainer._estimate_optimizer_batch_loss_scales() == {
        "rl": 64.0,
        "ce": 0.0,
        "ref_kl": 0.0,
    }
    assert captured["scales_are_cp_replicated"] is True


def test_optimizer_batch_scales_count_each_loss_component_independently() -> None:
    config = RLConfig(data={"batch_size": 2, "micro_batch_size": 1, "seq_len": 8})
    records = [_rl_example(2), _rl_example(2)]
    records[0].ce_weight = [1.0, 0.0]
    records[0].ref_kl_weight = [1.0, 1.0]
    records[0].teacher_logprobs = [-0.5, -0.5]
    records[1].ce_weight = [2.0, 3.0]

    trainer = RLTrainer(config)
    trainer.world = _cpu_world()
    trainer.accumulation_steps = 2
    trainer.dataset = RLDataset(
        records=records,
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=config.data,
    )

    assert trainer._estimate_optimizer_batch_loss_scales() == {
        "rl": 4.0,
        "ce": 3.0,
        "ref_kl": 2.0,
    }


def test_orchestrated_trainer_defers_loss_scale_until_rollouts_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RLConfig(
        data={"batch_size": 2, "micro_batch_size": 1, "seq_len": 8},
        orchestrator={"enabled": True},
    )
    trainer = RLTrainer(config)
    trainer.tokenizer = Mock(pad_token_id=0)
    trainer.world = _cpu_world()
    dataset = Mock()
    dataloader = Mock()
    monkeypatch.setattr(
        "wavelet.trainer.rl.setup_rl_dataset", Mock(return_value=dataset)
    )
    monkeypatch.setattr(
        "wavelet.trainer.rl.setup_rl_dataloader", Mock(return_value=dataloader)
    )
    estimate = Mock(side_effect=AssertionError("raw prompts are not trainable"))
    monkeypatch.setattr(trainer, "_estimate_optimizer_batch_loss_scales", estimate)

    trainer._setup_data()

    assert trainer.dataset is dataset
    assert trainer.dataloader is dataloader
    assert trainer._optimizer_batch_loss_scales is None
    estimate.assert_not_called()

    trainer._rollout_batch_loaded = True
    estimate.return_value = {"rl": 4.0, "ce": 0.0, "ref_kl": 0.0}
    estimate.side_effect = None
    trainer._setup_data()

    estimate.assert_called_once_with()
    assert trainer._optimizer_batch_loss_scales == {
        "rl": 4.0,
        "ce": 0.0,
        "ref_kl": 0.0,
    }


def test_loss_scale_uses_global_token_mean_for_averaged_dp_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = RLTrainer(RLConfig())
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
    )
    dp_group = object()

    class _Mesh:
        def get_group(self) -> object:
            return dp_group

    class _ParallelDims:
        dp_replicate = 1
        dp_shard = 2
        cp = 1
        tp = 1
        ep = 1

        def get_mesh(self, name: str) -> _Mesh:
            assert name == "dp"
            return _Mesh()

    trainer.parallel_dims = _ParallelDims()  # type: ignore[assignment]
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(
        tensor: torch.Tensor,
        *,
        op: object,
        group: object,
    ) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert group is dp_group
        assert tensor.tolist() == [4.0, 0.0, 0.0]
        tensor.add_(torch.tensor([8.0, 6.0, 2.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert trainer._average_data_parallel_loss_scale(4.0) == 6.0


def test_loss_scale_uses_dp_cp_for_context_parallel_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = RLTrainer(RLConfig())
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
    )
    dp_cp_group = object()

    class _Mesh:
        def get_group(self) -> object:
            return dp_cp_group

    class _ParallelDims:
        dp_replicate = 1
        dp_shard = 1
        cp = 2
        tp = 1
        ep = 1

        def get_mesh(self, name: str) -> _Mesh:
            assert name == "dp_cp"
            return _Mesh()

    trainer.parallel_dims = _ParallelDims()  # type: ignore[assignment]
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(
        tensor: torch.Tensor,
        *,
        op: object,
        group: object,
    ) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert group is dp_cp_group
        assert tensor.tolist() == [4.0, 0.0, 0.0]
        tensor.add_(torch.tensor([4.0, 0.0, 0.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert trainer._average_data_parallel_loss_scale(4.0) == 4.0


def test_precomputed_loss_scale_is_localized_before_dp_cp_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = RLTrainer(RLConfig())
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
    )
    dp_cp_group = object()

    class _Mesh:
        def get_group(self) -> object:
            return dp_cp_group

    class _ParallelDims:
        dp_replicate = 1
        dp_shard = 1
        cp = 2
        tp = 1
        ep = 1

        def get_mesh(self, name: str) -> _Mesh:
            assert name == "dp_cp"
            return _Mesh()

    trainer.parallel_dims = _ParallelDims()  # type: ignore[assignment]
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(
        tensor: torch.Tensor,
        *,
        op: object,
        group: object,
    ) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert group is dp_cp_group
        # The full-row estimate is 4. CP-localize it before reduction.
        assert tensor.tolist() == [2.0, 0.0, 0.0]
        tensor.add_(torch.tensor([2.0, 0.0, 0.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert (
        trainer._average_data_parallel_loss_scales(
            {"rl": 4.0, "ce": 0.0, "ref_kl": 0.0},
            scales_are_cp_replicated=True,
        )["rl"]
        == 2.0
    )


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

    actual = trainer._model_logprobs(
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

    actual = trainer._model_logprobs(
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


def test_model_logprobs_accepts_masked_chunked_output() -> None:
    config = RLConfig()
    config.model.fused_lm_head_token_chunk_size = 1024
    trainer = RLTrainer(config)
    logprobs = torch.tensor([[-0.2, -0.4]])
    model = _LogprobModel(logprobs)
    trainer.model = model  # type: ignore[assignment]

    actual = trainer._model_logprobs(
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "position_ids": torch.zeros((1, 2), dtype=torch.long),
            "target_ids": torch.ones((1, 2), dtype=torch.long),
            "temperatures": torch.ones((1, 2), dtype=torch.float32),
            "loss_mask": torch.ones((1, 2), dtype=torch.bool),
            "sampling_mask_ids": torch.tensor([[[1], [1]]]),
            "sampling_mask_lengths": torch.ones((1, 2), dtype=torch.long),
        },
        attention_mask=None,
    )

    torch.testing.assert_close(actual, logprobs)


def test_model_logprobs_rejects_missing_required_sampling_masks() -> None:
    trainer = RLTrainer(RLConfig(inference={"sampling": {"top_k": 8}}))
    trainer.model = _LogitModel(torch.zeros(1, 2, 5))  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="trainable token has no sampling mask"):
        trainer._model_logprobs(
            {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "position_ids": torch.zeros((1, 2), dtype=torch.long),
                "target_ids": torch.ones((1, 2), dtype=torch.long),
                "temperatures": torch.ones((1, 2), dtype=torch.float32),
                "loss_mask": torch.ones((1, 2), dtype=torch.bool),
                "sampling_mask_ids": torch.zeros((1, 2, 1), dtype=torch.long),
                "sampling_mask_lengths": torch.zeros((1, 2), dtype=torch.long),
            },
            attention_mask=None,
        )


def test_entropy_metrics_cover_only_loss_masked_tokens() -> None:
    trainer = RLTrainer(RLConfig())
    metrics = trainer._entropy_metrics(
        torch.tensor([[1.0, 2.0, 100.0], [3.0, 4.0, 200.0]]),
        torch.tensor([[True, True, False], [True, True, False]]),
    )

    assert metrics["entropy/mean"].item() == pytest.approx(2.5)
    assert metrics["entropy/min"].item() == pytest.approx(1.0)
    assert metrics["entropy/max"].item() == pytest.approx(4.0)
    assert metrics["_entropy_sum"].item() == pytest.approx(10.0)
    assert metrics["_entropy_count"].item() == pytest.approx(4.0)


def test_entropy_mean_is_token_weighted_across_micro_batches() -> None:
    trainer = RLTrainer(RLConfig())
    metrics = trainer._aggregate_train_metrics(
        [
            {
                "_entropy_sum": 2.0,
                "_entropy_count": 1.0,
                "entropy/mean": 2.0,
                "entropy/min": 2.0,
                "entropy/max": 2.0,
            },
            {
                "_entropy_sum": 12.0,
                "_entropy_count": 3.0,
                "entropy/mean": 4.0,
                "entropy/min": 3.0,
                "entropy/max": 5.0,
            },
        ]
    )
    metrics = trainer._finalize_synced_metrics(metrics)

    assert metrics["entropy/mean"] == pytest.approx(3.5)
    assert metrics["entropy/min"] == pytest.approx(2.0)
    assert metrics["entropy/max"] == pytest.approx(5.0)
    assert "_entropy_sum" not in metrics
    assert "_entropy_count" not in metrics


def test_float32_fsdp_config_uses_bfloat16_params_and_float32_reduce(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    policy = _fsdp_mixed_precision(ModelConfig(torch_dtype="float32"))

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
            activation_checkpointing=None,
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


def _rl_example(trainable_tokens: int) -> RLExample:
    return RLExample(
        prompt=[],
        completion=[],
        advantage=1.0,
        reward=1.0,
        input_ids=list(range(trainable_tokens)),
        target_ids=list(range(1, trainable_tokens + 1)),
        loss_mask=[True] * trainable_tokens,
        inference_logprobs=[-1.0] * trainable_tokens,
        temperatures=[1.0] * trainable_tokens,
    )


def _cpu_world() -> World:
    return World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )


def test_after_resume_uses_measured_scale_until_dataloader_state_applies() -> None:
    config = RLConfig(data={"batch_size": 2, "micro_batch_size": 1, "seq_len": 8})
    records = [_rl_example(2), _rl_example(2), _rl_example(4), _rl_example(4)]

    def _trainer() -> RLTrainer:
        trainer = RLTrainer(config)
        trainer.world = _cpu_world()
        trainer.accumulation_steps = 2
        trainer.dataset = RLDataset(
            records=list(records),
            tokenizer=None,  # type: ignore[arg-type]
            seq_len=8,
            data_config=config.data,
        )
        trainer.dataloader = setup_rl_dataloader(
            trainer.dataset, config.data, pad_token_id=0
        )
        return trainer

    saved = _trainer()
    saved_scales = saved._estimate_optimizer_batch_loss_scales()
    assert saved_scales is not None
    saved._optimizer_batch_loss_scale = saved_scales["rl"]
    assert saved._optimizer_batch_loss_scale == 4.0
    iterator = iter(saved.dataloader)
    next(iterator)
    next(iterator)
    state = saved.dataloader.state_dict()

    resumed = _trainer()
    resumed_scales = resumed._estimate_optimizer_batch_loss_scales()
    assert resumed_scales is not None
    resumed._optimizer_batch_loss_scale = resumed_scales["rl"]
    resumed.dataloader.load_state_dict(state)
    # StatefulDataLoader defers dataset state until the next iterator exists,
    # so a static estimate taken here would describe the step-0 batch (4 tokens)
    # even though the next optimizer batch holds 8.
    assert resumed.dataset.step == 0
    resumed._after_resume()
    assert resumed._optimizer_batch_loss_scale is None

    resumed_iterator = iter(resumed.dataloader)
    first = next(resumed_iterator)
    second = next(resumed_iterator)
    assert int(first["loss_mask"].sum()) + int(second["loss_mask"].sum()) == 8
    # After the first resumed optimizer step the cursor has moved and the static
    # estimate describes the following batch again.
    final_scales = resumed._estimate_optimizer_batch_loss_scales()
    assert final_scales is not None
    assert final_scales["rl"] == 4.0


def test_packed_micro_batch_count_covers_whole_epoch_without_spillover() -> None:
    config = RLConfig(
        data={
            "batch_size": 2,
            "micro_batch_size": 2,
            "seq_len": 8,
            "pack_sequences": True,
        }
    )
    trainer = RLTrainer(config)
    trainer.world = _cpu_world()
    trainer.dataset = PackedRLDataset(
        records=[_rl_example(6) for _ in range(5)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=config.data,
    )

    assert trainer._packed_dataloader_batch_count() == 3
    assert trainer.dataset.micro_batch_count() == 6


def test_ipo_mask_metrics_use_ipo_namespace() -> None:
    trainer = RLTrainer(RLConfig(loss={"type": "ipo"}))

    aliases = trainer._standard_metric_aliases(
        {
            "is_masked": 0.25,
            "is_masked_low": 0.10,
            "is_masked_high": 0.15,
        }
    )

    assert aliases["ipo/is_masked"] == pytest.approx(0.25)
    assert aliases["ipo/is_masked_low"] == pytest.approx(0.10)
    assert aliases["ipo/is_masked_high"] == pytest.approx(0.15)
    assert "dppo/is_masked" not in aliases

    dppo_aliases = RLTrainer(RLConfig())._standard_metric_aliases({"is_masked": 0.20})
    assert dppo_aliases["dppo/is_masked"] == pytest.approx(0.20)
    assert "ipo/is_masked" not in dppo_aliases

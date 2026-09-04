from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from wavelet.configs.sft import SFTConfig
from wavelet.data.sft import Example
from wavelet.trainer import trainer as trainer_module
from wavelet.trainer.distributed import World
from wavelet.trainer.trainer import BaseTrainer, SFTTrainer


def _cpu_world() -> World:
    return World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )


def _config(*, eval_on_start: bool = True) -> SFTConfig:
    return SFTConfig(
        loss_impl="torch",
        max_steps=4,
        val={
            "interval": 2,
            "eval_on_start": eval_on_start,
            "data": {
                "source": "fake",
                "batch_size": 1,
                "micro_batch_size": 1,
                "seq_len": 8,
                "fake_vocab_size": 8,
            },
        },
    )


def _batch(marker: int, labels: list[int]) -> dict[str, torch.Tensor]:
    length = len(labels)
    return {
        "input_ids": torch.tensor([[marker] * length]),
        "attention_mask": torch.ones((1, length), dtype=torch.long),
        "position_ids": torch.arange(length).unsqueeze(0),
        "labels": torch.tensor([labels]),
    }


class _ValidationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grad_enabled: list[bool] = []
        self.training_modes: list[bool] = []

    def forward(self, *, input_ids: torch.Tensor, **_kwargs: object) -> object:
        self.grad_enabled.append(torch.is_grad_enabled())
        self.training_modes.append(self.training)
        logits = torch.zeros((*input_ids.shape, 2), dtype=torch.float32)
        if int(input_ids[0, 0].item()) == 1:
            logits[..., 0] = 2.0
        return SimpleNamespace(logits=logits)


def test_setup_builds_independent_finite_validation_dataloader(monkeypatch) -> None:
    trainer = SFTTrainer(_config())
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)  # type: ignore[assignment]
    trainer.world = _cpu_world()
    training_dataset = object()
    validation_dataset = object()
    records = [Example(prompt=[], completion=[])]
    monkeypatch.setattr(
        BaseTrainer,
        "_setup_data",
        lambda self: setattr(self, "dataset", training_dataset),
    )
    load_records = Mock(return_value=records)
    setup_dataset = Mock(return_value=validation_dataset)
    setup_dataloader = Mock(side_effect=["train-loader", "validation-loader"])
    monkeypatch.setattr(trainer_module, "load_records", load_records)
    monkeypatch.setattr(trainer_module, "setup_dataset", setup_dataset)
    monkeypatch.setattr(trainer_module, "setup_dataloader", setup_dataloader)

    trainer._setup_data()

    assert trainer.dataloader == "train-loader"
    assert trainer.val_dataloader == "validation-loader"
    load_records.assert_called_once_with(trainer.config.val.data)
    assert setup_dataset.call_args.kwargs["records"] is records
    assert setup_dataset.call_args.kwargs["max_epochs_per_iteration"] == 1
    assert setup_dataloader.call_args_list[0].args[0] is training_dataset
    assert setup_dataloader.call_args_list[1].args[0] is validation_dataset


def test_validation_logs_token_weighted_loss_without_gradients(monkeypatch) -> None:
    trainer = SFTTrainer(_config())
    model = _ValidationModel()
    trainer.model = model  # type: ignore[assignment]
    trainer.world = _cpu_world()
    trainer.monitor = Mock()
    trainer.val_dataloader = [
        _batch(0, [0, -100, -100]),
        _batch(1, [0, 0, 0]),
    ]  # type: ignore[assignment]
    next_loader = Mock(return_value=[])
    monkeypatch.setattr(trainer, "_build_validation_dataloader", next_loader)

    trainer._run_validation(step=0)

    neutral_loss = torch.log(torch.tensor(2.0))
    confident_loss = torch.logsumexp(torch.tensor([2.0, 0.0]), dim=0) - 2.0
    expected = float((neutral_loss + 3 * confident_loss) / 4)
    metrics, step = trainer.monitor.log.call_args.args
    assert metrics["val/loss"] == pytest.approx(expected)
    assert step == 0
    assert model.grad_enabled == [False, False]
    assert model.training_modes == [False, False]
    assert model.training is True
    next_loader.assert_called_once_with()


def test_validation_schedule_honors_start_and_interval(monkeypatch) -> None:
    trainer = SFTTrainer(_config(eval_on_start=True))
    run_validation = Mock()
    monkeypatch.setattr(trainer, "_run_validation", run_validation)

    trainer._before_train_loop()
    trainer._before_train_loop()
    trainer.step = 1
    trainer._after_optimizer_step()
    trainer.step = 2
    trainer._after_optimizer_step()
    trainer._after_optimizer_step()
    trainer.step = 4
    trainer._after_optimizer_step()

    assert [call.args[0] for call in run_validation.call_args_list] == [0, 2, 4]


def test_validation_start_is_optional(monkeypatch) -> None:
    trainer = SFTTrainer(_config(eval_on_start=False))
    run_validation = Mock()
    monkeypatch.setattr(trainer, "_run_validation", run_validation)

    trainer._before_train_loop()

    run_validation.assert_not_called()

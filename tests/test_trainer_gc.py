from __future__ import annotations

import gc
from unittest.mock import Mock

import pytest

from wavelet.configs.config import TrainerConfig
from wavelet.trainer.garbage_collection import DeterministicGarbageCollector
from wavelet.trainer.trainer import BaseTrainer


def test_deterministic_gc_collects_on_interval_and_restores_automatic_gc(
    monkeypatch,
) -> None:
    automatic_gc_was_enabled = gc.isenabled()
    gc.enable()
    collect = Mock(return_value=0)
    monkeypatch.setattr(gc, "collect", collect)
    controller = DeterministicGarbageCollector(interval=3)
    try:
        assert not gc.isenabled()
        collect.assert_called_once_with(1)

        controller.run(1)
        controller.run(2)
        collect.assert_called_once_with(1)

        controller.run(3)
        assert collect.call_count == 2
        collect.assert_called_with(1)
    finally:
        controller.close()

    assert gc.isenabled()
    if not automatic_gc_was_enabled:
        gc.disable()


def test_deterministic_gc_preserves_preexisting_disabled_state(monkeypatch) -> None:
    automatic_gc_was_enabled = gc.isenabled()
    gc.disable()
    monkeypatch.setattr(gc, "collect", Mock(return_value=0))
    controller = DeterministicGarbageCollector(interval=1)
    try:
        controller.close()
        assert not gc.isenabled()
    finally:
        if automatic_gc_was_enabled:
            gc.enable()


def test_trainer_gc_can_be_disabled() -> None:
    trainer = BaseTrainer(TrainerConfig(gc=None))

    trainer._ensure_garbage_collector()

    assert trainer._garbage_collector is None


def test_gc_interval_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TrainerConfig(gc={"interval": 0})

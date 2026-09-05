from __future__ import annotations

import math
import sys
import types

import pytest
import torch

from wavelet.configs.rl_config import RLLossConfig
from wavelet.trainer.rl_loss import (
    LossInputs,
    component_normalization_unit_counts,
    compute_loss,
    normalization_unit_count,
)
from wavelet.trainer.types import LossOutput


def test_loss_scale_matches_token_normalization() -> None:
    loss_config = RLLossConfig(kl_tau=0.0)
    trainer_logprobs = torch.zeros(1, 4)
    inference_logprobs = torch.zeros_like(trainer_logprobs)
    advantages = torch.tensor([[2.0, 2.0, 2.0, -1.0]])
    loss_mask = torch.tensor([[True, True, True, True]])
    position_ids = torch.tensor([[0, 1, 2, 0]])

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        advantages,
        loss_mask,
        loss_config,
        loss_scale=4,
        position_ids=position_ids,
    )

    assert isinstance(output, LossOutput)
    assert output.loss.item() == pytest.approx(-1.25)


def test_sequence_normalization_remains_available() -> None:
    loss_config = RLLossConfig(kl_tau=0.0, normalization="sequence")
    trainer_logprobs = torch.zeros(1, 4)
    inference_logprobs = torch.zeros_like(trainer_logprobs)
    advantages = torch.tensor([[2.0, 2.0, 2.0, -1.0]])
    loss_mask = torch.tensor([[True, True, True, True]])
    position_ids = torch.tensor([[0, 1, 2, 0]])

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        advantages,
        loss_mask,
        loss_config,
        loss_scale=2,
        position_ids=position_ids,
    )

    assert output.loss.item() == pytest.approx(-0.5)


def test_normalization_unit_count_matches_packed_sequence_boundaries() -> None:
    loss_mask = torch.tensor([[True, True, False, True, False]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    assert (
        normalization_unit_count(
            loss_mask,
            normalization="token",
            position_ids=position_ids,
        )
        == 3
    )
    assert (
        normalization_unit_count(
            loss_mask,
            normalization="sequence",
            position_ids=position_ids,
        )
        == 2
    )


def test_policy_gradient_moves_logprobs_in_advantage_direction() -> None:
    trainer_logprobs = torch.zeros((1, 2), requires_grad=True)
    optimizer = torch.optim.SGD([trainer_logprobs], lr=0.1)

    output = compute_loss(
        trainer_logprobs,
        torch.zeros_like(trainer_logprobs),
        None,
        torch.tensor([[1.0, -1.0]]),
        torch.ones((1, 2), dtype=torch.bool),
        RLLossConfig(kl_tau=0.0),
        loss_scale=2,
    )
    output.loss.backward()
    optimizer.step()

    assert trainer_logprobs[0, 0].item() > 0.0
    assert trainer_logprobs[0, 1].item() < 0.0


def test_ipo_masks_probability_moves_symmetrically() -> None:
    trainer_logprobs = torch.log(
        torch.tensor([[0.32, 0.18, 0.59, 0.41]])
    ).requires_grad_()
    inference_logprobs = torch.log(torch.tensor([[0.20, 0.30, 0.50, 0.50]]))

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        torch.tensor([[1.0, 1.0, 1.0, -1.0]]),
        torch.ones((1, 4), dtype=torch.bool),
        RLLossConfig(type="ipo", ipo_epsilon=0.1, kl_tau=0.0),
        loss_scale=4,
    )
    output.loss.backward()

    assert output.metrics["is_masked"].item() == pytest.approx(0.5)
    assert output.metrics["is_masked_high"].item() == pytest.approx(0.25)
    assert output.metrics["is_masked_low"].item() == pytest.approx(0.25)
    torch.testing.assert_close(trainer_logprobs.grad[0, :2], torch.zeros(2))
    assert trainer_logprobs.grad[0, 2].item() < 0.0
    assert trainer_logprobs.grad[0, 3].item() > 0.0


def test_packed_loss_metrics_are_averaged_per_sequence() -> None:
    loss_config = RLLossConfig(kl_tau=0.0)
    trainer_logprobs = torch.tensor([[0.0, 0.0, 0.0, math.log(0.5)]])
    inference_logprobs = torch.tensor([[0.0, 0.0, 0.0, math.log(0.25)]])
    loss_mask = torch.tensor([[True, True, True, True]])
    advantages = torch.zeros_like(trainer_logprobs)
    position_ids = torch.tensor([[0, 1, 2, 0]])

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        None,
        advantages,
        loss_mask,
        loss_config,
        loss_scale=4,
        position_ids=position_ids,
    )

    expected_second_sequence_kl = 2.0 - math.log(2.0) - 1.0
    assert output.metrics["mismatch_kl"].item() == pytest.approx(
        expected_second_sequence_kl / 2
    )


def test_rl_only_dppo_matches_golden_values_with_default_component_weights() -> None:
    trainer_logprobs = torch.tensor(
        [[math.log(0.3), math.log(0.7), math.log(0.2)]],
        requires_grad=True,
    )
    inference_logprobs = torch.tensor([[math.log(0.25), math.log(0.6), math.log(0.25)]])
    teacher_logprobs = torch.tensor([[math.log(0.4), math.log(0.65), math.log(0.3)]])
    config = RLLossConfig(
        dppo_mask_high=0.2,
        dppo_mask_low=0.2,
        kl_tau=0.01,
        adv_tau=0.8,
        teacher_tau=0.15,
    )

    legacy = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        torch.tensor([[1.25, -0.5, 0.75]]),
        torch.ones((1, 3), dtype=torch.bool),
        config,
        loss_scale=3,
    )
    componentized = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        torch.tensor([[1.25, -0.5, 0.75]]),
        torch.ones((1, 3), dtype=torch.bool),
        config,
        loss_scale=3,
        rl_weights=torch.ones((1, 3)),
        ce_weights=torch.zeros((1, 3)),
        ref_kl_weights=torch.zeros((1, 3)),
        component_loss_scales={"rl": 3.0, "ce": 1.0, "ref_kl": 1.0},
    )

    assert legacy.loss.item() == pytest.approx(-0.43324506282806396)
    assert torch.equal(componentized.loss, legacy.loss)
    assert componentized.metrics.keys() == legacy.metrics.keys()
    for key in legacy.metrics:
        assert torch.equal(componentized.metrics[key], legacy.metrics[key])


def test_ce_and_ref_kl_components_use_independent_token_scales() -> None:
    trainer_logprobs = torch.full((1, 4), -1.0)
    inference_logprobs = trainer_logprobs.clone()
    teacher_logprobs = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    loss_mask = torch.ones((1, 4), dtype=torch.bool)

    output = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        torch.zeros_like(trainer_logprobs),
        loss_mask,
        RLLossConfig(),
        rl_weights=torch.zeros_like(trainer_logprobs),
        ce_weights=torch.tensor([[0.0, 2.0, 1.0, 0.0]]),
        ref_kl_weights=torch.tensor([[1.0, 2.0, 0.0, 0.0]]),
        component_loss_scales={"rl": 1.0, "ce": 2.0, "ref_kl": 2.0},
    )

    assert output.loss.item() == pytest.approx(1.5 - 2.5)
    assert output.metrics["ce/nll"].item() == pytest.approx(1.0)
    assert output.metrics["ref_kl"].item() == pytest.approx(1.5)


def test_ref_kl_one_sided_trust_region_drops_large_probability_decrease() -> None:
    output = compute_loss(
        torch.tensor([[math.log(0.1), math.log(0.8)]]),
        torch.tensor([[math.log(0.5), math.log(0.5)]]),
        torch.tensor([[math.log(0.9), math.log(0.9)]]),
        torch.zeros((1, 2)),
        torch.ones((1, 2), dtype=torch.bool),
        RLLossConfig(),
        rl_weights=torch.zeros((1, 2)),
        ref_kl_weights=torch.ones((1, 2)),
        component_loss_scales={"rl": 1.0, "ce": 1.0, "ref_kl": 2.0},
    )

    assert output.metrics["ref_kl/is_masked"].item() == pytest.approx(0.5)


def test_ref_kl_requires_teacher_logprobs() -> None:
    with pytest.raises(ValueError, match="requires teacher_logprobs"):
        compute_loss(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            None,
            torch.zeros((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
            RLLossConfig(),
            rl_weights=torch.zeros((1, 1)),
            ref_kl_weights=torch.ones((1, 1)),
        )


def test_component_counts_use_nonzero_weights_and_rl_sequence_mode() -> None:
    counts = component_normalization_unit_counts(
        torch.tensor([[True, True, True, True]]),
        rl_weights=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        ce_weights=torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
        ref_kl_weights=torch.tensor([[0.0, 0.0, -1.0, 1.0]]),
        rl_normalization="sequence",
        position_ids=torch.tensor([[0, 1, 0, 1]]),
    )

    assert counts == {"rl": 2, "ce": 1, "ref_kl": 2}


def test_custom_rl_loss_is_imported_and_receives_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("wavelet_test_custom_loss")

    def custom_loss(inputs: LossInputs, *, multiplier: float) -> LossOutput:
        loss = -(inputs.trainer_logprobs[inputs.loss_mask] * multiplier).sum()
        return LossOutput(loss=loss, metrics={"custom": loss.detach()})

    module.custom_loss = custom_loss  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    output = compute_loss(
        torch.tensor([[-1.0, -2.0]]),
        torch.zeros((1, 2)),
        None,
        torch.zeros((1, 2)),
        torch.ones((1, 2), dtype=torch.bool),
        RLLossConfig(
            type="custom",
            import_path="wavelet_test_custom_loss.custom_loss",
            kwargs={"multiplier": 2.0},
        ),
        loss_scale=2,
    )

    assert output.loss.item() == pytest.approx(3.0)
    assert output.metrics["custom"].item() == pytest.approx(6.0)

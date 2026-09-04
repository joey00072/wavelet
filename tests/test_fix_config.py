from __future__ import annotations

import inspect
import signal
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wavelet import debug as debug_module
from wavelet.configs.config import DEFAULT_LORA_TARGET_MODULES
from wavelet.configs.rl_config import (
    GRPOAlgorithmConfig,
    RLConfig,
    TokensLengthPenaltyConfig,
)
from wavelet.configs.sft import LoRAConfig, SFTConfig
from wavelet.kernels import lora as lora_kernels
from wavelet.kernels import patch as kernel_patch
from wavelet.orchestrator import launcher as launcher_module
from wavelet.orchestrator.launcher import RayRoleHandle, RoleSpec
from wavelet.trainer import model as model_module
from wavelet.trainer.debug import DEBUG_LORA_TARGET_MODULES

# ── RL data.num_workers ───────────────────────────────────────────────────────


def test_rl_data_rejects_multiple_dataloader_workers() -> None:
    RLConfig(data={"num_workers": 1})

    with pytest.raises(ValueError, match="num_workers"):
        RLConfig(data={"num_workers": 2})


# ── legacy advantage keys ─────────────────────────────────────────────────────


def test_legacy_advantage_keys_cannot_combine_with_explicit_algo() -> None:
    with pytest.raises(ValueError, match="orchestrator.advantage_mode"):
        RLConfig(
            algo={"type": "grpo"},
            orchestrator={"advantage_mode": "group_reward"},
        )


def test_legacy_grpo_only_keys_require_group_reward_mode() -> None:
    with pytest.raises(ValueError, match="normalize_group_advantages"):
        RLConfig(
            orchestrator={
                "advantage_mode": "reward",
                "normalize_group_advantages": True,
            }
        )


def test_legacy_group_reward_keys_reach_the_algorithm() -> None:
    config = RLConfig(
        orchestrator={
            "advantage_mode": "group_reward",
            "normalize_group_advantages": True,
            "advantage_epsilon": 1e-4,
            "length_penalty": "tokens",
        }
    )

    assert isinstance(config.algo, GRPOAlgorithmConfig)
    assert config.algo.normalize_advantages is True
    assert config.algo.epsilon == pytest.approx(1e-4)
    assert isinstance(config.algo.length_penalty, TokensLengthPenaltyConfig)
    dumped = config.orchestrator.model_dump()
    assert "advantage_mode" not in dumped
    assert "length_penalty" not in dumped


# ── LoRA target modules / alpha ───────────────────────────────────────────────


def test_lora_target_modules_default_has_a_single_source() -> None:
    assert LoRAConfig().target_modules == DEFAULT_LORA_TARGET_MODULES
    assert model_module.DEFAULT_LORA_TARGET_MODULES is DEFAULT_LORA_TARGET_MODULES


def test_default_lora_targets_remap_for_gpt2_debug_models() -> None:
    gpt2 = SimpleNamespace(config=SimpleNamespace(model_type="gpt2"))
    llama = SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    assert (
        model_module._resolve_lora_target_modules(gpt2, LoRAConfig())
        == DEBUG_LORA_TARGET_MODULES
    )
    assert (
        model_module._resolve_lora_target_modules(llama, LoRAConfig())
        == DEFAULT_LORA_TARGET_MODULES
    )
    explicit = LoRAConfig(target_modules=["c_attn"])
    assert model_module._resolve_lora_target_modules(gpt2, explicit) == ["c_attn"]


def test_apply_lora_passes_float_alpha_to_peft(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _RecordingPeftConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(model_module, "PeftLoraConfig", _RecordingPeftConfig)
    monkeypatch.setattr(model_module, "get_peft_model", lambda model, config: model)
    monkeypatch.setattr(model_module, "enforce_single_lora_adapter", lambda model: None)
    monkeypatch.setattr(
        model_module, "_normalize_hf_tp_linear_feature_metadata", lambda model: None
    )
    model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    model_module.apply_lora(model, LoRAConfig(rank=4, alpha=12.5))

    assert captured["lora_alpha"] == 12.5
    assert captured["r"] == 4


# ── checkpoint / fused kernel validators ──────────────────────────────────────


def test_checkpoint_settings_with_disabled_mode_are_rejected() -> None:
    with pytest.raises(ValueError, match="ckpt.interval"):
        SFTConfig(ckpt={"interval": 10})

    SFTConfig(ckpt={"mode": "async", "interval": 10})
    SFTConfig(ckpt={})


def test_fused_lora_kernels_reject_lora_dropout() -> None:
    with pytest.raises(ValueError, match="lora.dropout"):
        RLConfig(model={"fused_lora_mlp": True}, lora={"dropout": 0.1})

    RLConfig(model={"fused_lora_mlp": True}, lora={"dropout": 0.0})
    RLConfig(model={"fused_lora_mlp": True}, lora=None)


def test_rollout_reward_mode_survives_role_config_round_trip() -> None:
    config = RLConfig(
        inference={"mode": "vllm_http"},
        reward={"mode": "reference_match"},
        orchestrator={
            "custom_rollout_function": "wavelet.orchestrator.verifiers:generate_rollouts"
        },
    )

    # Process-mode roles re-validate a full dump of the parent's config; the
    # result must be accepted exactly like the original.
    RLConfig.model_validate(config.model_dump(mode="json", exclude_none=False))
    assert debug_module._rollout_reward_mode_check(config).status == "ok"


def test_preflight_flags_passthrough_reward_for_generated_rollouts() -> None:
    config = RLConfig(inference={"mode": "vllm_http"}, reward={"mode": "passthrough"})

    check = debug_module._rollout_reward_mode_check(config)

    assert check.status == "error"
    assert "passthrough" in check.message


# ── fused kernel patching ─────────────────────────────────────────────────────


class _PlainSwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _PlainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _PlainSwiGLU()


def test_fused_mlp_patch_skips_modules_without_lora_adapters() -> None:
    model = _PlainModel()
    original_forward = _PlainSwiGLU.forward

    assert kernel_patch.patch_fused_mlp(model) is False

    assert _PlainSwiGLU.forward is original_forward
    assert "forward" not in model.mlp.__dict__
    x = torch.randn(2, 4)
    assert torch.equal(model.mlp(x), original_forward(model.mlp, x))


def test_fused_qkv_patch_leaves_class_forward_without_eligible_layers() -> None:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    original_forward = Qwen3Attention.forward

    assert kernel_patch.patch_fused_qkv(_PlainModel()) is False

    assert Qwen3Attention.forward is original_forward


def test_upstream_qwen3_attention_takes_cache_position_by_keyword() -> None:
    # The fused wrapper forwards cache_position through **kwargs because the
    # installed transformers signature has no positional slot for it.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    parameters = inspect.signature(Qwen3Attention.forward).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert "cache_position" not in positional
    assert any(p.kind is p.VAR_KEYWORD for p in parameters.values())


def test_lora_kernel_backward_helpers_skip_inactive_adapters() -> None:
    x = torch.randn(3, 4)
    d_y = torch.randn(3, 5)

    assert lora_kernels._lora_adapter_grads(x, d_y, None, None, None) == (None, None)
    assert lora_kernels._transpose_adapters((None, None), torch.float32) == (
        None,
        None,
    )

    d_x = torch.zeros(3, 4)
    before = d_x.clone()
    lora_kernels._add_lora_input_grad(d_x, d_y, None, None, None)
    assert torch.equal(d_x, before)


def test_lora_kernel_backward_helpers_match_reference_math() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    d_y = torch.randn(3, 5)
    a = torch.randn(2, 4)  # PEFT layout: (rank, in)
    b = torch.randn(5, 2)  # PEFT layout: (out, rank)
    scale = 0.5
    a_t, b_t = lora_kernels._transpose_adapters((a, b), torch.float32)

    d_a, d_b = lora_kernels._lora_adapter_grads(x, d_y, a_t, b_t, scale)
    d_x = torch.zeros(3, 4)
    lora_kernels._add_lora_input_grad(d_x, d_y, a_t, b_t, scale)

    # y = scale * x @ A^T @ B^T
    torch.testing.assert_close(d_a, scale * (d_y @ b).t() @ x)
    torch.testing.assert_close(d_b, scale * d_y.t() @ (x @ a.t()))
    torch.testing.assert_close(d_x, scale * d_y @ b @ a)


# ── launcher teardown ─────────────────────────────────────────────────────────


class _InterruptedProcess:
    pid = 1234

    def __init__(self) -> None:
        self.wait_calls = 0
        self.polled = False

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise KeyboardInterrupt
        return -15


def test_role_wait_tears_down_process_group_on_interrupt(monkeypatch) -> None:
    process = _InterruptedProcess()
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(launcher_module.os, "getpgid", lambda pid: 4321)
    monkeypatch.setattr(
        launcher_module.os,
        "killpg",
        lambda pgid, signal_number: killpg_calls.append((pgid, signal_number)),
    )

    with pytest.raises(KeyboardInterrupt):
        launcher_module._wait_for_role_process(process, timeout_seconds=1.0)  # type: ignore[arg-type]

    assert killpg_calls == [(4321, signal.SIGTERM)]
    assert process.wait_calls == 2


@pytest.mark.parametrize("finishes", [True, False])
def test_ray_role_handle_prefers_cooperative_cancel(finishes: bool) -> None:
    calls: list[tuple[str, object]] = []

    class _FakeRay:
        @staticmethod
        def cancel(ref: object, *, force: bool) -> None:
            calls.append(("cancel", force))

        @staticmethod
        def wait(refs: list[object], *, timeout: float) -> tuple[list, list]:
            calls.append(("wait", timeout))
            return (refs, []) if finishes else ([], refs)

    handle = RayRoleHandle(
        RoleSpec("trainer", "rl-trainer", "config.yaml", "trainer"),  # type: ignore[arg-type]
        ref=object(),
        ray_module=_FakeRay,
        log_path="trainer.log",  # type: ignore[arg-type]
    )
    handle.terminate(timeout_seconds=2.0)

    expected: list[tuple[str, object]] = [("cancel", False), ("wait", 2.0)]
    if not finishes:
        expected.append(("cancel", True))
    assert calls == expected


# ── preflight launcher checks ─────────────────────────────────────────────────


def test_preflight_flags_trainer_process_count_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(debug_module, "_available_gpu_indices", lambda: None)
    config = RLConfig(
        launcher={"trainer_cuda_visible_devices": "0,1", "trainer_num_processes": 1}
    )

    checks = {check.name: check for check in debug_module._device_group_checks(config)}

    assert checks["trainer_num_processes"].status == "error"
    assert "2 pinned trainer device" in checks["trainer_num_processes"].message


def test_preflight_accepts_matching_trainer_process_count(monkeypatch) -> None:
    monkeypatch.setattr(debug_module, "_available_gpu_indices", lambda: None)
    config = RLConfig(
        launcher={"trainer_cuda_visible_devices": "0,1", "trainer_num_processes": 2}
    )

    names = {check.name for check in debug_module._device_group_checks(config)}

    assert "trainer_num_processes" not in names


@pytest.mark.parametrize("mode", ["process", "colocate"])
def test_preflight_torchrun_check_covers_every_multi_role_mode(
    monkeypatch, mode: str
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(debug_module, "_available_gpu_indices", lambda: None)
    config = RLConfig(launcher={"mode": mode})

    checks = {check.name: check for check in debug_module._launcher_checks(config)}

    assert checks["torchrun_launcher"].status == "error"


def test_launcher_rejects_passthrough_reward_for_generated_rollouts() -> None:
    from wavelet.orchestrator.placement import (
        rollout_reward_mode_error,
        validate_rollout_reward_mode,
    )

    invalid = RLConfig(inference={"mode": "vllm_http"}, reward={"mode": "passthrough"})
    with pytest.raises(ValueError, match="passthrough"):
        validate_rollout_reward_mode(invalid)

    scored = RLConfig(inference={"mode": "vllm_http"}, reward={"mode": "math_format"})
    custom = RLConfig(
        reward={"mode": "passthrough"},
        orchestrator={
            "custom_rollout_function": "wavelet.orchestrator.verifiers:generate_rollouts"
        },
    )
    assert rollout_reward_mode_error(scored) is None
    assert rollout_reward_mode_error(custom) is None

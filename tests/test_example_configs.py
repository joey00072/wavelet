from __future__ import annotations

from pathlib import Path

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import SFTConfig
from wavelet.utils.serialization import load_yaml

RL_CONFIGS = [
    path
    for path in sorted(Path("examples").rglob("*.yaml"))
    if path.name.startswith("rl")
]
SFT_CONFIGS = [
    path
    for path in sorted(Path("examples").rglob("*.yaml"))
    if path.name.startswith("sft")
]


@pytest.mark.parametrize("path", RL_CONFIGS, ids=str)
def test_rl_example_config_validates(path: Path) -> None:
    RLConfig.model_validate(load_yaml(path))


@pytest.mark.parametrize("path", SFT_CONFIGS, ids=str)
def test_sft_example_config_validates(path: Path) -> None:
    SFTConfig.model_validate(load_yaml(path))


def test_wordle_rl_uses_reference_environment_id() -> None:
    config = RLConfig.model_validate(load_yaml(Path("examples/wordle/rl.yaml")))

    assert config.orchestrator.verifier_env_id == "primeintellect/wordle"
    assert config.eval is not None
    assert config.eval.env[0].id == "primeintellect/wordle"


def test_wordle_rl_matches_reference_training_shape() -> None:
    config = RLConfig.model_validate(load_yaml(Path("examples/wordle/rl.yaml")))

    assert config.model.name == "PrimeIntellect/Qwen3-1.7B-Wordle-SFT"
    assert config.model.torch_dtype == "float32"
    assert config.model.attn_implementation == "auto"
    assert config.data.batch_size == 1024
    assert config.data.seq_len == 8192
    assert config.orchestrator.examples_per_step == 64
    assert config.orchestrator.rollouts_per_example == 16
    assert config.inference.sampling.max_completion_tokens == 1024
    assert config.inference.vllm.dtype == "bfloat16"
    assert config.inference.vllm.data_parallel_size == 6
    assert config.launcher.trainer_num_processes == 2
    assert config.optim.lr == pytest.approx(1e-6)


def test_reverse_text_rl_matches_reference_training_shape() -> None:
    config = RLConfig.model_validate(load_yaml(Path("examples/reverse_text/rl.yaml")))

    assert config.model.name == "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
    assert config.model.torch_dtype == "float32"
    assert config.data.seq_len == 2048
    assert config.algo.type == "grpo"
    assert config.orchestrator.examples_per_step == 128
    assert config.orchestrator.rollouts_per_example == 16
    assert config.inference.sampling.max_completion_tokens == 128
    assert config.inference.vllm.dtype == "bfloat16"
    assert config.inference.vllm.gpu_memory_utilization == pytest.approx(0.9)
    assert config.optim.type == "adamw"
    assert config.optim.lr == pytest.approx(3e-6)


def test_hendrycks_sanity_rl_matches_reference_training_shape() -> None:
    config = RLConfig.model_validate(
        load_yaml(Path("examples/hendrycks_sanity/rl.yaml"))
    )

    assert config.model.name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert config.model.torch_dtype == "float32"
    assert config.model.attn_implementation == "auto"
    assert config.data.batch_size == 512
    assert config.data.seq_len == 16384
    assert config.orchestrator.verifier_env_id == "math-env"
    assert config.orchestrator.verifier_env_args == {
        "dataset_name": "mikasenghaas/Sanity-Test-R1D-1.5B",
        "dataset_subset": "default",
    }
    assert config.orchestrator.examples_per_step == 64
    assert config.orchestrator.rollouts_per_example == 8
    assert config.inference.sampling.max_completion_tokens is None
    assert config.inference.vllm.dtype == "bfloat16"
    assert config.inference.vllm.max_model_len == 8192
    assert config.inference.vllm.data_parallel_size == 4
    assert config.launcher.trainer_num_processes == 4
    assert config.optim.lr == pytest.approx(1e-6)


def test_wiki_search_rl_matches_reference_training_shape() -> None:
    config = RLConfig.model_validate(load_yaml(Path("examples/wiki_search/rl.yaml")))

    assert config.model.name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model.torch_dtype == "float32"
    assert config.model.attn_implementation == "auto"
    assert config.data.batch_size == 512
    assert config.data.seq_len == 4096
    assert config.orchestrator.verifier_env_id == "primeintellect/wiki-search"
    assert config.orchestrator.examples_per_step == 32
    assert config.orchestrator.rollouts_per_example == 16
    assert config.orchestrator.oversampling_factor == 2.0
    assert config.inference.sampling.max_completion_tokens == 512
    assert config.inference.vllm.dtype == "bfloat16"
    assert config.inference.vllm.data_parallel_size == 6
    assert config.launcher.trainer_num_processes == 2
    assert config.lora is not None
    assert config.lora.rank == 8
    assert config.optim.weight_decay == 0.0


def test_qwen30b_math_uses_full_reference_completion_budget() -> None:
    config = RLConfig.model_validate(load_yaml(Path("examples/qwen30b_math/rl.yaml")))

    assert config.inference.sampling.max_completion_tokens == 32768
    assert config.eval is not None
    assert config.eval.sampling.max_completion_tokens == 32768
    assert config.inference.vllm.max_model_len == 32768


def test_qwen4b_math_int4_smoke_uses_two_gpu_qlora_shape() -> None:
    config = RLConfig.model_validate(
        load_yaml(Path("examples/qwen4b_math/rl_int4_2gpu_smoke.yaml"))
    )

    assert config.model.name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model.load_in_4bit is True
    assert config.fsdp.enabled is False
    assert config.lora is not None
    assert config.lora.rank == 16
    assert config.optim.type == "paged_adamw_8bit"
    assert config.optim.implementation == "for-loop"
    assert config.inference.vllm.quantization == "bitsandbytes"
    assert config.inference.vllm.load_format == "bitsandbytes"
    assert config.launcher.inference_cuda_visible_devices == "0"
    assert config.launcher.trainer_cuda_visible_devices == "1"
    assert config.launcher.trainer_num_processes == 1
    assert config.max_steps == 1


def test_moe_reverse_text_sft_uses_int4_moe_qlora_shape() -> None:
    config = SFTConfig.model_validate(
        load_yaml(Path("examples/moe_reverse_text/sft.yaml"))
    )

    assert config.model.name == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert config.model.trust_remote_code is True
    assert config.model.load_in_4bit is True
    assert config.model.kbit_cast_non_quantized_to_float32 is False
    assert config.fsdp.enabled is False
    assert config.lora is not None
    assert config.lora.rank == 16
    assert set(config.lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert config.optim.type == "paged_adamw_8bit"
    assert config.deployment.num_gpus == 1
    assert config.max_steps == 2


def test_polaris_recovery_sft_continues_step208_adapter_for_three_epochs() -> None:
    config = SFTConfig.model_validate(
        load_yaml(Path("examples/qwen2_5_7b_polaris/sft_recoveries.yaml"))
    )

    assert config.model.adapter_path == Path(
        "outputs/qwen2_5_7b_polaris_grpo_100k_v2/final_adapter_step-000208/adapter"
    )
    assert config.model.load_in_4bit is False
    assert config.data.batch_size == 8
    assert config.data.seq_len == 2048
    assert config.optim.lr == pytest.approx(1e-5)
    assert config.epochs == 3
    assert config.max_steps == 117


def test_polaris_recovery_sft_plus3_continues_sft_adapter_at_requested_lr() -> None:
    config = SFTConfig.model_validate(
        load_yaml(Path("examples/qwen2_5_7b_polaris/sft_recoveries_plus3_lr1e4.yaml"))
    )

    assert config.model.adapter_path == Path(
        "outputs/qwen2_5_7b_polaris_recovery_sft_step208/adapter"
    )
    assert config.model.load_in_4bit is False
    assert config.data.batch_size == 8
    assert config.optim.lr == pytest.approx(1e-4)
    assert config.epochs == 3
    assert config.max_steps == 117


def test_polaris_recovery_fresh_sft_initializes_new_lora_at_requested_lr() -> None:
    config = SFTConfig.model_validate(
        load_yaml(Path("examples/qwen2_5_7b_polaris/sft_recoveries_fresh_lr2e4.yaml"))
    )

    assert config.model.adapter_path is None
    assert config.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert config.model.load_in_4bit is False
    assert config.lora is not None
    assert config.lora.rank == 16
    assert config.data.batch_size == 8
    assert config.optim.lr == pytest.approx(2e-4)
    assert config.epochs == 3
    assert config.max_steps == 117


def test_polaris_async_rl_uses_canonical_two_gpu_batch_shape() -> None:
    config = RLConfig.model_validate(
        load_yaml(Path("examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32.yaml"))
    )

    assert config.model.adapter_path == Path(
        "outputs/qwen2_5_7b_polaris_recovery_sft_fresh_lr2e4/adapter"
    )
    assert config.model.load_in_4bit is False
    assert config.model.attn_implementation == "flash_attention_2"
    assert config.model.fused_lm_head_token_chunk_size == "auto"
    assert config.orchestrator.examples_per_step == 32
    assert config.orchestrator.rollouts_per_example == 8
    assert config.orchestrator.rollout_chunk_examples == 8
    assert config.orchestrator.max_async_level == 4
    assert config.orchestrator.max_off_policy_steps == 4
    assert config.orchestrator.max_pending_rollout_chunks == 8
    assert config.orchestrator.zero_advantage_max_retries == 8
    assert config.data.batch_size == 32
    assert config.data.micro_batch_size == 16
    assert config.data.pack_sequences is False
    assert config.data.seq_len == 8192
    assert config.loss.kl_tau == 0.0
    assert config.optim.lr == pytest.approx(5e-5)
    assert config.launcher.mode == "process"
    assert config.launcher.trainer_cuda_visible_devices == "0"
    assert config.launcher.inference_cuda_visible_devices == "1"
    assert config.eval is not None
    assert config.eval.interval == 100
    assert config.policy_transfer.keep_last == 8
    assert config.ckpt is not None
    assert config.ckpt.keep_last == 2
    assert config.max_steps == 100000


def test_moe_reverse_text_rl_starts_from_sft_adapter_on_two_gpus() -> None:
    config = RLConfig.model_validate(
        load_yaml(Path("examples/moe_reverse_text/rl.yaml"))
    )

    assert config.model.name == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert config.model.adapter_path == Path("outputs/moe_reverse_text_sft/adapter")
    assert config.model.trust_remote_code is True
    assert config.model.load_in_4bit is True
    assert config.model.kbit_cast_non_quantized_to_float32 is False
    assert config.fsdp.enabled is False
    assert config.lora is not None
    assert config.lora.rank == 16
    assert config.optim.type == "paged_adamw_8bit"
    assert config.inference.vllm.quantization is None
    assert config.inference.vllm.load_format is None
    assert config.orchestrator.verifier_env_id == "reverse-text"
    assert config.eval is not None
    assert config.eval.final_eval is True
    assert config.launcher.inference_cuda_visible_devices == "0"
    assert config.launcher.trainer_cuda_visible_devices == "1"
    assert config.launcher.trainer_num_processes == 1
    assert config.max_steps == 2


def test_reverse_text_rl_matches_reference_core_hyperparams() -> None:
    config = RLConfig.model_validate(load_yaml(Path("examples/reverse_text/rl.yaml")))

    assert config.model.name == "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
    assert config.data.batch_size == 128
    assert config.data.seq_len == 2048
    assert config.orchestrator.verifier_env_id == "reverse-text"
    assert config.orchestrator.examples_per_step == 128
    assert config.orchestrator.rollouts_per_example == 16
    assert config.inference.sampling.max_completion_tokens == 128
    assert config.optim.lr == pytest.approx(3e-6)


def test_reverse_text_int4_4b_tracks_reward_eval_shape() -> None:
    config = RLConfig.model_validate(
        load_yaml(Path("examples/reverse_text/rl_int4_4b.yaml"))
    )

    assert config.model.name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model.load_in_4bit is True
    assert config.fsdp.enabled is False
    assert config.lora is not None
    assert config.lora.rank == 16
    assert config.optim.type == "paged_adamw_8bit"
    assert config.inference.vllm.quantization == "bitsandbytes"
    assert config.inference.vllm.load_format == "bitsandbytes"
    assert config.eval is not None
    assert config.eval.eval_base_model is True
    assert config.eval.final_eval is True
    assert config.eval.interval == 25
    assert config.eval.num_examples == 16
    assert config.launcher.inference_cuda_visible_devices == "0"
    assert config.launcher.trainer_cuda_visible_devices == "1"
    assert config.max_steps == 100


@pytest.mark.parametrize(
    "path",
    [
        Path("examples/alphabet_sort/rl.yaml"),
        Path("examples/alphabet_sort/rl_colocate.yaml"),
        Path("examples/alphabet_sort/rl_colocate_sleep.yaml"),
        Path("examples/alphabet_sort/rl_fsdp_multi.yaml"),
        Path("examples/alphabet_sort/rl_fsdp_multi_reward.yaml"),
    ],
    ids=str,
)
def test_alphabet_sort_rl_uses_reference_environment_id(path: Path) -> None:
    config = RLConfig.model_validate(load_yaml(path))

    assert config.orchestrator.verifier_env_id == "primeintellect/alphabet-sort"
    if config.eval is not None:
        assert all(
            env.id == "primeintellect/alphabet-sort"
            for env in config.eval.env
            if env.name == "alphabet-sort"
        )

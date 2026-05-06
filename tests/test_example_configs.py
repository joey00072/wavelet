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


def test_reverse_text_rl_matches_reference_core_hyperparams() -> None:
    config = RLConfig.model_validate(load_yaml(Path("examples/reverse_text/rl.yaml")))

    assert config.model.name == "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
    assert config.data.batch_size == 128
    assert config.data.seq_len == 2048
    assert config.orchestrator.verifier_env_id == "reverse-text"
    assert config.orchestrator.examples_per_step == 8
    assert config.orchestrator.rollouts_per_example == 16
    assert config.inference.sampling.max_completion_tokens == 128
    assert config.optim.lr == pytest.approx(3e-6)


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

# Qwen4B Math

This example adapts the long-context 30B math RL recipe to Wavelet for
`Qwen/Qwen3-4B-Instruct-2507`.

It uses:

- `math-env` verifier rollouts
- AIME 2025 verifier evals every 25 policy steps
- 64 math examples per step with 8 rollouts each
- zero-advantage filtering with 2x oversampling
- LoRA rank 32 / alpha 64
- a four-rank FSDP trainer on GPUs `4,5,6,7`
- four vLLM OpenAI-compatible inference replicas on GPUs `0,1,2,3`

Install verifier extras:

```bash
uv sync --extra verifiers --extra envs
```

Prepare local verifier examples:

```bash
uv run python examples/qwen4b_math/prepare_rl_data.py
```

Use a specific math dataset if needed:

```bash
uv run python examples/qwen4b_math/prepare_rl_data.py \
  --dataset-name PrimeIntellect/Hendrycks-Math \
  --dataset-subset default
```

Validate the configuration without starting training:

```bash
uv run python -m wavelet rl @ examples/qwen4b_math/rl.yaml --dry_run true
```

Run training:

```bash
uv run python -m wavelet rl @ examples/qwen4b_math/rl.yaml
```

## Two-GPU INT4 Smoke

`rl_int4_2gpu_smoke.yaml` is a minimal QLoRA smoke version inspired by Orbit's
2-GPU INT4 Qwen3-4B path. It uses bitsandbytes 4-bit LoRA training on GPU `1`
and one quantized vLLM server on GPU `0`. The smoke keeps the rollout step,
batch size, context, and completion budget small; it is intended to prove the
train/inference/policy-transfer path before scaling the full `rl.yaml` recipe.

Prepare one local math row, then preflight:

```bash
uv run python examples/qwen4b_math/prepare_rl_data.py --examples 1
uv run python -m wavelet debug preflight \
  @ examples/qwen4b_math/rl_int4_2gpu_smoke.yaml --json
```

Run the smoke on two visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run python -m wavelet rl @ examples/qwen4b_math/rl_int4_2gpu_smoke.yaml
```

The original 30B recipe uses 32k context, expert parallelism, custom kernels,
multi-node training, and tensor-parallel inference. This 4B recipe keeps the
same math/verifier structure but uses 8k context and single-node FSDP settings
that fit the current Wavelet runtime.

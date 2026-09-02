# Qwen2.5 7B MATH-500 GRPO

This example trains `Qwen/Qwen2.5-7B-Instruct` with LoRA and GRPO on the
500-problem MATH-500 test split. Each optimizer step accepts eight problem
groups with eight rollouts per group. The scheduler loops over the local
500-row dataset for the 100,000-step run.

The `math500-tagged` environment requires every response to use
`<think>...</think><answer>...</answer>` during both training and evaluation.
It uses `math-verify` for binary correctness. Zero-advantage filtering removes
all-correct and all-wrong groups and retries until a mixed, trainable batch is
available. This is the repository's available dynamic-sampling filter; it does
not discard a group merely because a majority of its eight rollouts are correct.

The single-GPU run uses `colocate_sleep`: vLLM releases its GPU allocation
during each 8k-token LoRA optimizer step, then wakes for the next rollout batch.
This keeps full-precision LoRA training within memory without QLoRA.

Periodic evaluation runs all 500 problems with eight rollouts every 100 policy
steps, beginning at step 100. These evaluations reuse the training problems, so
they measure training-set improvement rather than held-out generalization.

## Setup and smoke test

```bash
uv sync --extra verifiers
uv pip install --python .venv/bin/python \
  --editable environments/math500_tagged
uv run python examples/qwen2_5_7b_math500/prepare_rl_data.py
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_math500/rl_smoke.yaml --json
uv run python -m wavelet rl @ examples/qwen2_5_7b_math500/rl_smoke.yaml
```

## 100k run

```bash
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_math500/rl_100k.yaml --json
uv run python -m wavelet rl @ examples/qwen2_5_7b_math500/rl_100k.yaml
```

Policy snapshots are retained for only the latest five steps to keep the long
run bounded. Async checkpoints are written every 100 steps with the latest two
retained. Only the latest five consumed training batches and latest two complete
evaluation rollout sets are retained, and the monitor writes at most eight
sampled training completions every ten steps for reward-hacking audits.

## Dashboard

Start the WebUI from `webui/` and connect it to the state server on port 8765.
The MATH-500 chart has a fixed 70–100% y-axis, a 74.2% base-model baseline, an
83.1% Qwen2.5-72B-Instruct reference, and new `avg@8`/`pass@8` points after each
100-step evaluation.

# Qwen2.5 7B Polaris GRPO

This example trains `Qwen/Qwen2.5-7B-Instruct` with BF16 LoRA and GRPO on
`POLARIS-Project/Polaris-Dataset-53K`. It keeps difficulty buckets `1/8`
through `6/8`, removes normalized duplicates, and removes exact normalized
overlaps with the held-out AIME 2024 evaluation set. Proof requests are removed
because the binary final-answer rubric cannot validate proofs and some of those
rows contain corrupted target fragments. The environment also removes narrowly
detected incomplete labels such as empty fraction operands; after all default
filters, the current training split contains 29,105 examples.

Every response must be exactly
`<think>...</think><answer>...</answer>`. The canonical run uses 32 problems per
optimizer step and eight rollouts per problem. Zero-advantage filtering retries
all-correct and all-wrong groups so the trainer receives mixed groups.

AIME 2024 evaluation uses the pinned 30-problem source from Prime's environment,
eight rollouts per problem, and the same strict response parser and math
verifier. Run the zero-step config first to establish the untrained baseline;
it starts no trainer process.

```bash
uv sync --extra flash-attn
uv pip install --python .venv/bin/python \
  --editable environments/polaris_math_tagged
uv run python examples/qwen2_5_7b_polaris/prepare_rl_data.py
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/eval_baseline.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/eval_baseline.yaml
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_smoke.yaml --json
uv run python -m wavelet rl @ examples/qwen2_5_7b_polaris/rl_smoke.yaml
```

The smoke config uses the same unpacked, no-KL, `5e-5` training semantics and
writes a checkpoint at step one to validate the complete path before the
canonical long run documented below. The long run evaluates AIME
2024 every 100 policy steps and retains only recent policies, checkpoints,
consumed training batches, evaluation sets, and sampled completions needed for
reward-hacking inspection.

## Incorrect synthetic solutions

`generate_incorrect_synthetic.py` selects 100 deterministic, unique Polaris
problems from the hard `1/8` bucket after AIME 2024 decontamination. It samples
eight responses per prompt from a loaded policy adapter. Responses that violate
the exact `<think>...</think><answer>...</answer>` contract or pass Prime's math
verifier are discarded. Reasoning that contains line-start list markers `1.`,
`2.`, and `3.` is also discarded. Only valid-format, non-numbered incorrect
solutions are written to `incorrect.jsonl`. `summary.json` contains aggregate
rejection counts without retaining rejected completions.

With a Wavelet inference server running, generate from a stable adapter using:

```bash
uv run python examples/qwen2_5_7b_polaris/generate_incorrect_synthetic.py \
  --policy-dir outputs/<run>/policies/step-000100 \
  --policy-step 100
```

`generate_wait_recoveries.py` turns those incorrect traces into recovery data.
For each trace it snaps the center of `<think>` to the nearest paragraph or line
boundary and creates one `Alternatively,` prefix and one `Wait,` prefix. It
samples four continuations from each prefix and retains only deduplicated,
strict-format, non-numbered traces that the math verifier marks correct:

```bash
uv run python examples/qwen2_5_7b_polaris/generate_wait_recoveries.py \
  --policy-dir outputs/<run>/policies/step-000100 \
  --policy-step 100 \
  --rollouts 4
```

Browse random verified recovery traces with the read-only viewer:

```bash
uv run python examples/qwen2_5_7b_polaris/serve_recoveries.py --port 8781
```

Train a fresh rank-16 LoRA on the verified recovery traces for three epochs at
learning rate `2e-4`, then evaluate it on all 30 AIME 2024 problems with eight
samples per problem:

```bash
uv run python -m wavelet sft \
  @ examples/qwen2_5_7b_polaris/sft_recoveries.yaml
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/eval_sft.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/eval_sft.yaml
```

The SFT config uses full BF16 LoRA and a global batch size of eight. Its
2,048-token training window covers the verified traces; AIME evaluation retains
the 8,192-token generation window and the same strict tagged system prompt and
verifier used by `eval_baseline.yaml`.

Start the canonical 100,000-step two-GPU GRPO run from that SFT adapter:

```bash
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32.yaml
```

GPU 0 trains while GPU 1 serves vLLM. Each optimizer step contains 32 problems
and eight rollouts per problem, published as four eight-problem chunks. The
trainer uses unpacked microbatches of 16, FlashAttention 2, learning rate
`5e-5`, and no KL loss. Policy lag is bounded by both the four-stage async
pipeline and `max_off_policy_steps`. AIME 2024 runs every 100 policy steps, and
retention keeps only the recent policies, checkpoints, consumed rollouts, eval
sets, and the configured rolling sample history needed for debugging.

To resume, set `model.adapter_path` to the chosen immutable policy adapter and
set a new `output_dir`; checkpoint resume should use the checkpoint controls
instead of adding a step-specific copy of this config. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the shell if the target
machine needs allocator fragmentation mitigation.

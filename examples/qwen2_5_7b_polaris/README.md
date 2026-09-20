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
all-correct and all-wrong groups so the trainer receives mixed groups. Useful
groups remain buffered across bounded retry attempts while rejected groups are
replaced.

AIME 2024 evaluation uses the pinned 30-problem upstream environment source,
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

## Four-node SLURM run

`rl_slurm_4node_128x16.yaml` starts from the base instruct model, without a
recovery SFT adapter. It assigns one eight-GPU node to FSDP training and three
eight-GPU nodes to inference. Each inference node runs eight vLLM data-parallel
workers. An optimizer step uses 128 distinct problems with 16 rollouts each
(2,048 completions), in eight 16-problem chunks; the trainer microbatch is two
completions per GPU. The existing policy freshness limit remains four steps.

Stage the source, Python interpreter, virtual environment, data, model cache,
and outputs on storage visible at the same paths on all nodes. A shared venv
whose Python symlink points into one node's local home directory will not work.
Prepare data with the commands above, adapt the SLURM partition and project
directory to the cluster, and provide W&B credentials through the job environment
or a private file sourced by `slurm.setup_commands`. W&B is enabled under project
`wavelet-polaris`, with one shared run for training and rollout metrics.

```bash
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_slurm_4node_128x16.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_slurm_4node_128x16.yaml
```

The config retains the 100,000-step target and has a 24-hour allocation limit.
AIME 2024 is evaluated before training and every 100 policy steps. Use a new
output directory for retries unless explicitly resuming a stable checkpoint.

### Short diagnostic run

`rl_slurm_4node_16x16.yaml` retains the same placement and model but runs only
20 optimizer steps. It takes the first 16 records in the prepared data file,
keeps their order fixed, and samples 16 responses per problem (256 completions
per update). Zero-advantage filtering is disabled so every update includes all
16 problems and the reward denominator stays fixed. This is an overfitting and
pipeline diagnostic; training reward on this subset is not a generalization
measurement. AIME runs before training, every five steps, and at completion.
The diagnostic disables per-phase live tracing to keep shared-filesystem I/O
out of timing comparisons; metrics and sampled completed rollouts remain enabled.

```bash
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_slurm_4node_16x16.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_slurm_4node_16x16.yaml
```

Use the same prepared 16 records for comparisons with another implementation;
matching only the seed is insufficient if the dataset ordering differs. Inspect
correct and incorrect completions, failed rollout counts, policy lag, and
baseline/final evaluation before interpreting the reward curve. Twenty steps
can diagnose a broken pipeline but do not guarantee a reward increase.

## Incorrect synthetic solutions

`generate_incorrect_synthetic.py` selects 100 deterministic, unique Polaris
problems from the hard `1/8` bucket after AIME 2024 decontamination. It samples
eight responses per prompt from a loaded policy adapter. Responses that violate
the exact `<think>...</think><answer>...</answer>` contract or pass the
upstream math verifier are discarded. Reasoning that contains line-start list
markers `1.`, `2.`, and `3.` is also discarded. Only valid-format,
non-numbered incorrect solutions are written to `incorrect.jsonl`.
`summary.json` contains aggregate
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

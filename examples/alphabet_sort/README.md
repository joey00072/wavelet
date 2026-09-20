# Alphabet Sort

This example trains `Qwen/Qwen3-4B-Instruct-2507` with LoRA rank 32 / alpha 64
to maintain a cumulative alphabetically sorted list across turns.

Wavelet loads `primeintellect/alphabet-sort` through `verifiers`, runs multi-turn rollouts
against an OpenAI-compatible vLLM endpoint, consumes verifier rewards, and
trains on the model-generated assistant turns.

The verifier asks the model to:

- sort by first or last name, chosen per episode
- maintain the prior sorted list across turns
- tag only names introduced in the current turn with `// new name!`
- return the list inside `<alphabetical_sorted>` or
  `<combined_alphabetical_sorted>` tags

Install the verifier extras:

```bash
uv sync --extra verifiers --extra envs
```

For a short 4B learning check, `rl_diagnostic.yaml` uses 16 problem groups ×
16 rollouts, IPO, FP32 optimization with BF16 compute, packed 2,048-token
training sequences, and 768-token responses per turn. It runs 20 updates with
128 fixed evaluation examples at baseline, every 10 updates, and completion.
Evaluation uses the training task distribution; it is not a held-out test.

```bash
uv run python examples/alphabet_sort/prepare_rl_data.py \
  --env-id alphabet-sort --preserve-order --similarity-power 4 \
  --output outputs/alphabet_sort_data/rl_train_source_order.jsonl
uv run python -m wavelet debug preflight @ examples/alphabet_sort/rl_diagnostic.yaml --json
uv run python -m wavelet rl @ examples/alphabet_sort/rl_diagnostic.yaml
```

This config assigns inference to GPU 0 and training to GPU 1. It uses the
model-native serving context; check available GPU memory before launching.
Use `--max_steps 100` for a longer reward curve. Zero-advantage groups are
replaced, so raw generation can exceed 256 episodes per update. The pinned
legacy verifier provides a finite source pass; comparisons against an infinite
native task stream must export that stream when advancing beyond the first pass.

Wavelet's effective policy-age limit is
`min(max_async_level - 1, max_off_policy_steps)`. This diagnostic's async level
2 therefore permits one step of lag. Preflight reports this as
`summary.policy_freshness.effective_max_policy_lag`. Increasing the candidate pool with
`oversampling_factor` while keeping a tight freshness window can discard much
of that extra work; inspect `generation/rollouts/cancelled_total` and
`off_policy/max` along with throughput.

Generate verifier examples:

```bash
uv run python examples/alphabet_sort/prepare_rl_data.py
```

For a task-for-task reference comparison, preserve the verifier taskset's source
order and use the 8B long-run config, which deliberately disables Wavelet's
dataset shuffle:

```bash
uv run python examples/alphabet_sort/prepare_rl_data.py \
  --preserve-order \
  --output outputs/alphabet_sort_data/rl_train_source_order.jsonl
uv run python -m wavelet rl @ examples/alphabet_sort/rl_8b_long.yaml
```

`rl_8b_long.yaml` uses Qwen3-8B, 256-rollout GRPO/IPO, a rank-32 FP32 LoRA
optimizer, a one-inference/one-trainer GPU layout, and source task order. It is
a 100,000-step long-run recipe; use a CLI `--max_steps` override for a short
validation. Like the upstream recipe, it lets vLLM use the model-native context
length. Its 128-rollout ceiling is conservatively bootstrapped from vLLM's KV
capacity and then adjusted from completed-rollout turnover, queue pressure,
preemptions, and KV usage. Set `inference.vllm.max_model_len` explicitly when a
smaller serving context is intentional.

Each row stores one verifier dataset example. During RL, Wavelet calls
`wavelet.orchestrator.verifiers:generate_rollouts`, which runs
`orchestrator.rollouts_per_example` verifier rollouts per example.

Validate the configuration without starting training:

```bash
uv run python -m wavelet rl @ examples/alphabet_sort/rl.yaml --dry_run true
```

Run training on a GPU host:

```bash
uv run python -m wavelet rl @ examples/alphabet_sort/rl.yaml
```

Run a colocated job when training and inference must share one GPU:

```bash
uv run python -m wavelet rl @ examples/alphabet_sort/rl_colocate.yaml
```

The colocated recipe launches vLLM and the trainer as separate processes with
the same `CUDA_VISIBLE_DEVICES`. It caps `inference.vllm.gpu_memory_utilization`
at `0.5`; tune that value for the available GPU memory.

Run sleep-colocated training when vLLM and trainer should alternate ownership of
the same GPU:

```bash
uv run python -m wavelet rl @ examples/alphabet_sort/rl_colocate_sleep.yaml
```

This starts vLLM, sleeps it before trainer startup, wakes it for rollout
generation and policy loading, sleeps it after each rollout batch, then reloads
the trainer model and optimizer for the training step. Because memory ownership
alternates, this mode requires synchronous rollouts.

Run a local multi-role job with two independent inference replicas
and a two-rank FSDP trainer:

```bash
uv run python -m wavelet rl @ examples/alphabet_sort/rl_fsdp_multi.yaml
```

The multi-role config launches two vLLM HTTP servers on ports `8000` and `8001`,
routes verifier rollouts across both endpoints, and launches the trainer through
`torchrun --standalone --nproc-per-node 2`.

For an 8-GPU reward-check recipe that showed a clear 50-step reward lift in
local testing, use:

```bash
uv run python -m wavelet rl @ examples/alphabet_sort/rl_fsdp_multi_reward.yaml
```

This launches one data-parallel vLLM server with four workers on GPUs `0,1,2,3`
and a four-rank FSDP trainer on GPUs `4,5,6,7`. Each generation step targets 64
examples with 8 rollouts each (512 admitted rollouts), with 2x oversampling to
replace zero-advantage groups. It uses packed training,
`max_completion_tokens=768`, and `lr=1e-5` for a stable multi-GPU
alphabet-sort reward run.

The reward recipe also runs verifier evals every 20 exported policy steps and at
the end of training. Eval metrics are written to `eval_metrics.jsonl`, and raw
eval rollouts are written under `evals/step-*`.

## Policy retention during async refresh

Filesystem `policy_transfer.keep_last` is a minimum. Wavelet also retains the
effective lag window plus two snapshots, accounting for the export interval,
so a selected adapter stays available while outstanding requests finish.
Preflight reports this as `policy_freshness.retained_policy_snapshots`.

For a wider-window throughput comparison, retain the same 16×16 admitted batch
while allowing more candidate groups and up to eight steps of policy lag:

```bash
uv run python -m wavelet debug preflight @ examples/alphabet_sort/rl_diagnostic.yaml \
  --orchestrator.max_async_level 9 --orchestrator.oversampling_factor 2 \
  --output_dir outputs/alphabet_sort_async_diagnostic --json
uv run python -m wavelet rl @ examples/alphabet_sort/rl_diagnostic.yaml \
  --orchestrator.max_async_level 9 --orchestrator.oversampling_factor 2 \
  --output_dir outputs/alphabet_sort_async_diagnostic
```

This changes sampling freshness and may change the learning trajectory; compare
fixed-policy evaluations alongside timing and accepted output-token counts.

# Alphabet Sort

This example trains `Qwen/Qwen3-4B-Instruct-2507` with LoRA rank 32 / alpha 64
to maintain a cumulative alphabetically sorted list across turns.

Wavelet loads `alphabet-sort` through `verifiers`, runs multi-turn rollouts
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

Generate verifier examples:

```bash
uv run python examples/alphabet_sort/prepare_rl_data.py
```

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

This launches four vLLM replicas on GPUs `0,1,2,3` and a four-rank FSDP trainer
on GPUs `4,5,6,7`. It uses enforced zero-advantage filtering, 128 rollouts per
step, packed training, `max_completion_tokens=768`, and `lr=1e-5` for a stable
multi-GPU alphabet-sort reward run.

The reward recipe also runs verifier evals every 20 exported policy steps and at
the end of training. Eval metrics are written to `eval_metrics.jsonl`, and raw
eval rollouts are written under `evals/step-*`.

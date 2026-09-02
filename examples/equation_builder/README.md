# Equation Builder RL

This example trains a model to arrange four distinct two-digit numbers with `+`
and `-` so the resulting equation equals a requested target.

Install the local environment and Verifiers dependencies:

```bash
uv sync --extra verifiers
uv pip install --python .venv/bin/python \
  --editable environments/equation_builder
```

Prepare deterministic RL examples:

```bash
uv run --no-sync python examples/prepare_verifier_rl_data.py \
  --env-id equation-builder \
  --output outputs/equation_builder_data/rl_train.jsonl \
  --examples 4096 \
  --env-arg num_examples=4096 \
  --env-arg eval_examples=256 \
  --env-arg seed=42 \
  --env-arg num_numbers=4 \
  --env-arg target_min=0 \
  --env-arg target_max=99
```

Validate and launch:

```bash
uv run --no-sync python -m wavelet debug preflight \
  @ examples/equation_builder/rl.yaml --json
uv run --no-sync python -m wavelet rl \
  @ examples/equation_builder/rl.yaml --dry_run true
uv run --no-sync python -m wavelet rl @ examples/equation_builder/rl.yaml
```

For the single-GPU `Qwen/Qwen2.5-7B-Instruct` QLoRA smoke run, use:

```bash
uv run --no-sync python -m wavelet debug preflight \
  @ examples/equation_builder/rl_qwen2_5_7b_smoke.yaml --json
uv run --no-sync python -m wavelet rl \
  @ examples/equation_builder/rl_qwen2_5_7b_smoke.yaml
```

For the 10,000-step unquantized LoRA run, use:

```bash
uv run --no-sync python -m wavelet debug preflight \
  @ examples/equation_builder/rl_qwen2_5_7b_10k.yaml --json
uv run --no-sync python -m wavelet rl \
  @ examples/equation_builder/rl_qwen2_5_7b_10k.yaml
```

This long-run recipe evaluates every 100 steps and uses a 35% vLLM memory cap
so the BF16 base model, LoRA trainer, and inference server can share a 96 GB GPU.
It uses groups of eight rollouts for each of eight questions per optimizer step
and a sequence length of 1,024 tokens.

The 7B config keeps every consumed rollout batch by setting
`transport.cleanup_consumed: false`. Raw batches remain at
`outputs/equation_builder_qwen2_5_7b_rl/rollouts/step-*/rollouts.jsonl`, and
sampled responses are also appended to `samples.jsonl`. Zero-advantage groups
are retained so the saved population is not biased toward mixed-reward groups.

Run the independent reward-hacking audit after or during training:

```bash
uv run --no-sync python examples/equation_builder/audit_rollouts.py \
  --run-dir outputs/equation_builder_qwen2_5_7b_rl
```

The report is saved as `reward_hacking_audit.json` in the run directory. It
reparses each prompt and completion and flags positive rewards that fail the
standalone equation checker.

The 7B recipe uses `launcher.mode: colocate`: vLLM and the QLoRA trainer share
GPU 0, with vLLM capped at 55% memory utilization. Check available memory before
launching and lower that cap if another process is using the device.

The RL config enables the state server on port 8765. Start the dashboard in a
second terminal:

```bash
cd webui
bun install
bun run dev --host 0.0.0.0
```

Open `http://<host>:5173/?api=http://<host>:8765`.

The default launcher assigns vLLM to GPU 0 and the trainer to GPU 1. Adjust the
`launcher` block before running if those devices are not available.

# Wavelet

<p align="center">
  <img src="assets/wavelet-logo.png" alt="Wavelet logo" width="360">
</p>

Minimal post-training scaffolding for SFT and RL experiments.

## Documentation

Start with the [documentation index](docs/index.md) for the human and
agent-readable map of workflows, diagnostics, and repository guidance.

- [Examples](examples/README.md): runnable configs and environment notes
- Run preflight checks before expensive RL launches:
  `uv run python -m wavelet debug preflight @ examples/reverse_text/rl.yaml --json`
- [Inference diagnostics](docs/inference.skill.md): model serving, policy loading,
  logprobs, and throughput checks
- [Orchestrator diagnostics](docs/orchestrator.skill.md): scheduling, rollout,
  reward, filtering, and materialization checks
- [Agent instructions](AGENTS.md): repository rules for coding agents and
  contributors

## RL Quick Start

Prepare the Unsloth math data:

```bash
uv run python examples/unsloth_math/prepare_sft_data.py
```

Run the SFT warmup:

```bash
uv run python -m wavelet sft @ examples/unsloth_math/sft.yaml
```

Run the RL example:

```bash
uv run python -m wavelet rl @ examples/unsloth_math/rl.yaml
```

Current distributed scope is experimental:

- root FSDP and HSDP-style DP meshes are wired into the trainer bootstrap
- hybrid backend config is accepted
- QLoRA config is accepted
- tensor-parallel model loading/saving now works for full-model paths when the
  selected architecture exposes a `transformers` TP plan
- EP and CP are still not wired into the model kernels or attention stack
- LoRA adapter export from TP-sharded models is not implemented yet

## RL Framework Shape

The RL stack is now split into minimal but scalable pieces:

- `wavelet rl-inference`: owns rollout generation/inference and publishes
  batches into a step-based filesystem queue
- `wavelet rl-trainer`: waits for queued rollout batches and trains one step per
  published batch
- `wavelet rl`: convenience launcher; set `launcher.mode: process` to supervise
  `rl-trainer` and `rl-inference` as separate subprocesses

The transport is intentionally simple and durable: each batch is written under
`<output_dir>/rollouts/step-000000/rollouts.jsonl` with an atomic stable marker.

For distributed trainer jobs, do not run `wavelet rl` under `torchrun`. Start one
inference process and one trainer job:

```bash
uv run python -m wavelet rl-inference @ outputs/my_run/configs/rl_inference.yaml
uv run torchrun --standalone --nproc_per_node=2 -m wavelet rl-trainer @ outputs/my_run/configs/rl_trainer.yaml
```

That command will:

1. Generate math rollouts from `outputs/unsloth_math_data/rl_train.jsonl`
2. Train a LoRA-adapted policy from the generated rollout batches
3. Save the resulting adapter under `outputs/unsloth_math_rl/adapter`

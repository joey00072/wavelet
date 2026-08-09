# Wavelet

<p align="center">
  <img src="assets/wavelet-logo.png" alt="Wavelet logo" width="360">
</p>

Minimal post-training scaffolding for SFT and RL experiments.

## Documentation

Start with the [documentation index](docs/index.md) for the human and
agent-readable map of workflows, diagnostics, and repository guidance.

- [Examples](examples/README.md): runnable configs and environment notes
- [Architecture](docs/architecture.md): subsystem ownership, process flow, and
  extension boundaries
- [Data pipeline](docs/data_pipeline.md): loading, tokenization, RL packing, and
  collation contracts
- Run preflight checks before expensive RL launches:
  `uv run python -m wavelet debug preflight @ examples/reverse_text/rl.yaml --json`
  The report includes low-precision checks for QLoRA and vLLM quantized
  inference settings.
- [Inference diagnostics](docs/inference.skill.md): model serving, policy loading,
  logprobs, and throughput checks
- [Orchestrator diagnostics](docs/orchestrator.skill.md): scheduling, rollout,
  reward, filtering, and materialization checks
- [RL algorithms](docs/algorithms.md): configure GRPO, MaxRL, OPD, mixed
  per-source algorithms, or algorithms from user-owned Python files
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

RL advantage assignment is selected explicitly with `algo.type`. The
`reverse_text` example uses GRPO; see the [algorithm guide](docs/algorithms.md)
for the supported choices and their contracts. The mixed reverse-text example
combines GRPO and OPD against an external frozen teacher while training one
student and one LoRA adapter. Start the teacher described in the algorithm
guide before launching it:

```bash
uv run python -m wavelet rl @ examples/reverse_text/rl_mixed_grpo_opd.yaml
```

Current distributed scope is experimental:

- root FSDP and HSDP-style DP meshes are wired into the trainer bootstrap
- hybrid backend config is accepted
- QLoRA config is accepted for LoRA adapter training with replicated DDP;
  preflight rejects unsupported full-model 4-bit, FSDP, tensor-parallel, and
  `colocate_sleep` combinations
- exported policies record trainer, adapter, and inference precision metadata
  beside the policy artifact for later parity checks
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
When rollout metadata is enabled, each batch also records payload size and
transfer time in the manifest and queue event log for transfer observability.
Trainer receivers can also emit queue wait time and payload-byte events when
they claim filesystem batches.
Inference policy receivers emit matching wait-time and payload-byte events when
they observe exported policies.

Implementation ownership is similarly explicit: `wavelet.transport` owns queue
and policy transfer, `wavelet.orchestrator.scheduler` owns rollout scheduling,
`wavelet.orchestrator.envs` owns verifier clients and evaluation, and
`wavelet.data.sft` / `wavelet.data.rl` own the two data pipelines. Historical
module paths remain import-compatible.

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

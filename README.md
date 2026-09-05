# Wavelet

<p align="center">
  <img src="assets/wavelet-logo.png" alt="Wavelet logo" width="360">
</p>

Small, explicit post-training infrastructure for SFT and online RL experiments.

Wavelet keeps training, rollout generation, policy transfer, and run artifacts
separate. You can run the pieces together for development or as independent
processes for distributed jobs.

## Setup

Wavelet supports Python 3.11 and 3.12 and uses
[uv](https://docs.astral.sh/uv/) for environments and commands.

```bash
uv sync
uv run wavelet --help
```

Commands accept YAML configs after `@`:

```bash
uv run wavelet debug preflight @ examples/reverse_text/rl.yaml --json
```

Multiple config files compose from left to right, and dotted command-line
overrides apply last.

## Quick Start

Run supervised fine-tuning:

```bash
uv run wavelet sft @ examples/reverse_text/sft.yaml
```

Prepare the reverse-text rollout data and run RL:

```bash
uv sync --extra envs
uv run python examples/reverse_text/prepare_rl_data.py
uv run wavelet debug preflight @ examples/reverse_text/rl.yaml --json
uv run wavelet rl @ examples/reverse_text/rl.yaml
```

The example config documents its GPU placement and output directory. Browse
[examples](examples/README.md) for smaller smoke runs, QLoRA recipes, math
environments, and multi-GPU configurations.

For multi-GPU SFT, launch the trainer explicitly:

```bash
uv run torchrun --standalone --nproc-per-node=2 \
  -m wavelet sft @ path/to/sft.yaml
```

Do not wrap the combined `wavelet rl` launcher in `torchrun`.

## Dashboard

The dashboard reads live and completed run directories without modifying them.
It includes training metrics, rollout groups, evaluations, queue state,
infrastructure, resolved config, and run comparison.

Build the browser app once, then serve your runs:

```bash
cd webui
bun install
bun run build
cd ..
uv run wavelet dashboard --runs-root outputs
```

Open `http://127.0.0.1:8766`.

To try the dashboard without a GPU, generate a synthetic run first:

```bash
uv run wavelet synth-run --output outputs/demo_run
uv run wavelet dashboard --runs-root outputs
```

See the [dashboard guide](webui/README.md) for live-state-server and remote API
options.

## RL Framework Shape

The online RL path has three clear owners:

- `wavelet rl-inference` generates and scores rollout groups.
- `wavelet rl-trainer` consumes rollout batches and updates the policy.
- `wavelet rl` resolves config and supervises both roles for a normal run.

Rollouts and policy snapshots move through durable filesystem queues under the
configured `output_dir`. Logs, metrics, checkpoints, evaluations, and resolved
configs live beside them, so a run can be inspected or resumed without relying
on process-local state.

For a manually split job, start one inference process and one distributed
trainer job from the resolved role configs:

```bash
uv run wavelet rl-inference @ outputs/my_run/configs/rl_inference.yaml
uv run torchrun --standalone --nproc-per-node=2 \
  -m wavelet rl-trainer @ outputs/my_run/configs/rl_trainer.yaml
```

The preflight command reports resolved role commands, device placement, missing
data, port conflicts, and incompatible settings before an expensive launch.

## Documentation

- [Documentation index](docs/index.md)
- [Examples](examples/README.md)
- [Architecture](docs/architecture.md)
- [Configuration and data pipeline](docs/data_pipeline.md)
- [RL algorithms](docs/algorithms.md)
- [Deployment](docs/deployment.md)
- [Inference diagnostics](docs/inference.skill.md)
- [Orchestrator diagnostics](docs/orchestrator.skill.md)
- [Dashboard](webui/README.md)
- [Contributor and agent guidance](AGENTS.md)

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

Build the frontend with `bun run build` from `webui/`. Generated run artifacts,
checkpoints, and dashboard build output are not source files and should not be
committed.

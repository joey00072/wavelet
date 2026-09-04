# Wavelet Documentation

Wavelet docs are written for both people and coding agents. Keep commands
copy-pasteable from the repository root, prefer JSON diagnostics for machine
comparison, and make the expected inspection path explicit before expensive
training or inference runs.

## Start Here

- [README](../README.md): project overview, quick start, and the RL process
  split.
- [Examples](../examples/README.md): runnable example status, environment notes,
  and verifier data preparation.
- [RL algorithms](algorithms.md): named advantage assignment, configuration,
  compatibility, and extension points.
- [Architecture](architecture.md): package ownership, RL process flow,
  scheduling, policy artifacts, and extension boundaries.
- [Data pipeline](data_pipeline.md): source loading, normalization,
  tokenization, RL packing, and collation contracts.
- [FSDP2 migration](fsdp2_migration.md): opt-in configuration, wrapper and
  checkpoint design, compatibility boundary, and validation status.
- [Inference diagnostics](inference.skill.md): isolate model serving, policy
  loading, generation, logprobs, and throughput.
- [Orchestrator diagnostics](orchestrator.skill.md): inspect scheduling,
  rollout generation, scoring, filtering, and rollout materialization without
  starting the trainer.
- [Agent trajectory artifacts](agent_trajectory.md): token provenance contract
  for custom multi-turn and tool rollouts.
- [WebUI](../webui/README.md): run the browser dashboard for the RL state server.
- [Agent instructions](../AGENTS.md): repository rules for coding agents and
  contributors.

## Common Workflows

### First Local Smoke Run

Use the smallest example that exercises the subsystem you changed.

```bash
uv sync
uv run python examples/reverse_text/prepare_rl_data.py
uv run python -m wavelet debug preflight @ examples/reverse_text/rl.yaml --json
uv run python -m wavelet rl @ examples/reverse_text/rl.yaml
```

For a deterministic debug-only pass before launching RL:

```bash
uv run python -m wavelet debug orchestrator inspect @ examples/reverse_text/rl.yaml --json
uv run python -m wavelet debug orchestrator sample \
  @ examples/reverse_text/rl.yaml --step 0 --examples 2 --json
```

Run the sample command twice with the same `--step` and `--examples`; selected
rows should match because the example uses a fixed seed.

For a math SFT-to-RL path:

```bash
uv run python examples/unsloth_math/prepare_sft_data.py
uv run python -m wavelet sft @ examples/unsloth_math/sft.yaml
uv run python -m wavelet rl @ examples/unsloth_math/rl.yaml
```

### Debug Inference First

Before changing RL hyperparameters, prove that inference can serve the intended
policy and attach the training metadata that RL needs.

```bash
uv run python -m wavelet debug inference inspect @ examples/wordle/rl.yaml --json
uv run python -m wavelet debug inference health @ examples/wordle/rl.yaml --json
uv run python -m wavelet debug inference smoke \
  @ examples/wordle/rl.yaml --count 2 --json
```

Then use the full [inference diagnostics](inference.skill.md) runbook if policy
step, logprob, token accounting, or throughput looks wrong.

### Debug Orchestration Without Trainer

Keep the trainer stopped until the orchestrator can select examples, generate
and score rollouts, assign advantages, and write trainable batches.

```bash
uv run python -m wavelet debug orchestrator inspect @ examples/wordle/rl.yaml --json
uv run python -m wavelet debug orchestrator sample \
  @ examples/wordle/rl.yaml --step 0 --examples 4 --json
uv run python -m wavelet debug orchestrator materialize \
  @ examples/wordle/rl.yaml --step 0 --examples 2 --rollouts 2 --json
```

Then use the full [orchestrator diagnostics](orchestrator.skill.md) runbook to
interpret timings, reward metrics, trainable records, and failure patterns.

### Debug Trainer Inputs Without Inference

Use a saved rollout batch to check trainer-facing token artifacts before
starting the trainer.

```bash
uv run python -m wavelet debug trainer inspect \
  --rollout-path outputs/my_run/rollouts/step-000000/rollouts.jsonl --json \
  @ examples/reverse_text/rl.yaml
```

The report validates `input_ids`, `target_ids`, `loss_mask`, rollout logprobs,
and teacher logprobs when present. It proves the saved batch is structurally
ready for trainer replay; it does not compare model logprobs or run an optimizer
step.

Export a compact token sidecar when a batch needs closer inspection:

```bash
uv run python -m wavelet debug trainer tokens \
  --rollout-path outputs/my_run/rollouts/step-000000/rollouts.jsonl \
  --write-tokens outputs/my_run/debug/trainer-step-000000-tokens.jsonl \
  --json @ examples/reverse_text/rl.yaml
```

The token sidecar contains token ids, loss masks, trainable target ids,
rollout-time logprobs, temperatures, rewards, advantages, and row provenance.
It is intended for debugging and should stay under the run output directory.

When trainer-side logprobs have been exported into the rollout JSONL, compare
them with rollout-time logprobs:

```bash
uv run python -m wavelet debug trainer parity \
  --rollout-path outputs/my_run/rollouts/step-000000/rollouts.jsonl \
  --trainer-logprobs-column trainer_logprobs \
  --write-report outputs/my_run/parity/runtime-step-000000.json \
  --json @ examples/reverse_text/rl.yaml
```

If trainer logprobs are absent, the parity report records an explicit skip
reason instead of silently claiming parity.

### Run Split RL Processes

Use the split commands when the trainer needs distributed launch or when
inference and training need different GPU placement. Do not wrap the combined
`wavelet rl` launcher in `torchrun`.

```bash
uv run python -m wavelet rl-inference @ outputs/my_run/configs/rl_inference.yaml
uv run torchrun --standalone --nproc_per_node=2 \
  -m wavelet rl-trainer @ outputs/my_run/configs/rl_trainer.yaml
```

### Add Or Change An Example

When an example changes behavior, update:

- the example config or helper under `examples/<name>/`
- [examples/README.md](../examples/README.md)
- a small smoke config when the main config is GPU-memory-sensitive or slow
- tests under `tests/` when config normalization, scheduling, rollout queues,
  policy sync, reward calculation, or trainer behavior changes

## Agent Runbook Contract

Every diagnostic guide should make the next action obvious to a coding agent:

- State which subsystem is isolated and which processes should be stopped.
- Put commands before interpretation notes.
- Prefer `--json` output when the command supports it.
- Name the metrics or fields that prove the subsystem is healthy.
- Include common failure patterns and the next subsystem to inspect.
- Keep generated outputs under the configured run directory.
- Update this index when adding a new runbook or public workflow.

## Verification

For documentation-only changes, run:

```bash
git diff --check
```

For code changes, run a targeted test for the changed behavior, then:

```bash
uvx ruff check wavelet tests
uvx ruff format --check wavelet tests
git diff --check
```

Install the optional `dev` dependencies and enable the equivalent staged-file
checks with `uv run pre-commit install`. To check the full tracked Python
surface used by CI without installing a hook, run
`uv run pre-commit run --all-files`.

Pytest rejects unknown markers. Registered suites can be selected or excluded
with expressions such as `uv run pytest -m integration` or
`uv run pytest -m "not gpu and not slow"`; tests marked `gpu` must also guard
themselves when CUDA is unavailable.

The CPU integration suite includes a real reverse-text SFT subprocess followed
by checkpoint resume. It verifies continuous step/token progress, a lower final
training loss, and a stable post-resume checkpoint:

```bash
uv run pytest tests/integration/test_reverse_text_sft.py -q
```

Pull requests and pushes to `master` run the same Ruff checks plus separate CPU
unit and integration jobs from the locked `uv` environment. The manual GPU
workflow targets a self-hosted runner carrying the `linux` and `gpu` labels;
GPU-only coverage remains an explicit release or hardware-runner check.

Track project size when a change adds or removes meaningful surface area:

```bash
uv run python scripts/project_size.py --write docs/project_size.jsonl
```

The snapshot records file count, total lines, source lines, Python function/class
counts, a rough complexity proxy, and the largest files. It intentionally skips
external checkouts, generated outputs, caches, virtualenvs, and WandB artifacts.

For repeatable performance checks, use the hardware-keyed benchmark harness and
reviewed baselines described in [benchmarks/README.md](../benchmarks/README.md).

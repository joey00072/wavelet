# Reverse Text Smoke Path

Use this path before a long RL run when you need deterministic checks on the
small reverse-text example.

## Prepare Data

```bash
uv run python examples/reverse_text/prepare_rl_data.py
```

## Check Config And Split Commands

```bash
uv run python -m wavelet debug preflight \
  @ examples/reverse_text/rl.yaml --json
```

The report should show:

- `ok: true`
- resolved trainer, inference, and orchestrator commands
- low-precision checks that are either passed or explicitly skipped
- `paths.queue_dir` under `outputs/reverse_text_rl/rollouts`

## Check Orchestrator Selection

This does not start inference or the trainer. It proves data loading,
scheduling, example selection, and rollout grouping are deterministic.

```bash
uv run python -m wavelet debug orchestrator inspect \
  @ examples/reverse_text/rl.yaml --json
uv run python -m wavelet debug orchestrator sample \
  @ examples/reverse_text/rl.yaml --step 0 --examples 2 --json
```

Run the `sample` command twice with the same `--step` and `--examples`; the
selected sample should be identical because the example seed is fixed.

## Optional Inference Probe

Use this only when a compatible inference server/model can start on the selected
GPU. Keep count small and JSON output on.

```bash
uv run python -m wavelet debug inference inspect \
  @ examples/reverse_text/rl.yaml --json
uv run python -m wavelet debug inference smoke \
  @ examples/reverse_text/rl.yaml \
  --count 1 \
  --prompt "Reverse this text: abc" \
  --json
```

The smoke report should show at least one trainable token and inference
logprobs on the sample.

## Inspect A Saved Rollout Batch

After `wavelet rl`, `wavelet rl-inference`, or orchestrator materialization has
written a rollout JSONL, check the trainer-facing token artifact before replay.

```bash
uv run python -m wavelet debug trainer inspect \
  --rollout-path outputs/reverse_text_rl/rollouts/step-000000/rollouts.jsonl \
  --json @ examples/reverse_text/rl.yaml
uv run python -m wavelet debug trainer tokens \
  --rollout-path outputs/reverse_text_rl/rollouts/step-000000/rollouts.jsonl \
  --write-tokens outputs/reverse_text_rl/debug/step-000000-tokens.jsonl \
  --json @ examples/reverse_text/rl.yaml
```

The inspect report should have `ok: true`, nonzero `trainable_tokens`, and
aligned rollout logprobs when present. The token sidecar is for local debugging
and should remain under the run output directory.

## Parity Report

When trainer-side logprobs have been added to the rollout JSONL, run:

```bash
uv run python -m wavelet debug trainer parity \
  --rollout-path outputs/reverse_text_rl/rollouts/step-000000/rollouts.jsonl \
  --trainer-logprobs-column trainer_logprobs \
  --write-report outputs/reverse_text_rl/parity/step-000000.json \
  --json @ examples/reverse_text/rl.yaml
```

If trainer logprobs are not present yet, the command should return a skipped
report with an explicit skip reason. That is still useful: it proves the parity
gate is wired without silently claiming agreement.

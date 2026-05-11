# Orchestrator Diagnostics Skill

Use this guide when debugging Wavelet orchestration without starting the trainer.
The goal is to verify scheduling, example selection, rollout generation, reward
assignment, advantage shaping, filtering, and materialization timing in isolation.

## First Principles

- Treat the orchestrator as a dataflow: load examples, select step records,
  generate rollouts, score rewards, assign advantages, filter, then write.
- Measure each boundary separately before blaming the trainer.
- Keep trainer stopped until the orchestrator can produce trainable rollouts with
  the expected policy step, reward distribution, and sequence lengths.
- Use small `--examples` and `--rollouts` limits first, then scale up.
- Prefer JSON output so another agent can compare runs across configs.

## Commands

Inspect schedule and rollout settings:

```bash
uv run python -m wavelet orchestrator-debug inspect @ examples/wordle/rl.yaml --json
```

Sample selected examples without generation:

```bash
uv run python -m wavelet orchestrator-debug sample @ examples/wordle/rl.yaml --step 0 --examples 4 --json
```

Benchmark one small Wordle rollout step against a running inference server:

```bash
uv run python -m wavelet orchestrator-debug benchmark @ examples/wordle/rl.yaml --step 0 --examples 1 --rollouts 1 --json
```

Benchmark a larger Wordle rollout step:

```bash
uv run python -m wavelet orchestrator-debug benchmark @ examples/wordle/rl.yaml --step 0 --examples 4 --rollouts 8 --json
```

Materialize a rollout file without trainer:

```bash
uv run python -m wavelet orchestrator-debug materialize @ examples/wordle/rl.yaml --step 0 --examples 2 --rollouts 2 --json
```

For native passthrough or dataset-only checks, skip inference setup:

```bash
uv run python -m wavelet orchestrator-debug benchmark @ path/to/rl.yaml --no-inference --json
```

## What To Read

- `timings.load_records`: dataset loading/parsing time.
- `timings.select_records`: deterministic step selection time.
- `timings.generate_score`: custom verifier/native rollout generation, reward
  scoring, and advantage assignment.
- `timings.filter_zero_advantage`: post-advantage filtering cost.
- `timings.write`: JSONL materialization cost when using `materialize`.
- `records_selected`: number of base examples selected for the optimizer step.
- `records_scored`: number of rollout records after expansion/generation.
- `records_trainable`: rollout records that survive filtering.
- `metrics.progress/tokens`: total rollout sequence tokens.
- `metrics.progress/decode_tokens`: generated/trainable token proxy.
- `metrics.reward/all/mean`: reward level for the probed batch.
- `metrics.effective_batch_size/all`: how much useful training signal survived.

## Wordle Workflow

1. Start only the vLLM inference server, not trainer.
2. Run `wavelet inference-debug health @ examples/wordle/rl.yaml --json`.
3. Run `wavelet orchestrator-debug inspect @ examples/wordle/rl.yaml --json`.
4. Run `sample --examples 4` to verify data is present and step selection works.
5. Run `benchmark --examples 1 --rollouts 1` to measure one rollout.
6. Increase to `--examples 4 --rollouts 8`, then compare `generate_score` time,
   decode tokens, reward mean, and effective batch size.
7. Only after this is stable, start full RL.

## Failure Patterns

- `records_selected` is zero: dataset path, split, or examples-per-step is wrong.
- `records_scored` is lower than expected: rollout function, inference failures,
  or verifier filtering dropped outputs.
- `records_trainable` is zero: all advantages were zero, completions were empty,
  or reward parsing failed.
- High `load_select` time: dataset loading is the bottleneck; cache or reduce
  repeated parsing.
- High `generate_score` time: inference, verifier environment latency, or rollout
  concurrency is the bottleneck.
- Good orchestrator metrics but bad trainer metrics: focus next on dataset
  collation, logprob alignment, KL, and optimizer behavior.

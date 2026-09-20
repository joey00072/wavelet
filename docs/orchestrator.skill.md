# Orchestrator Diagnostics Runbook

Use this human- and agent-readable guide when debugging Wavelet orchestration
without starting the trainer. The goal is to verify scheduling, example
selection, rollout generation, reward assignment, advantage shaping, filtering,
and materialization timing in isolation.

Related docs: [documentation index](index.md).

## First Principles

- Treat the orchestrator as a dataflow: load examples, select step records,
  generate rollouts, score rewards, assign advantages, filter, then write.
- Measure each boundary separately before blaming the trainer.
- Keep trainer stopped until the orchestrator can produce trainable rollouts with
  the expected policy step, reward distribution, and sequence lengths.
- Use small `--examples` and `--rollouts` limits first, then scale up.
- Prefer JSON output so another agent can compare runs across configs.

## Async throughput invariants

Independently scored rollouts consume one inference slot when dispatched, even
when their advantage group contains multiple rollouts. A new group can begin as
soon as one slot is free. Environments that score a whole group together still
require enough slots for the entire group. Candidate budgets, policy freshness,
and group completion requirements apply in both cases.

Within one generated batch, token counting, trainability checks, and finalization
reuse converted trajectory records. The cache retains pristine records so
filtering cannot erase tokens needed by later distillation; it is cleared on
batch completion, retry, and shutdown. Do not reuse it across policy batches.

Async publication runs JSONL writing and queue copying in worker threads so
in-flight tool and model requests can progress. Publication remains ordered and
awaited: failures propagate and no partial file becomes a stable queue batch.
Metrics use the in-memory serialized rows instead of rereading the artifact.
Measure generation, materialization, publication, and training separately;
improvements in one stage do not establish end-to-end throughput parity.

## Commands

Inspect schedule and rollout settings:

```bash
uv run python -m wavelet debug orchestrator inspect @ examples/wordle/rl.yaml --json
```

Sample selected examples without generation:

```bash
uv run python -m wavelet debug orchestrator sample @ examples/wordle/rl.yaml --step 0 --examples 4 --json
```

Benchmark one small Wordle rollout step against a running inference server:

```bash
uv run python -m wavelet debug orchestrator benchmark @ examples/wordle/rl.yaml --step 0 --examples 1 --rollouts 1 --json
```

Benchmark a larger Wordle rollout step:

```bash
uv run python -m wavelet debug orchestrator benchmark @ examples/wordle/rl.yaml --step 0 --examples 4 --rollouts 8 --json
```

Materialize a rollout file without trainer:

```bash
uv run python -m wavelet debug orchestrator materialize @ examples/wordle/rl.yaml --step 0 --examples 2 --rollouts 2 --json
```

For native passthrough or dataset-only checks, skip inference setup:

```bash
uv run python -m wavelet debug orchestrator benchmark @ path/to/rl.yaml --no-inference --json
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
2. Run `wavelet debug inference health @ examples/wordle/rl.yaml --json`.
3. Run `wavelet debug orchestrator inspect @ examples/wordle/rl.yaml --json`.
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
- To read individual rollouts from a published batch rather than the sampled
  diagnostics above, start `uv run wavelet dashboard --runs-root outputs` and
  use the Traces tab to select a step, filter retained episodes, and open the
  three-pane transcript and metadata viewer. The Metrics
  tab can search and chart every `reward/*`, `fate/*`, `generation/*`, and
  `off_policy/*` signal from `orchestrator_metrics.jsonl`.

### Survivor-based rollout batches

For a fixed count of clean episodes, set `batch_selection: rollouts`,
`examples_per_step`, `rollouts_per_example`, and `refill_zero_advantage: false`.
The target is the product of the two counts before zero-advantage pruning.
Independent request failures finish group slots and are never converted into
reward-zero examples or retried to bias the group. Advantages use the surviving
clean outputs. The scheduler scores complete terminal groups, cuts exactly at
the rollout target, and buffers the scored tail without recomputing advantages.
Policy freshness still applies to the buffered tail. An all-zero training batch
is discarded with bounded retries and no optimizer update. Exact zero RL credit
is filtered in this mode; auxiliary CE/reference-KL components remain trainable.

The default `groups` selection retains complete-group/retry behavior. Environments
requiring joint verifier group scoring still need complete scoring calls, even
with rollout-count batching. Survivor selection currently requires a fixed
`examples_per_step` target and does not combine with effective-group refilling.

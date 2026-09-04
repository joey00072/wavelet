# Wavelet

<p align="center">
  <img src="assets/wavelet-logo.png" alt="Wavelet logo" width="360">
</p>

Minimal post-training scaffolding for SFT and RL experiments.

After `uv sync`, run the installed CLI as `uv run wavelet <command>`. The
module form, `uv run python -m wavelet <command>`, remains supported.
Pass multiple `@ config.yaml` arguments to compose configs from left to right,
then use dotted overrides such as `--inference.sampling.top-p 0.9`. Override
names accept kebab-case, and boolean fields accept bare `--flag` and
`--no-flag` forms. Run a config-backed command with `--help` to list its nested
fields, types, defaults, and available field descriptions.

## Documentation

Start with the [documentation index](docs/index.md) for the human and
agent-readable map of workflows, diagnostics, and repository guidance.

- [Examples](examples/README.md): runnable configs and environment notes
- [Architecture](docs/architecture.md): subsystem ownership, process flow, and
  extension boundaries
- [Data pipeline](docs/data_pipeline.md): loading, tokenization, RL packing, and
  collation contracts
- Run preflight checks before expensive RL launches:
  `uv run wavelet debug preflight @ examples/reverse_text/rl.yaml --json`
  This also validates local training data and required LoRA adapter artifacts.
  The report includes low-precision checks for QLoRA and vLLM quantized
  inference settings. Config loading rejects duplicate YAML keys instead of
  silently accepting the last value.
- [Inference diagnostics](docs/inference.skill.md): model serving, policy loading,
  logprobs, and throughput checks
- [Orchestrator diagnostics](docs/orchestrator.skill.md): scheduling, rollout,
  reward, filtering, and materialization checks
- [RL algorithms](docs/algorithms.md): configure GRPO, MaxRL, reward, or
  passthrough advantage assignment, including algorithms from user-owned
  Python files
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

SFT configs may include a `val` block with independent data settings,
`eval_on_start`, and an optimizer-step `interval`. Validation runs for one
finite epoch without gradients and records token-weighted `val/loss`; see
`examples/reverse_text/sft.yaml` for a configured example.

SFT configs do not self-launch distributed workers. For multi-GPU SFT, launch
the command explicitly:

```bash
uv run torchrun --standalone --nproc-per-node=N \
  -m wavelet sft @ <config>.yaml
```

The former `deployment` block is rejected because it did not affect process
launch.

To continue training an existing LoRA, set `model.adapter_path` to its adapter
directory. Wavelet uses tokenizer artifacts from that directory when present
and otherwise falls back to the tokenizer named by `model.name`.
Keep that adapter outside a new run's `output_dir` when `clean_output_dir: true`;
preflight rejects layouts where cleanup would delete the input adapter, and the
launchers likewise refuse to delete the configured `data.path` inputs or an
`output_dir` that resolves to the filesystem root, the home directory, or the
current working directory and its parents.

Run the RL example:

```bash
uv run python -m wavelet rl @ examples/unsloth_math/rl.yaml
```

RL advantage assignment is selected explicitly with `algo.type`. The
`reverse_text` example uses GRPO; see the [algorithm guide](docs/algorithms.md)
for the supported choices and their contracts.

Current distributed scope is experimental:

- FSDP1 remains the default; `fsdp.impl: fsdp2` opts into per-transformer-block
  `fully_shard` plus a root shard, using the existing HSDP mesh. See the
  [FSDP2 migration guide](docs/fsdp2_migration.md) for the tested boundary and
  pending meta-device weight-loading work
- `fsdp.backend: auto` selects `cpu:gloo,cuda:nccl` on CUDA so async
  checkpoints, which require a CPU backend in the default group, work out of
  the box; the explicit `hybrid` setting remains accepted
- QLoRA config is accepted for LoRA adapter training with replicated DDP;
  preflight rejects unsupported full-model 4-bit, FSDP, tensor-parallel, and
  `colocate_sleep` combinations
- exported policies record trainer, adapter, and inference precision metadata
  beside the policy artifact for later parity checks
- tensor-parallel model loading/saving now works for full-model paths when the
  selected architecture exposes a `transformers` TP plan
- EP and CP degrees above 1 are rejected during config loading because they are
  not yet wired into the model kernels or attention stack
- LoRA adapter export from TP-sharded models is not implemented yet

## RL Framework Shape

The RL stack is now split into minimal but scalable pieces:

- `wavelet rl-inference`: owns rollout generation/inference and publishes
  batches into a step-based filesystem queue
- `wavelet rl-trainer`: waits for queued rollout batches and trains one step per
  published batch
- `wavelet rl`: convenience launcher; set `launcher.mode: process` to supervise
  `rl-trainer` and `rl-inference` as separate subprocesses

Each `wavelet rl` launch allocates matching
`<output_dir>/{configs,logs}/attempt_N/` directories. The config attempt stores
the command, a copy of the supplied root config, and resolved role configs;
supervised role output goes to the matching log attempt. `latest` symlinks make
the current attempt easy to inspect without overwriting resume history.
Ray-backed launchers create the selected log directory on the worker node and
disconnect from Ray during teardown after their role handles are closed. A
`SIGTERM` sent to `wavelet rl` (systemd, SLURM, `timeout`) tears down the role
processes exactly like Ctrl-C instead of orphaning GPU workers.

The transport is intentionally simple and durable: each batch is written under
`<output_dir>/rollouts/step-000000/rollouts.jsonl` with an atomic stable marker.
When rollout metadata is enabled, each batch also records payload size and
transfer time in the manifest and queue event log for transfer observability.
Trainer receivers can also emit queue wait time and payload-byte events when
they claim filesystem batches.
Inference policy receivers emit matching wait-time and payload-byte events when
they observe exported policies. LoRA snapshots record the tensor artifact size
without rereading the file. Inference advances its loaded policy step only after
every server acknowledges loading the immutable, versioned snapshot.

Process and colocated training require `policy_transfer.export_initial: true`.
Filesystem policy exports are transient transport artifacts: Wavelet keeps the
current and previous snapshot by default and requires `keep_last >= 2`. Use
trainer checkpoints, not policy exports, for durable resume history.
Consumed rollout queues likewise retain the latest two batches by default.
Increase `transport.keep_last_consumed` for a larger reward-hacking audit
window; do not disable cleanup for long runs.
Checkpoint and evaluation-rollout retention also default to the latest two
sets when those features are enabled. Metrics and traces remain the compact
long-term record. Set `ckpt.keep_interval` to retain every Nth stable checkpoint
permanently in addition to the `ckpt.keep_last` rolling window. A successful
run also saves its final optimizer step when that step is between configured
checkpoint intervals.
Sample logging retains a rolling window of 256 rows by default
(`monitor.samples.keep_last`) and compacts it in batches, so enabling rollout
examples cannot grow `samples.jsonl` without bound during a long run.
Role logging writes structured JSONL under `<output_dir>/logs/` by default.
Set `log.json_file: false` to disable that file sink or
`log.json_console: true` to emit the same structured records to the console;
distributed ranks receive distinct `.rank-N.jsonl` files.
This publishes policy step 0 before rollout generation and prevents the trainer
and inference scheduler from waiting on each other at startup.

Checkpoint resume is absolute-step based. Both trainer and process-mode rollout
scheduler restart from the resolved checkpoint optimizer step; streaming modes
convert that step to the corresponding queue-chunk offset. Completed runs wait
for a pending async checkpoint to become stable before process teardown. A
trainer checkpoint also persists cumulative global model-token and logical
sample counts, which resume without resetting the corresponding
`progress/total_tokens` and `progress/total_samples` metrics. Checkpoints created
before these counters were added resume them from zero. A
restored trainer also forces one policy export at the checkpoint step, even when
that step is between normal export intervals, and removes newer snapshots left by
the interrupted run for both filesystem and NCCL transports. The resumed
orchestrator likewise discards rollout queue batches (and materialized files)
beyond the checkpoint step, because the interrupted run generated them with
policy versions the resumed run re-derives differently; batches for the resume
step itself stay reusable. Async checkpoints
finish copying the live tensors to their staging buffer before the trainer
returns to the next optimizer step; only the upload continues in the background.

Convert a stable full-model DCP checkpoint into an inference-ready Hugging Face
safetensors directory with:

```bash
uv run wavelet convert-checkpoint outputs/run/checkpoint-100
```

The default destination is `checkpoint-100/weights`. The converter discovers
`rl_trainer.yaml` or `sft.yaml` from the run's latest resolved config; pass
`--config PATH` when checkpoints use a separate volume. It runs in one process,
requires enough CPU memory for the full model, refuses incomplete and non-empty
destinations, and intentionally rejects LoRA, adapter-backed, and 4-bit runs.
Set `ckpt.output_dir` to place large checkpoint step directories on a separate
volume without moving logs, rollouts, policies, or other run state out of the
top-level `output_dir`. Resume and preflight resolve the same checkpoint volume.
Trainer metrics report byte counts and free-space ratios for both the run and
checkpoint filesystems, including when either configured directory has not yet
been created.
Set `ckpt.resume_dir` to a stable `checkpoint-N` directory from another run to
fork its state into the configured output directory; it is mutually exclusive
with `ckpt.resume_step`. The `ckpt.skip_optimizer`, `skip_scheduler`,
`skip_dataloader`, and `skip_progress` flags selectively keep fresh local state
while loading the checkpoint model. Skipping progress restarts optimizer-step,
token, sample, rollout-queue, and policy version counters from zero; skipping
the scheduler rebuilds it from the configured schedule and learning rate over
the remaining optimizer steps.
Run-directory cleanup refuses to start when it would delete the configured
input adapter, even if the launch skips the optional preflight command.
SFT examples longer than `data.seq_len` train on the available assistant-token
prefix; an end-of-sequence token is required in the rendered sample, but it need
not fit inside the truncated context window.
Concatenative SFT packing checkpoints both the source-row position and its
unconsumed token remainder, so resuming does not skip tokens already read into a
partially filled packed sequence.
Static packed-RL checkpoints likewise persist the next packed-bin cursor rather
than replaying the current epoch from its first bin.

With `orchestrator.filter_zero_advantage: true`, the persistent verifier
scheduler resamples zero-signal groups until `examples_per_step` admitted groups
are available. Rejected groups remain visible in scheduler diagnostics but do
not silently occupy most of an optimizer batch as zero-loss rows.
Finite verifier training datasets are traversed once per epoch instead of
sampled with replacement. `data.shuffle: true` uses a deterministic permutation
for each epoch; sampler cursor and epoch are included in generation metrics.
Synchronous and native rollout paths apply the same full-group-count invariant;
a single surviving group cannot silently stand in for a requested batch, and
native groups must contain exactly the configured rollout count.
The orchestrator separately logs `generation/reward/mean`, group admission,
and generated solve-rate metrics before filtering. Use those raw generation
metrics to judge policy progress; reward on admitted mixed groups is
selection-biased by design.
Fresh schedulers initialize evaluation from an unevaluated state, so
`eval_base_model: true` records policy step 0 before interval evaluations and
provides the fixed baseline used by the progress chart. Resumed schedulers use
persisted `eval_metrics.jsonl` policy steps as their evaluation cursor. Missing
eval records remain due instead of being inferred from the checkpoint step.
Final evaluation reuses an interval result from the same policy rather than
generating an identical benchmark twice. Before a required final evaluation,
the persistent scheduler cancels speculative rollout requests that can no
longer be consumed, then loads and evaluates the final policy in isolation.
Set `eval.sampling.reasoning_effort` (or the per-environment sampling override)
to `minimal`, `low`, `medium`, or `high` for reasoning-model evaluations.
Verifier thread and math-process pools scale to the scheduler's real in-flight
request high-water mark, which is logged as `generation/executor_concurrency`.
When `max_pending_rollout_chunks` is set, that queue-derived capacity is a hard
in-flight bound; `oversampling_factor` does not multiply it a second time.
`max_inflight_rollouts` is also an exact request ceiling. These explicit bounds
may intentionally leave inference client routes idle rather than exceed the
configured memory budget.
Verifier rollout dispatch grows by at most one group or 10% of its in-flight
ceiling per five-second window, whichever is larger; normally completed work
refunds that admission so steady-state replacements remain immediate. Set
`orchestrator.tasks_per_minute` to a positive integer to add a global rollout
rate limit for sandbox-backed environments. The rate limit is supported only
by `wavelet.orchestrator.verifiers:generate_rollouts`, and cancelled work does
not refund admission into an immediate refill wave.
Rollout metrics decompose `off_policy/mean` and `off_policy/max` into
`off_policy/in_flight/*` (policy updates while generation was running) and
`off_policy/in_queue/*` (additional lag before the rollout was published).
Verifier attempts that return exceptions, explicit errors, missing rewards, or
untrainable trajectories are counted under `fate/errors/<type>` even though
they are excluded from training rows. Available verifier phase timings are
normalized to seconds under `time/rollout/<phase>/*`.
Long-running process schedulers emit a one-line policy, queue, and in-flight
status every `orchestrator.pipeline_status_interval_seconds` (30 seconds by
default).
Cached verifier environments and registered executors are torn down when the
inference scheduler closes. Integrated runs also close the inference engine and
verifier resources on both success and failure before finalizing trainer state.
Secret-valued config fields are redacted before resolved configuration reaches
run metadata, the state API, or W&B; token-count and tokenizer fields remain
visible for debugging.

Trainers consume queue batches in exact queue order. Every batch manifest must
agree with its queue step, optimizer step, chunk index, row count, and configured
policy-freshness window before any tokens are trained.
Online RL requires stochastic sampling with a positive temperature, and the data
boundary rejects non-finite advantages, rewards, policy log-probabilities, and
temperatures as well as non-positive temperatures before model execution.
Integrated generation also records the policy version loaded by its inference
engine in every rollout manifest.
Persistent verifier requests retain the policy version used at dispatch; mixed
async chunks are labeled with their oldest contributing policy, and incomplete
groups are never refilled from a newer policy.
Checkpoint resume reuses an existing stable rollout only after the same manifest
checks pass. A stable queue directory is immutable and cannot be overwritten by
a racing producer.
Queue receive, claim, and consume events retain the originating optimizer and
policy steps so policy lag remains traceable through the full lifecycle.
Temporary files used to merge small streaming chunks are deleted after all
trainer ranks finish reading them; retained consumed queue batches remain the
auditable rollout source.

RL loss normalization is optimizer-batch exact for variable-length examples.
DPPO (`rl`), cross entropy (`ce`), and reference-policy KL (`ref_kl`) each use
their own global denominator, so adding sparsely weighted CE or distillation
tokens does not dilute another component. Rollout rows opt into the auxiliary
components with trainable-token-aligned `ce_weight` and `ref_kl_weight` streams;
`ref_kl` also requires `teacher_logprobs`. `loss.type: custom` imports a custom
RL component through `loss.import_path` and forwards `loss.kwargs`.
When dataloader workers prevent deterministic look-ahead, the RL-only trainer
sums raw microbatch losses and applies the measured global token or sequence
denominator once at the optimizer boundary. Auxiliary components with one data
worker require a single microbatch per optimizer step. Multi-chunk streaming
steps remain RL-only because their final auxiliary denominators are not known
when the first chunk is backpropagated.
Built-in online distillation supports OPD (frozen-teacher reference KL), OPSD
(demonstration-conditioned self-distillation), and SFT on frozen-teacher
samples. See [docs/algorithms.md](docs/algorithms.md#online-distillation) for the
two-server setup and configuration contract.
If any distributed rank observes a non-finite loss, all ranks abort before
backward and accumulated gradients are cleared; the trainer never silently
skips a microbatch into a later optimizer update.

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

Training rollouts currently use full-distribution sampling (`top_p: 1`,
`top_k: -1`, `min_p: 0`, `min_tokens: 0`, and `repetition_penalty: 1`).
Wavelet rejects truncated, stop-suppressed, or penalty-adjusted training
sampling—including distribution controls hidden in `extra_body`—because the
trainer cannot yet replay those transforms when computing importance ratios.
Missing sampled-token
logprobs and pre-tokenized source streams with misaligned response-side values
are rejected instead of being trimmed or replaced with synthetic values.
Legitimate context-tail truncation keeps the aligned prefix. Evaluation sampling
is unaffected by this restriction.
Verifier advantages are computed inside each dispatched rollout group, so
duplicate dataset `example_id` values cannot merge otherwise independent GRPO
comparisons. The internal dispatch-group identity is retained through rollout
metadata, complete-group admission, and per-problem rollout metrics; interleaved
rows are summarized against their own group rather than their file position,
and extra trajectory branches do not count as additional rollouts.

Evaluation `avg@k`, `pass@k` (at least one correct), and `pass^k` (all correct)
metrics count every requested generation with unbiased combinatorial estimators;
failed or missing-reward attempts count as incorrect instead of disappearing from
the denominator. Rollouts whose verifier recorded an internal error are failed
attempts too, even though verifiers reports them with a zero reward.
Corresponding `eval/<env>/effective/...` metrics describe only successful
verifier responses, and `failed_rollouts` reports the failure rate. A fixed
`eval.sampling.seed` is offset per rollout (`seed + rollout_index`) so the k
generations of one example stay distinct. Evaluations always query the served
policy model name, so LoRA runs evaluate the trained adapter rather than the
base model.
The orchestrator writes these fixed-policy metrics to `eval_metrics.jsonl` and
its W&B run at the matching optimizer step. W&B rows use the logged `step`
field as their step metric rather than an explicit monotonic step, so async
evaluation results for an earlier optimizer step are not discarded, and a
resumed run continues the previous W&B run through the id persisted in
`<output_dir>/wandb_run_id.txt`.
Evaluation rollout requests are bounded by `eval.max_inflight_rollouts`
(default `64`) so a large evaluation set does not enqueue every generation at
once.
Online process-mode RL roles use W&B shared mode with that same run id: the
trainer is the primary writer and the orchestrator is the final writer, so
training, rollout, and evaluation metrics land in one run. Offline and disabled
W&B modes keep their existing local behavior and do not enable shared mode.
Before process-mode roles start, the launcher downloads remote model artifacts
once into the shared Hugging Face cache; local paths and the built-in debug
model skip this step.
Metrics logged without an optimizer step use W&B wall time as their chart axis.
Online primary writers also create a versioned `overview` workspace with curated
training, evaluation, stability, wall-time inference, performance, and resource panels; set
`monitor.wandb.create_overview: false` to disable that project-level view.
Training metrics that become NaN or infinite are stored as `null` in local
JSONL logs and omitted from W&B rows; the monitor warns once for each affected
metric key.
Trainer performance rows include an architecture-aware
`perf/model_flops_per_token` estimate for dense, grouped-query-attention, MLA,
and MoE models. On supported low-precision accelerators they also include
`perf/mfu`, expressed as a percentage of dense peak FLOPs across the trainer
world. The built-in peak table covers A100, H100/H200, B200/B300, GB200/GB300,
MI300X, and MI325X; unknown devices and float32 runs omit MFU instead of using
a misleading fallback peak.
RL trainer metrics include `entropy/mean`, `entropy/min`, and `entropy/max`
over loss-masked policy tokens. Entropy uses each token's rollout temperature
and is computed without materializing full-vocabulary logits when the chunked
LM head is enabled.
Trainer-side automatic Python garbage collection is disabled while training and
generation-1 collection runs on every rank at the same optimizer-step boundary,
every `gc.interval` steps (default 50), avoiding rank-local pause jitter. Set
`gc: null` to retain Python's automatic garbage collector instead.
Set `profiler.start_step` and `profiler.end_step` to capture an inclusive
optimizer-step range as a Chrome trace. The default path is
`<output_dir>/profiler/trace-<start>-<end>.json`; distributed runs add a rank
suffix so workers never overwrite one another. Set `profiler.trace_path` to
override the file location, and open the result in Perfetto or Chrome tracing.
Enable `memory_profiler` on a CUDA trainer to record allocator history and write
PyTorch memory-viz snapshots under `<output_dir>/memory/step-N/rank-N.pickle`.
`memory_profiler.interval` controls the optimizer-step cadence and
`max_entries` bounds retained allocator events; `output_dir` overrides the
snapshot root. Enabling this profiler on a CPU-only runtime fails immediately.
Standalone trainers use the same resolved step count for optimization and the
learning-rate scheduler; in particular, RL `max_steps: 0` remains evaluation-only
instead of falling back to an implicit training run. Without `max_steps`,
`epochs` derives the step count from the dataset record count and the global
`data.batch_size`; packed datasets and fake data change the rows per epoch and
therefore require an explicit `max_steps`.
An explicit `scheduler.decay_steps` (at least 1) is clamped so decay begins when
warmup ends rather than being dropped, the `sqrt` schedule holds the peak until
its final `decay_steps`, and the `cosine` floor honors `min_lr_factor` like the
other schedules. A resumed RL trainer measures the token count of its first
optimizer step directly, because restored dataloader state only applies once
iteration begins, and then returns to the static per-batch estimate.

Every config model rejects unknown or misspelled keys, so a typo such as
`optim.mu` fails during config loading instead of becoming an inert setting.
Supported optimizers are AdamW, Adam, SGD, stateless SignSGD (`sign_sgd`), and
the configured 8-bit Adam/AdamW variants; Muon is not accepted because Wavelet
has no Muon runtime. SignSGD uses the sign of each gradient and applies
decoupled weight decay without allocating momentum or variance state.
FSDP1 does not expose a configurable `reshard_after_forward` policy, so setting
that key to `false` is rejected instead of being accepted without affecting the
wrapper. Select `fsdp.impl: fsdp2` to control that policy directly.
Likewise, `fsdp.cp` and `fsdp.ep` values above 1 are rejected before GPU setup;
the current model stack only supports data and tensor parallel dimensions.
SFT sample generation has no trainer implementation, so a `generate` block is
also rejected; use the inference commands after training instead.
SFT packing accepts only the implemented `pad` and `cat` modes; the former
`stack` mode and its bucket settings are rejected.
Activation offloading uses the trainer's single CUDA stream; the unsupported
`use_streams` and `max_fwd_stash_size` controls are rejected rather than
silently ignored. Its `pin_memory` setting remains configurable.
Additional vLLM server options belong in `inference.vllm.extra_args`. Keys may
use snake case or kebab case; structured values are JSON-encoded and boolean
values become `--flag`/`--no-flag`. Wavelet rejects collisions with arguments
it manages, such as `model`, `port`, `tensor_parallel_size`, `enable_lora`, and
`logprobs_mode`.
`inference.vllm.enforce_eager` defaults to `false`, allowing vLLM to use its
hybrid eager/CUDA-graph execution path. Set it to `true` for deployments whose
GPU-memory constraints or model behavior require eager-only execution.
Process launches can add environment variables globally or per role:

```yaml
launcher:
  env_vars:
    common:
      TOKENIZERS_PARALLELISM: "false"
    inference:
      VLLM_LOGGING_LEVEL: INFO
    trainer:
      TORCH_LOGS: recompiles
    orchestrator:
      HTTPX_LOG_LEVEL: warning
```

Role-specific values override `common`. Inference values apply to every vLLM
server replica, trainer values apply to the `torchrun` parent, and orchestrator
values apply to `rl-inference`. Launcher-owned GPU placement and distributed
rank variables are rejected in this table; configure device placement through
the normal launcher fields. Values are treated as secrets and redacted from
serialized role configs and dry-run output.
Both `tool_call_parser` and `reasoning_parser` default to `auto`. Known model
families resolve to the matching vLLM parser, while unknown or non-reasoning
models omit the corresponding flag. Set either field explicitly to override
resolution, or to `null` to disable that parser.
Policy load, pause, and resume calls retry transient connection failures and
HTTP 5xx responses up to three times with exponential backoff. Control calls
use a 300-second per-attempt timeout, while policy loads use 720 seconds; a
successful response acknowledging the wrong policy step fails immediately.
Inference startup checks worker RPC liveness at `/liveness` and then
`/v1/models`; a responsive API process with stuck model workers, or a server
that does not list the configured base model or policy adapter, fails before
rollout work begins. The worker RPC timeout is configurable with
`inference.http.liveness_timeout_seconds`.
`rollouts_per_examples` or `learning_rate` is an error rather than a silently
ignored setting. Several further misconfigurations fail fast instead of
degrading silently: `ckpt`
settings such as `interval` together with the default `mode: disabled`, fused
LoRA kernels combined with `lora.dropout > 0`, RL `data.num_workers > 1`, RL
`data.pad_to_multiple_of` values that do not divide `data.seq_len`, duplicate
`inference.http.ports`, legacy aliases such as `optim.betas` that disagree with
their canonical fields, a process-mode `policy_transfer.export_every_steps`
larger than the freshness window (`min(max_async_level - 1,
max_off_policy_steps) + 1`, which would leave the rollout scheduler waiting for
an export the trainer cannot produce), and
`reward.mode: passthrough` for rollouts that vLLM generates without a custom
rollout function, and `launcher.mode: process` device groups where the trainer
and an inference replica, or two replicas, are pinned to the same CUDA device
(both rejected by `wavelet rl` at launch and reported by preflight; colocate
modes share devices by design).
With `policy_transfer.type: nccl`, each inference replica must expose exactly
`tensor_parallel_size * data_parallel_size` CUDA devices, because only vLLM
workers join the weight-broadcast group; the group size is the inference rank
count plus the trainer, independent of each replica's rank offset.

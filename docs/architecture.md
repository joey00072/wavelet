# Architecture

Wavelet is organized around explicit process and data boundaries. Entrypoints
parse configuration and choose a run mode; importable modules own behavior;
filesystem artifacts carry state between independently restartable processes.

## Package Map

| Area | Responsibility |
| --- | --- |
| `wavelet.configs` | Pydantic schemas, legacy input normalization, and cross-field validation |
| `wavelet.data` | Canonical SFT and RL loading, normalization, tokenization, packing, and collation |
| `wavelet.entrypoints` | Thin command adapters that load a subsystem's `main` function |
| `wavelet.orchestrator` | Example selection, rollout scheduling/sources, verifier environments, scoring, algorithms, metrics, and run state |
| `wavelet.transport` | Filesystem rollout queues and filesystem/NCCL policy transfer |
| `wavelet.inference` | Native and vLLM policy inference, HTTP clients, policy loading, and diagnostics |
| `wavelet.trainer` | Model/LoRA and distributed setup, RL/SFT training, losses, optimization, and checkpointing |
| `wavelet.distributed` | Compatibility imports for distributed APIs now owned by `wavelet.trainer.distributed` |
| `wavelet.kernels` | Optional performance kernels and narrowly scoped runtime patches |
| `wavelet.utils` | Configuration loading and path helpers |

Entrypoints contain no runtime behavior. A new command belongs in
`wavelet.entrypoints`, while its configuration, lifecycle, and reusable
behavior belong to the subsystem it invokes. Shared rollout scheduling lives in
`wavelet.orchestrator.scheduler`, verifier clients and evaluation in
`wavelet.orchestrator.envs`, inference serving in `wavelet.inference.server`,
and trainer behavior in `wavelet.trainer.trainer` and `wavelet.trainer.rl`.
Historical module paths are retained as thin compatibility aliases where user
code may still import them.
Trainer process-group initialization uses `dist_timeout_seconds` from the SFT
or RL config. It defaults to 1800 seconds and can be increased for slow or
multi-node rendezvous without changing code.
`model.compile: true` compiles each decoder block in place after LoRA and
optional kernel patches, and before distributed wrapping. This preserves
checkpoint parameter names and composes with gradient checkpointing;
`model.compile_fullgraph` opts those block compilations into full-graph mode.

## RL Process Flow

The combined `wavelet rl` command supervises the same roles that can be run
separately:

1. The inference role selects examples, loads the required policy, generates
   rollouts, scores them, assigns advantages, and publishes stable queue items.
2. The trainer claims a stable queue item, validates and tokenizes its rows,
   computes an RL loss, performs an optimizer step, and exports a policy.
3. The next inference step observes the exported policy subject to the
   configured off-policy window.

`max_off_policy_steps` is always a hard freshness ceiling. Setting it to zero
requires the policy for the current optimizer step even if `max_async_level` is
larger; async capacity never silently widens the configured freshness window.
The trainer enforces this through `required_policy_step(rollout_step)`, and the
persistent verifier scheduler admits, ages, and cancels groups against that same
threshold for the rollout step it is building rather than against the policy it
currently has loaded. A group can therefore never be published with a policy
step the trainer would reject. Groups whose rollouts keep failing are retried a
bounded number of times and then dropped as rejected instead of spinning forever.
Groups interrupted by a policy swap are counted as cancelled work rather than
reward-filter rejections, so they never consume the zero-advantage retry budget.
Buffered ready groups ship before freshly completed ones, so older admissible
groups are not starved until they expire.

Process-style launchers additionally require every freshness window to contain
an export: `policy_transfer.export_every_steps` may not exceed the admissible lag
plus one, otherwise the scheduler would wait for a policy the trainer cannot
produce until it receives the very rollouts the scheduler is waiting to build.

The integrated launcher follows the same contract: it only pipelines rollout
generation for step `S+1` on policy `S` when the freshness window admits a lag of
one, waits only for policy steps the trainer actually exports (respecting
`policy_transfer.export_every_steps`), labels an engine that has not loaded any
policy as step 0, and reuses an existing stable rollout batch on resume.

Queue directories and stable markers are the synchronization contract. Run
state and queue events provide observability; process-local memory is never the
source of truth for a completed batch or policy.
Preflight reports optimizer batches explicitly as groups times rollouts and the
number of transport chunks. A non-divisible final chunk contains only the
remaining groups; chunking never rounds the optimizer batch upward.

## Inference Scheduling

`wavelet.orchestrator.scheduler` owns the scheduling strategies behind one
explicit source/publish-mode boundary:

- Prefetch scheduling publishes complete optimizer-step batches in order.
- Native chunk scheduling permits chunks to finish out of order while tracking
  the contiguous published frontier.
- Rolling verifier scheduling keeps verifier groups in flight and bounds their
  off-policy age.

Each scheduler keeps submitted, pending, completed, published, and policy-load
state explicit. Scheduling code should preserve those states and update the run
state server at the transition that actually occurred.

## Data Boundaries

`wavelet.data.sft` owns source loading, message normalization, tokenization,
collation, and SFT datasets. `wavelet.data.rl` owns RL records, serialization,
packing, collation, and datasets. Historical fine-grained imports remain
supported; see the [data pipeline guide](data_pipeline.md). Serialized
`RLExample` payloads are the boundary between rollout generation, HTTP
inference, queues, diagnostics, and training.

## Policy Artifacts

`wavelet.transport.policy` owns filesystem and NCCL policy transfer, while
`wavelet.transport.queue` owns queue artifacts and lifecycle events.
Filesystem policy exports use a temporary directory followed by an atomic
rename and stable marker. Metadata is written beside the model or adapter. NCCL
transfer uses the same metadata and readiness concepts, but broadcasts named
tensors after inference workers enter the update collective.
Inference loads LoRA adapters directly from the immutable published directory;
it does not make a second tmpfs copy of every policy.
Policy receive events reuse the tensor byte count recorded in `policy.json`;
they do not walk or reread the artifact to reconstruct diagnostic metadata.
Checkpoint resume removes policy versions beyond the restored step for both
filesystem and NCCL transports, discards rollout queue batches whose optimizer
step lies beyond the restored step (a manifest carries only a policy step, so a
stale batch from the old run's identically numbered policy would otherwise
validate), and reuses an exact complete snapshot when present. Ordinary exports never overwrite a stable policy directory, and every
rank waits for the main rank to reset the export directory before writing.
Every adapter snapshot is loaded with a one-shot in-place request so vLLM's
adapter cache, which is keyed by adapter id, swaps in the new weights; the
request kept for generation has the flag cleared so vLLM does not reread the
adapter during later scheduler work. The HTTP server registers a refreshed
adapter request only after vLLM accepted the load, so a failed load leaves the
previous adapter serving.

HTTP policy refreshes are transactions across all inference replicas. The
rollout scheduler first blocks new submissions, cancels in-flight requests whose
policy can no longer be admitted for the upcoming rollout step, and drains the
requests that remain. LoRA adapters then use vLLM's in-place load directly; a second server
pause would only repeat the scheduler drain. Full-model and collective updates
pause generation without clearing the version-salted prefix cache, update every
replica, and resume even when loading fails. The offline engine, whose adapter
id is stable across snapshots, resets the prefix cache after each in-place
adapter reload and forwards the client's `cache_salt`. Never replace adapter or
model weights while a request is decoding.

Lightweight LoRA exports from FSDP gather adapter shards over the `dp_shard_cp`
group, so every HSDP replica reconstructs the full adapter without duplicate
shards, and the snapshot includes `lora.modules_to_save` copies alongside the
LoRA tensors. Only the intended distributed rank writes metadata, stable
markers, queue
events, and final directories. Barriers protect visibility across trainer
ranks; they do not replace stable markers between trainer and inference
processes.
Rollout manifests and claim/consumed records are required queue state. Their
write failures stop the run; only duplicate diagnostic events and traces are
best-effort.

## Extension Points

- Algorithms: use a named built-in algorithm or load a user-owned Python file;
  see [RL algorithms](algorithms.md).
- Rollouts: configure a custom rollout function at the orchestrator boundary.
- Rewards: add a focused scorer and its config validation rather than branching
  inside launch code.
- Inference engines: implement the existing setup, annotate/generate,
  policy-load, sleep/wake, and close lifecycle.
- Diagnostics: expose a deterministic report before adding a long-running path.

## Maintainability Rules

- Normalize compatibility inputs at the config boundary without mutating the
  caller's dictionary.
- Keep one owner for serialization, packing, scoring, and policy publication.
- Use small helpers for transitions that update several related state fields.
- Preserve strict token alignment between `loss_mask` and trainable-only value
  streams.
- Treat optional runtime patches and kernels as isolated adapters; ordinary
  trainer and inference code should remain understandable without them.
- Add focused negative-path tests whenever a boundary rejects malformed state.

Run the repository maintainability check alongside ordinary lint when changing
control flow:

```bash
uvx ruff check wavelet \
  --select C901,PLR0911,PLR0912,PLR0915
```

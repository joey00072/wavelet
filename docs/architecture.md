# Architecture

Wavelet is organized around explicit process and data boundaries. Entrypoints
parse configuration and choose a run mode; importable modules own behavior;
filesystem artifacts carry state between independently restartable processes.

## Package Map

| Area | Responsibility |
| --- | --- |
| `wavelet.configs` | Pydantic schemas, legacy input normalization, and cross-field validation |
| `wavelet.data` | Source loading, message normalization, tokenization, packing, and collation |
| `wavelet.entrypoints` | Thin command adapters and process lifecycle |
| `wavelet.orchestrator` | Example selection, rollout scheduling, scoring, algorithms, queues, metrics, and run state |
| `wavelet.inference` | Native and vLLM policy inference, HTTP clients, policy loading, and diagnostics |
| `wavelet.trainer` | Model setup, RL/SFT training, loss calculation, checkpointing, and policy export |
| `wavelet.distributed` | World and device-mesh construction plus distributed collectives |
| `wavelet.kernels` | Optional performance kernels and narrowly scoped runtime patches |
| `wavelet.utils` | Configuration loading, monitoring, paths, and activation offloading |

Entrypoints should not become alternate implementations of these modules. A
new command belongs in `wavelet.entrypoints`; its reusable behavior belongs in
the subsystem it invokes.

## RL Process Flow

The combined `wavelet rl` command supervises the same roles that can be run
separately:

1. The inference role selects examples, loads the required policy, generates
   rollouts, scores them, assigns advantages, and publishes stable queue items.
2. The trainer claims a stable queue item, validates and tokenizes its rows,
   computes an RL loss, performs an optimizer step, and exports a policy.
3. The next inference step observes the exported policy subject to the
   configured off-policy window.

Queue directories and stable markers are the synchronization contract. Run
state and queue events provide observability; process-local memory is never the
source of truth for a completed batch or policy.

## Inference Scheduling

`wavelet.entrypoints.rl_inference` exposes three scheduling strategies with the
same policy-loading contract:

- Prefetch scheduling publishes complete optimizer-step batches in order.
- Native chunk scheduling permits chunks to finish out of order while tracking
  the contiguous published frontier.
- Rolling verifier scheduling keeps verifier groups in flight and bounds their
  off-policy age.

Each scheduler keeps submitted, pending, completed, published, and policy-load
state explicit. Scheduling code should preserve those states and update the run
state server at the transition that actually occurred.

## Data Boundaries

SFT and RL share raw-source loading and message normalization. RL-specific
types, collation, and packing live in focused modules; see the
[data pipeline guide](data_pipeline.md). Serialized `RLExample` payloads are the
boundary between rollout generation, HTTP inference, queues, diagnostics, and
training.

## Policy Artifacts

Filesystem policy exports use a temporary directory followed by an atomic
rename and stable marker. Metadata is written beside the model or adapter. NCCL
transfer uses the same metadata and readiness concepts, but broadcasts named
tensors after inference workers enter the update collective.

Only the intended distributed rank writes metadata, stable markers, queue
events, and final directories. Barriers protect visibility across trainer
ranks; they do not replace stable markers between trainer and inference
processes.

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

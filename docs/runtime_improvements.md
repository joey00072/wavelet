# Runtime improvements

Adaptive rollout concurrency now tapers its growth multiplier as inference KV
usage approaches the configured soft threshold. The controller reports
`generation/concurrency/growth_multiplier` with its existing concurrency gauges.
The multiplier tapers continuously from the configured factor at zero usage to
one at the soft KV threshold; `growth_kv_cache_usage` remains a compatibility
configuration field and does not freeze growth before that threshold.

Optimizer state offload keeps reusable CPU tensors per parameter and state key,
including pinned tensors when CUDA is available. This bounds host allocation
churn across optimizer steps while preserving state-dict movement hooks.

The optimizer module includes Muon for matrix parameters, with momentum state
and per-matrix Newton–Schulz orthogonalization; vectors, embeddings, heads,
and scalars use a standard AdamW fallback. DTensor matrix updates reconstruct
the logical full matrix for orthogonalization and redistribute the update to
the original placements; this is correctness focused and can be expensive.
Packed projection partitions and Dion's specialized mesh optimizations remain
unsupported. The RL loss module includes opt-in IcePop ratio masking. Both
features require the corresponding config schema entries documented by the
configuration models.

Select Muon with `optim.type: muon`; `optim.muon_momentum` defaults to `0.95`.
The optimizer keeps its AdamW parameter groups in the checkpoint so resuming
preserves matrix versus fallback updates. Two-rank CPU Gloo tests compare
sharded matrix updates with the unsharded rule, including checkpoint resume.

Select IcePop with `loss.type: icepop`. `loss.ratio_low` and `loss.ratio_high`
(defaults `0.2` and `5.0`) bound the accepted behavior-to-current-policy ratio.
These options are opt-in and do not change the default DPPO objective.

Streaming RL performance metrics accumulate across every rollout chunk in an
optimizer update. `perf/train_seconds` includes training all of its chunks;
`perf/step_seconds` and `time/step` include rollout waiting, loading, training,
and policy export. The first step can therefore include startup evaluation
while the trainer waits for data. `perf/throughput` uses model tokens divided
by training time, while `perf/step_tokens_per_second` uses the full elapsed
step time. Use the latter when comparing end-to-end pipeline throughput.
`perf/rollout_wait_seconds`, `perf/rollout_load_seconds`, and
`perf/policy_export_seconds` expose the non-training phases separately.

Per-environment `train/<env>/advantage/{mean,min,max,std}` now summarize
individual scalar rollout advantages, matching `advantage/all/*`. Averaging
inside each GRPO group first made every centered group zero and hid its
training signal. Reward grouping and training calculations are unchanged.

Async distributed checkpoints use a dedicated Gloo process group for their CPU
staging and storage collectives, while training keeps its configured backend.
The group is created lazily by all ranks and reused for subsequent saves and
loads. This prevents final checkpoint failures with an NCCL-only training group
and isolates background checkpoint communication from training collectives.
A CUDA regression can be run with:

```bash
uv run torchrun --standalone --nproc-per-node=2 tests/checkpoint_gpu_worker.py \
  outputs/checkpoint-nccl-smoke
```

Use an empty output directory. The regression saves during NCCL/FSDP training,
performs another update while the save is pending, and checks restored model
and optimizer tensors plus a subsequent stable save.

FSDP1 checkpoints use PyTorch's shard-aware blocking CPU staging because
`DefaultStager` cannot copy legacy ShardedTensor values. Snapshot staging
completes before the next optimizer update; disk upload remains asynchronous.
FSDP2 and unsharded models keep the default stager.

Lightweight LoRA exports remove both FSDP and activation-checkpoint wrapper
segments from tensor names. The manual shard gather uses `named_parameters`,
which bypasses PyTorch's usual checkpoint-wrapper `state_dict` hooks. Leaving
`_checkpoint_wrapped_module` in an adapter key prevents serving layers from
matching the trained tensor, even when the server acknowledges the policy
load. A regression compares exported names and values before and after wrapping
the same model block. Start a clean run after applying this fix: affected older
runs generated rollouts without the intended adapter updates.

Preflight reports both configured freshness limits and their effective maximum
policy lag. Async level includes the current update: level 2 with an off-policy
limit of 8 still permits only one step of lag. Diagnostics and admission use the
same helper, preventing the displayed limit from drifting from enforcement.
## Async policy retention and FSDP1 checkpoint mesh

Filesystem policy `keep_last` is a minimum: publication retains at least
`ceil(effective_max_policy_lag / export_every_steps) + 2` snapshots. This covers
a policy selected before request draining while the trainer consumes previously
published work. Keeping only two could delete the selected version and leave
the loader waiting indefinitely. Preflight reports the effective retention.

Plain FSDP1 data parallelism uses the shard process group directly. Passing a
flattened child of a one-dimensional mesh made checkpoint export take the
tensor-parallel path and fail because no second parent dimension existed.
Tensor-parallel and hybrid-shard configurations retain their device meshes.

The distributed regression exercises the actual wrapping path, asynchronous
checkpoint creation during another optimizer update, and exact restoration:

```bash
uv run torchrun --standalone --nproc-per-node=8 tests/checkpoint_gpu_worker.py \
  outputs/checkpoint_smoke
```

Use a fresh output directory and an allocated GPU worker for this check.

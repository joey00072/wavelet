# Wavelet feature gap TODO (vs prime-rl)

Handoff document for contributors. It lists what Wavelet is missing relative to
`ref/prime-rl`, where each feature should live in Wavelet, where to read the
reference implementation, and how to know a task is done.

Date of survey: 2026-09-04. Line numbers in reference paths drift; search for
the named function or class if a line no longer matches.

Implementation status (2026-09-05): the required Tier 0 through Tier 5 work in
this survey has landed locally, including FSDP2, Hugging Face Qwen3-MoE/GPT-OSS
expert parallelism, ring context parallelism, sampling-mask replay, and the
listed correctness, observability, checkpoint, inference, CLI, CI, and tooling
items. The descriptions below are retained as the original acceptance criteria;
their “Missing” wording is historical rather than a statement about the current
tree. The remaining owner decision is the license choice in T1-8. The following
items were explicitly optional and remain intentionally unscheduled:

- T4-7 ZMQ batch transport
- T5-6 SLURM job generation (Docker, model pre-download, and Ray documentation
  are implemented)
- T5-9 extended dashboard depth

Current verification: `uv run pytest tests -q` reports 1083 passed and 5
hardware-dependent skips; Ruff lint, Ruff formatting, `git diff --check`, and
package builds pass. The EP-specific two-process test covers output and gradient
parity, FSDP2 wrapping, DCP save/load, meta-device loading from split Qwen
safetensors, and comparison against Hugging Face logits.

Out of scope on purpose: multi-LoRA (multiple adapters per run), NIXL/RDMA
weight transfer, prefill/decode disaggregation, Kubernetes Helm charts, the
Prime platform monitor. Do not build these.

---

## 0. Ground rules (read before touching anything)

1. Read `AGENTS.md` first. It overrides this doc.
2. `ref/` is read-only reference. Never edit, import from, or run tooling on
   it. `ref/prime-rl` is a symlink; `grep -r` needs a trailing slash
   (`grep -rn foo ref/prime-rl/`).
3. Run everything with `uv run`. Before committing:
   `uvx ruff check wavelet tests`, `uvx ruff format --check wavelet tests`,
   `uv run pytest tests -q`, `git diff --check`.
4. Every bug fix or feature ships with a test under `tests/` and a docs update
   in the same change (`README.md`, `docs/architecture.md`,
   `docs/data_pipeline.md`, `docs/algorithms.md` as appropriate). Add a row to
   `rl_audit_progress.md` if you fix a correctness bug.
5. Fail fast. A config field that validates but does nothing is a bug. Either
   implement it or delete it and make the loader reject it.
6. Do not copy prime-rl code verbatim. Read it, understand the invariant it
   protects, then implement it in Wavelet's style and abstractions.
7. Never commit run artifacts (`outputs/`, `*.jsonl` traces, checkpoints) or
   `.env` contents.
8. One task per branch and PR. Keep PRs reviewable (under ~600 lines of
   non-test diff where possible). Tier 2+ items may need a short design note
   in the PR description before code.

---

## 1. Architecture primer

### 1.1 Wavelet roles and where they live

Wavelet is one Python package with a single Pydantic config tree (`RLConfig`
for RL, `SFTConfig` for SFT). One launcher spawns three roles.

| Role | Entry | Core modules |
|---|---|---|
| Launcher | `wavelet/entrypoints/rl_launcher.py`, `wavelet/orchestrator/runtime.py`, `wavelet/orchestrator/launcher.py` | Derives per-role configs from `RLConfig`, assigns CUDA devices (`orchestrator/placement.py`), spawns roles as subprocesses (`process`), in one process (`integrated`), or sharing GPUs (`colocate`, `colocate_sleep`). Optional Ray backend. |
| Inference | `wavelet/entrypoints/rl_inference.py`, `wavelet/inference/server.py` (OpenAI-compatible vLLM server with admin routes), `wavelet/inference/native_server.py`, `wavelet/inference/engine.py` | Serves rollouts. Admin routes: `/pause`, `/resume`, `/load_policy`, `/init_broadcaster`, `/sleep`, `/wake`, `/health`. Weight receive in `wavelet/transport/policy.py` (filesystem or NCCL). |
| Orchestrator | `wavelet/entrypoints/rl_orchestrator.py`, `wavelet/orchestrator/scheduler.py` (main loop, `VerifierRolloutScheduler`, `IntegratedRolloutScheduler`), `wavelet/orchestrator/envs.py` (verifiers env execution, eval), `wavelet/orchestrator/advantage.py`, `algorithms.py`, `custom_algorithms.py`, `schedule.py` (async policy), `reward.py` | Samples examples, runs rollouts through verifiers in-process, computes advantages, publishes batches to the filesystem queue (`wavelet/transport/queue.py`), decides when to load a new policy. |
| Trainer | `wavelet/entrypoints/rl_trainer.py`, `wavelet/trainer/rl.py` (RL step), `wavelet/trainer/trainer.py` (shared base, SFT loop), `wavelet/trainer/model.py` (load, LoRA, FSDP1 wrap, TP), `wavelet/trainer/losses.py` (DPPO loss, chunked LM head), `wavelet/trainer/optim.py` (optimizers, schedulers, activation offload), `wavelet/trainer/ckpt.py` (DCP checkpoints), `wavelet/trainer/distributed.py` (device mesh) | Consumes queue batches, tokenizes and packs (`wavelet/data/rl.py`), trains, exports policies (`wavelet/transport/policy.py`). |
| Shared | `wavelet/configs/config.py`, `wavelet/configs/rl_config.py`, `wavelet/configs/sft.py`, `wavelet/monitor.py` (metrics, W&B, events), `wavelet/debug.py` (preflight CLI), `wavelet/orchestrator/state_server.py` + `webui/` (dashboard) | |

Read `docs/architecture.md` and `docs/data_pipeline.md` for the data flow.

### 1.2 prime-rl map (reference layout)

Code root: `ref/prime-rl/src/prime_rl/`. Configs are a separate package:
`ref/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/`
(`trainer.py`, `orchestrator.py`, `inference.py`, `algorithm.py`, `sft.py`,
`shared.py`, `evals.py`). Docs: `ref/prime-rl/docs/*.md`.

| Wavelet module | prime-rl equivalent |
|---|---|
| `wavelet/trainer/rl.py` | `trainer/rl/train.py` |
| `wavelet/trainer/losses.py` | `trainer/rl/loss.py`, `trainer/models/layers/lm_head.py` |
| `wavelet/trainer/model.py` | `trainer/model.py`, `trainer/lora.py`, `trainer/models/*` |
| `wavelet/trainer/distributed.py` | `trainer/parallel_dims.py`, `trainer/world.py` |
| `wavelet/trainer/optim.py` | `trainer/optim/__init__.py`, `trainer/optim/state_offload.py`, `trainer/scheduler.py`, `utils/act_offloading.py` |
| `wavelet/trainer/ckpt.py` | `trainer/ckpt.py` |
| `wavelet/trainer/trainer.py` (SFT) | `trainer/sft/train.py` |
| `wavelet/data/sft.py` | `trainer/sft/data.py` |
| `wavelet/data/rl.py` | `trainer/rl/data.py`, `trainer/batch.py`, `orchestrator/packing.py` |
| `wavelet/orchestrator/scheduler.py` | `orchestrator/orchestrator.py`, `dispatcher.py`, `train_source.py`, `train_sink.py`, `watcher.py` |
| `wavelet/orchestrator/envs.py` | `orchestrator/envs.py`, `clients.py`, `eval_source.py`, `eval_sink.py` |
| `wavelet/orchestrator/advantage.py`, `algorithms.py` | `orchestrator/algo/*.py` |
| `wavelet/orchestrator/schedule.py` | `orchestrator/watcher.py`, `train_sink.py` (staleness) |
| `wavelet/inference/server.py` | `inference/vllm/server.py`, `inference/vllm/serving_tokens.py`, `inference/patches.py` |
| `wavelet/transport/policy.py` | `transports/weights/{base,filesystem,nccl}.py`, `inference/vllm/worker/*` |
| `wavelet/transport/queue.py` | `transports/batch/{filesystem,zmq,types}.py` |
| `wavelet/orchestrator/runtime.py`, `launcher.py` | `entrypoints/rl.py`, `utils/process.py`, `utils/pathing.py` |
| `wavelet/monitor.py` | `monitors/*`, `utils/logger.py` |
| `wavelet/orchestrator/state_server.py`, `webui/` | `dashboard/server.py` |

### 1.3 Key differences to keep in mind

- prime-rl runs verifiers environments as separate env-server processes over
  ZMQ. Wavelet runs them in-process in the orchestrator. Do not port the
  env-server model; adapt features to in-process execution.
- prime-rl uses FSDP2 (`fully_shard`) with its own model implementations for
  MoE. Wavelet uses FSDP1 with Hugging Face models. Several Tier 2 items
  depend on moving to FSDP2 first.
- prime-rl's orchestrator packs micro-batches and ships them to trainer ranks.
  Wavelet ships rows (JSONL) and the trainer tokenizes/packs. Keep that split
  unless a task says otherwise.
- Wavelet has strict configs (`extra="forbid"`) and a single config tree.
  New knobs go in `wavelet/configs/config.py` and must have validators.

---

## 2. How to work a task

1. Read the Wavelet module(s) named in the task and the prime-rl reference.
2. Write down the invariant (one paragraph) in the PR description.
3. Add the config field(s) with validators and defaults that preserve current
   behaviour.
4. Implement. Prefer small pure functions that can be unit-tested on CPU.
5. Tests: unit test on CPU always. If GPU is required, guard with
   `pytest.importorskip` / `torch.cuda.is_available()` skip and say so.
6. Docs: update the relevant `docs/*.md` and `README.md` section. Add an
   example YAML under `examples/` if the feature needs one.
7. Run the full check list from section 0.

Size legend: S = under a day, M = 1 to 3 days, L = about a week,
XL = multi-week and needs a design review first.

---

## 3. Tier 0: inert config fields (do these first)

Each of these validates today but has no effect. Pick one of: implement, or
delete the field so the strict loader rejects it. Every item is S unless noted.

| ID | Field | Wavelet location | Fix | Reference |
|---|---|---|---|---|
| T0-1 | `optim.type: muon`, `optim.mu` | `wavelet/configs/config.py` (`OptimizerConfig`), `wavelet/trainer/optim.py` `setup_optimizer` raises "Unsupported optimizer type" | Implement Muon (M). prime-rl uses the `dion` package's `Muon`; it excludes `lm_head` and embeddings from Muon and puts them in AdamW. Under FSDP the Muon update needs the shard mesh. If not implementing now, remove `"muon"` from the Literal. | `trainer/optim/__init__.py` `_create_muon_optimizer`, `configs/trainer.py` `MuonConfig` |
| T0-2 | `fsdp.reshard_after_forward` | `wavelet/configs/config.py` (`FSDPConfig`), never read in `wavelet/trainer/model.py` | Wire into the FSDP wrap. FSDP1 has no direct flag; map to `BACKWARD_PRE` vs `BACKWARD_POST` prefetch or defer to T2-1 (FSDP2) where it is a first-class argument. Until then, delete the field. | `trainer/model.py` `fully_shard(..., reshard_after_forward=...)` |
| T0-3 | `fsdp.cp > 1`, `fsdp.ep > 1` | Validators in `wavelet/configs/config.py` accept them; `wavelet/trainer/model.py` `maybe_wrap_fsdp` raises `NotImplementedError` at wrap time | Move the rejection into a config `model_validator` so it fails before any GPU work. Remove once T2-2 / T2-3 land. | n/a |
| T0-4 | SFT `val` block (`SFTValConfig`: `interval`, `eval_on_start`, `data`) | `wavelet/configs/config.py`, consumed nowhere in `wavelet/trainer/trainer.py` | Implement validation loss (M). Build a second dataloader from `val.data`, run a no-grad loop every `interval` steps and optionally at step 0, log `val/loss` through `RunMonitor`. | `trainer/sft/train.py` (search `val_dataloader`, `eval_on_start`), `configs/sft.py` `ValConfig` |
| T0-5 | SFT `generate` block (`GenerateConfig`) | `wavelet/configs/config.py`, no consumer | Either implement a periodic sample generation that logs to the samples table, or delete. Recommend delete. | none |
| T0-6 | `deployment.num_gpus` | `wavelet/configs/config.py` `SingleNodeDeploymentConfig`, no consumer | Either make `wavelet sft` self-launch `torchrun --nproc-per-node N` when `num_gpus > 1`, or delete. | `entrypoints/sft.py` (search `torchrun`) |
| T0-7 | `log.json_console`, `log.json_file` | `wavelet/configs/config.py` `LogConfig`; `wavelet/monitor.py` `setup_logger` is `logging.basicConfig` | Implement a JSON formatter for the file handler and optional console (S). | `utils/logger.py` |
| T0-8 | `data.pack_function: stack` plus `stack_bucket_*` | `wavelet/configs/config.py` `DataConfig`; `wavelet/data/sft.py` raises `NotImplementedError` | Delete the option and its two knobs unless someone owns bucketed packing. | none |
| T0-9 | Activation offload `use_streams`, `max_fwd_stash_size` | `wavelet/configs/config.py`; `wavelet/trainer/optim.py` builds the offload context without passing them | Pass them through to the offload implementation, or delete. | `utils/act_offloading.py`, `configs/trainer.py` `ActivationOffloadConfig` |

Test pattern for all of these: a config-level test in
`tests/test_fix_runtime_config.py` that either asserts the feature has an
observable effect or asserts the loader rejects the removed key.

---

## 4. Tier 1: cheap correctness and observability wins

### T1-1 Entropy metric (S)
- Missing: per-token policy entropy. `ChunkedLmHeadOutput.entropy` in
  `wavelet/trainer/losses.py` is hard-coded `None`. This is the main
  mode-collapse signal and it is absent from all trainer logs.
- Where: `wavelet/trainer/losses.py` (`ChunkedLogprobLmHead`), surface in
  `wavelet/trainer/rl.py` metrics as `entropy/mean`, `entropy/min`,
  `entropy/max` over loss-masked tokens.
- Reference: `trainer/rl/loss.py` `compute_entropy`,
  `trainer/models/layers/lm_head.py` (chunked logsumexp minus expected logit).
- Note: compute inside the chunk loop so full logits are never materialised.
  Respect per-token temperature the same way logprobs do.
- Test: CPU test comparing chunked entropy to a naive
  `torch.distributions.Categorical(logits).entropy()` on a tiny vocab.

### T1-2 vLLM argument passthrough (S)
- Missing: `RLVLLMConfig` in `wavelet/configs/config.py` is a closed set.
  Users cannot set `max_num_seqs`, `seed`, `rope_scaling`, `kv_cache_dtype`,
  `enable_prefix_caching`, `compilation_config` without code changes.
- Where: `wavelet/configs/config.py` (`RLVLLMConfig`),
  `wavelet/inference/server.py` where CLI args are assembled.
- Reference: `configs/inference.py` (`extra="allow"`, kebab/snake
  normalisation, JSON values), `docs/inference.md` "passthrough" section.
- Design: keep strictness for known fields. Add an explicit
  `extra_args: dict[str, Any]` that is appended as `--key value` (JSON-encode
  dict/list values). Reject keys that collide with fields Wavelet manages
  (`model`, `port`, `tensor_parallel_size`, `enable_lora`, `logprobs_mode`).
- Test: argv assembly test; collision rejection test.

### T1-3 `enforce_eager` default (S)
- Wavelet defaults `inference.vllm.enforce_eager` to `True`, disabling CUDA
  graphs. prime-rl defaults to `False`.
- Where: `wavelet/configs/config.py`. Flip the default, keep it `True` in
  `colocate*` modes if memory headroom requires it (check
  `examples/*/rl_colocate*.yaml`). Document in `README.md`.

### T1-4 Admin-call robustness (S)
- Missing: `/pause`, `/load_policy`, `/resume` calls in
  `wavelet/inference/engine.py` (`HTTPPolicyInferenceEngine`) are a single
  `urllib` request with one flat timeout and no retry.
- Reference: `orchestrator/clients.py` (tenacity retry on 5xx and transport
  errors, `ADMIN_TIMEOUT_S`, `UPDATE_WEIGHTS_TIMEOUT_S`).
- Design: bounded retries with backoff for connection errors and 5xx; a
  longer per-attempt timeout for load than for pause/resume. Never retry a
  load that returned a different `policy_step` than requested.
- Test: fake HTTP server or monkeypatched opener that fails twice then
  succeeds.

### T1-5 Model listing check at startup (S)
- Missing: orchestrator only polls `/health`. Add a `/v1/models` check that
  the served model id matches `model.name` (or the adapter name).
- Where: `wavelet/orchestrator/runtime.py` `_wait_for_vllm_http_server`,
  `wavelet/inference/engine.py`.
- Reference: `orchestrator/clients.py` (search `models.list`).

### T1-6 SFT observability (S)
- Missing: SFT logs no throughput and no per-source progress. Counters exist
  in `wavelet/data/_stateful.py` (`stats()`) and are checkpointed but never
  read by `wavelet/trainer/trainer.py`.
- Add: `perf/tokens_per_second`, `perf/peak_memory_gib`,
  `progress/<source>/ratio_samples`, `progress/<source>/ratio_tokens`,
  `progress/epoch`.
- Reference: `trainer/sft/train.py` (search `progress/`), `trainer/perf.py`.

### T1-7 Non-finite metric sanitisation (S)
- Where: `wavelet/monitor.py` `RunMonitor.log`. Replace NaN/inf with `None`
  in JSONL and skip them in W&B rows, log one warning per key.
- Reference: `monitors/file/monitor.py`, `utils/utils.py` (search `isfinite`).

### T1-8 Cosmetic hygiene (S each)
- `pyproject.toml`: add `[tool.ruff]` matching what `uvx ruff` is run with,
  add `.pre-commit-config.yaml`.
- Add `LICENSE` (ask the repo owner which one).
- Add pytest markers (`gpu`, `slow`, `integration`) and
  `--strict-markers`; tag the existing GPU-only tests. Reference:
  `ref/prime-rl/pyproject.toml` `[tool.pytest.ini_options]`,
  `ref/prime-rl/tests/conftest.py`.

---

## 5. Tier 2: trainer scaling

### T2-1 FSDP2 migration (XL, design review first)
- Missing: Wavelet wraps with FSDP1 (`FullyShardedDataParallel`,
  `transformer_auto_wrap_policy`, `use_orig_params`) in
  `wavelet/trainer/model.py` `maybe_wrap_fsdp`. Every rank loads full
  weights on CPU before wrapping (`meta_device_init` only sets
  `low_cpu_mem_usage`).
- Target: `torch.distributed.fsdp.fully_shard` per transformer block with
  `MixedPrecisionPolicy(param_dtype, reduce_dtype)`, `reshard_after_forward`
  honoured (closes T0-2), meta-device init, then load weights with DCP from
  the HF safetensors directly into shards.
- Reference: `trainer/model.py` (`fully_shard` calls, `load_dcp_from_hf`,
  `MixedPrecisionPolicy`), `trainer/parallel_dims.py`, `docs/scaling.md`.
- Wavelet touch points: `wavelet/trainer/model.py` (wrap, state dict
  gather for export, LoRA gather in `save_lora_adapter_snapshot_from_fsdp`),
  `wavelet/trainer/ckpt.py` (DCP state dict API changes),
  `wavelet/trainer/trainer.py` `clip_grad_norm_` (FSDP1 method call must
  become `torch.nn.utils.clip_grad_norm_` on DTensors),
  `wavelet/trainer/optim.py` (activation offload hooks),
  `wavelet/transport/policy.py` (full state dict export path),
  `wavelet/trainer/distributed.py` (mesh names `dp_shard_cp`, `hsdp`).
- Keep working: HSDP (`dp_replicate`), TP via HF `tp_plan`, LoRA lightweight
  export, QLoRA, async DCP checkpointing, `colocate_sleep` CPU move.
- Tests: existing `tests/test_trainer_lora.py`, `tests/test_rl_trainer.py`,
  `tests/test_fix_export_ckpt.py` must pass. Add a 2-process gloo test for
  wrap plus DCP save/load round-trip on a tiny model.
- Deliver behind `fsdp.impl: fsdp1 | fsdp2` for one release, default `fsdp1`,
  then flip.

### T2-2 Expert parallelism for MoE (XL, depends on T2-1)
- Missing: the mesh in `wavelet/trainer/distributed.py` already carves `ep`
  from `dp_shard * cp`, but `maybe_wrap_fsdp` raises. No MoE metrics, no
  router freeze, no fp32 router.
- Reference: `trainer/parallel_dims.py` (ep in mesh),
  `trainer/moe_runtime.py`, `trainer/distributed/*.py`
  (`ExpertWeightParallel`), `trainer/models/layers/moe.py` (grouped GEMM,
  load-balance stats `max_vio`), `trainer/model.py` (`freeze_moe_router`,
  `moe_router_dtype`), `README.md` model table.
- Scope for a first PR (M): `model.freeze_moe_router: bool`,
  `model.moe_router_dtype: fp32 | none`, and load-balance metrics for HF
  MoE models (Qwen3-MoE, GPT-OSS). EP itself is a second PR after T2-1.
- Note: prime-rl ships its own MoE model implementations. Decide with the
  repo owner whether Wavelet stays HF-only (EP via HF hooks) or adds a
  `wavelet/trainer/models/` zoo.

### T2-3 Context parallelism (L, depends on T2-1)
- Missing: `fsdp.cp` validators exist but the wrap raises.
- Reference: `utils/cp.py` (input/label sharding, logprob gather),
  `trainer/models/layers/ring_attn.py`, `ulysses_attn.py`,
  `trainer/rl/train.py` (search `cp_` for how loss normalisation and
  logprob all-gather interact), `configs/trainer.py` `cp_style`.
- Wavelet touch points: `wavelet/data/rl.py` (shard packed rows along seq),
  `wavelet/trainer/rl.py` (loss scale must count tokens across `dp_cp`),
  `wavelet/trainer/losses.py` (chunked LM head over the local shard),
  `wavelet/trainer/distributed.py` (`dp_shard_cp`, `cp` submeshes already
  exist).
- Start with ring attention through
  `torch.distributed.tensor.experimental.context_parallel` on SDPA before
  attempting Ulysses.

### T2-4 `torch.compile` per block (M)
- Missing entirely.
- Reference: `trainer/model.py` (search `torch.compile`, recompile limit
  and cache size env vars), `configs/trainer.py` `CompileConfig`.
- Where: `wavelet/trainer/model.py` after wrap, gated by
  `model.compile: bool` and `model.compile_fullgraph: bool`. Compile each
  decoder layer, not the whole model. Confirm it composes with LoRA fused
  kernels in `wavelet/kernels/` and with gradient checkpointing.
- Test: CPU smoke test that compiled tiny model matches eager loss.

### T2-5 Optimizer state CPU offload (M)
- Missing: only the `colocate_sleep` whole-optimizer move exists in
  `wavelet/trainer/trainer.py`.
- Reference: `trainer/optim/state_offload.py` (pinned CPU state, H2D/D2H
  streams around `step`), `configs/trainer.py` `optim_cpu_offload`.
- Where: `wavelet/trainer/optim.py`; config `optim.cpu_offload: bool`.
- Do not port the optimizer-in-backward C++ CPU Adam (`trainer/optim/offload.py`,
  `cpu_adam/`); too large for now.

### T2-6 Selective activation checkpointing (M)
- Missing: Wavelet calls HF `gradient_checkpointing_enable` on all layers.
- Reference: `trainer/activation_checkpointing.py` (`mode: full | selective`,
  op-policy `targets`, `freq`).
- Where: `wavelet/trainer/model.py`; config `model.activation_checkpointing`
  block. Must compose with `wavelet/kernels/smart_gc.py` (sqrt-N CPU offload
  checkpointing) rather than fight it.

### T2-7 Loss architecture (L)
- Missing: single hard-coded DPPO loss in `wavelet/trainer/losses.py`.
  No custom loss import, no separate `rl / ce / ref_kl` components, teacher KL
  is folded into the advantage.
- Reference: `trainer/rl/loss.py` (`compute_loss`, component weights each
  normalised by its own global token count, `ref_kl` one-sided trust region,
  `ce` component), `configs/trainer.py` `LossConfig` (`import_path`).
- Where: `wavelet/trainer/losses.py`, `wavelet/trainer/rl.py`
  (`_average_data_parallel_loss_scale` must become per-component),
  `wavelet/configs/config.py` `RLLossConfig`, `wavelet/data/rl.py`
  (per-token weight streams already exist for advantages and temperatures;
  add `ce_weight` and `ref_kl_weight` streams).
- Keep the current DPPO behaviour byte-identical when only the `rl` component
  is configured. Add a golden-value test.

### T2-8 Sampling-mask replay for top-p / top-k (L, needs vLLM bump T4-1)
- Missing: Wavelet rejects `top_p < 1`, `top_k > 0`, `min_p > 0` at config
  time (`validate_train_sampling_replay` in `wavelet/configs/config.py`).
- Reference: `trainer/rl/loss.py` (masked logsumexp),
  `trainer/models/layers/lm_head.py` (`sampling_mask`),
  `trainer/rl/data.py` (`sampling_mask` field),
  `transports/batch/types.py`, `configs/orchestrator.py` (search
  `sampling_mask`), `docs/inference.md` "sampling replay".
- Flow: vLLM returns the per-position sampling mask, orchestrator stores it
  on the row (`wavelet/data/rl.py` `RLExample`), trainer renormalises logits
  over the mask before computing logprobs. Then relax the validator.

### T2-9 Checkpoint features (M)
- Any-world-size resume: `wavelet/trainer/ckpt.py` rejects a mismatched
  `world_size`. DCP supports resharding; remove the check once the state
  dict is fully DCP-managed (after T2-1). Keep the check for the
  dataloader state, which is per rank, and document the limitation.
- `ckpt.keep_interval`: permanent checkpoints every N steps in addition to
  `keep_last`. Reference: `trainer/ckpt.py` (search `keep_interval`).
- End-of-run checkpoint: always save at the final step even if
  `step % interval != 0`. Reference: `trainer/rl/train.py` (final save).
- Resume from an external directory (fork a run) and skip flags
  (`skip_optimizer`, `skip_scheduler`, `skip_dataloader`, `skip_progress`).
  Reference: `configs/shared.py` `ResumeConfig`, `trainer/ckpt.py`.
- Progress state: add `total_tokens` and `total_samples` to `TrainerState`.

### T2-10 Profiling and MFU (M)
- Memory profiler: `torch.cuda.memory._record_memory_history` snapshot per
  step to `<output_dir>/memory/`. Reference: `trainer/utils.py`
  (search `memory_snapshot`).
- Torch profiler chrome trace for steps in a range. Reference:
  `trainer/rl/train.py` (search `trace_path`).
- MFU: architecture peak-FLOPs table and FLOPs/token (dense and MoE aware).
  Reference: `trainer/perf.py`.
- Deterministic GC: `gc.disable()` and `gc.collect()` every N steps.
  Reference: `trainer/utils.py` (search `gc.freeze`), `configs/trainer.py`
  `GCConfig`.

### T2-11 Smaller trainer items (S each)
- `matmul_precision: highest | high | medium` instead of the `allow_tf32`
  bool. Reference: `configs/trainer.py`, `trainer/rl/train.py`.
- Configurable `dist_timeout_seconds` (currently hard-coded 30 minutes in
  `wavelet/trainer/trainer.py`).
- SignSGD optimizer. Reference: `trainer/sign_sgd.py`.
- FA3 attention backend selection by SM version. Reference:
  `trainer/model.py` (search `flash_attention_3`).

---

## 6. Tier 3: orchestrator and algorithms

### T3-1 Multiple training environments with mixing (L)
- Missing: `orchestrator.verifier_env_id` is a single string. No per-env
  ratio, sampling override, group size, or algorithm.
- Reference: `configs/orchestrator.py` (`TrainSourceConfig`: `ratio`,
  `sampling`, `group_size`, `algo`), `orchestrator/train_source.py`
  (weighted source selection, per-env metrics), `orchestrator/orchestrator.py`
  (`finalize_train_batch` per-env metric prefixes).
- Where: `wavelet/configs/config.py` (new `orchestrator.envs: list[EnvSpec]`
  with legacy alias from `verifier_env_id`), `wavelet/orchestrator/envs.py`
  (load N environments, per-env client sampling args),
  `wavelet/orchestrator/scheduler.py` (`_next_record` samples an env by
  ratio, then a record from that env's cursor; cursors and epochs per env),
  `wavelet/orchestrator/advantage.py` (per-env algorithm dispatch),
  `wavelet/monitor.py` (`_add_environment_metrics` already groups by
  `env_name`; verify it emits `train/<env>/...`),
  `examples/prepare_verifier_rl_data.py` (multiple env datasets).
- Resume: per-env cursors must be recoverable. Today the cursor is derived
  from chunk offsets; extend the manifest with per-env counts.

### T3-2 Adaptive in-flight concurrency (L)
- Missing: `orchestrator.max_inflight_rollouts` is static.
- Reference: `orchestrator/concurrency.py` (AIMD controller on KV-cache
  usage, preemption count, queue depth; cancels youngest requests on
  overload), `orchestrator/inference_metrics.py` (scrapes vLLM Prometheus
  `/metrics`), `configs/orchestrator.py` (`min_inflight`, `max_inflight`,
  targets).
- Where: new `wavelet/orchestrator/concurrency.py`; hook into
  `VerifierRolloutScheduler._fill_inflight` in
  `wavelet/orchestrator/scheduler.py`; metrics scraper in
  `wavelet/inference/http.py` or a new module. Log the scraped engine
  metrics to W&B as `inference/<replica>/...`.
- Keep static caps as the fallback when scraping fails.

### T3-3 Curriculum: difficulty pool and admission gates (L)
- Missing: only the zero-advantage filter. No easy/hard reweighting or
  online difficulty tracking.
- Reference: `orchestrator/curriculum/base.py`, `gates/adv.py`
  (`AdvRangeGate`), `samplers/pool.py` (per-task reward EMA, easy/normal/hard
  pools with thresholds and weights, checkpointed `state_dict`),
  `configs/orchestrator.py` (search `curriculum`).
- Where: new `wavelet/orchestrator/curriculum.py` with `Gate` and `Sampler`
  protocols; wire into `_next_record` and `_is_usable_training_group` in
  `wavelet/orchestrator/scheduler.py`. Persist sampler state next to the
  queue manifest so resume restores it (this also closes the open audit
  item "record cursor not checkpointed" in `rl_audit_progress.md`).

### T3-4 Prefill-scoring client and distillation algorithms (L)
- Missing: the trainer consumes `teacher_logprobs` but nothing produces them.
  No OPD, OPSD, or online SFT-distill.
- Reference: `orchestrator/clients.py` (prefill with `prompt_logprobs`),
  `orchestrator/algo/opd.py`, `algo/opsd.py`, `algo/sft.py`,
  `orchestrator/generation_source.py` (frozen generator), `docs/algorithms.md`.
- Where: `wavelet/orchestrator/envs.py` (teacher client using vLLM
  `prompt_logprobs` on the sampled tokens),
  `wavelet/orchestrator/algorithms.py` (new algorithms filling
  `teacher_logprobs` on rows), `wavelet/inference/server.py` (make sure the
  scoring route returns per-token prompt logprobs at temperature 1 for a
  second model instance), config for `teacher.model`, `teacher.base_url`.
- Depends on T2-7 for a proper `ref_kl` component; can start with the
  existing advantage-folding path.

### T3-5 Token-based batch size (M)
- Missing: `examples_per_step` only.
- Reference: `configs/orchestrator.py` `token_batch_size`,
  `orchestrator/train_sink.py` (accumulate groups until token budget, ship
  partial groups to next step).
- Where: `wavelet/orchestrator/scheduler.py` `generate_batch` and
  `_finalize_batch`; `wavelet/orchestrator/schedule.py` for chunk
  accounting; trainer expects variable group counts already (verify in
  `wavelet/trainer/rl.py`).

### T3-6 Evals: concurrency, cancellation, standalone command (M)
- Missing: evals run inline in `_record_loaded_policy` and block publishing;
  `_run_eval_examples` gathers unbounded; evals cannot be cancelled when a
  newer policy arrives; eval-only requires the full RL launcher with
  `max_steps: 0`.
- Reference: `orchestrator/dispatcher.py` (shared permit pool,
  `PREFER_EVAL`, `cancel_eval_step`), `evals/evals.py` (standalone runner
  with resume cursor, watches weight broadcasts), `configs/evals.py`.
- Where: `wavelet/orchestrator/envs.py` (`_run_eval_examples` with a
  semaphore), `wavelet/orchestrator/scheduler.py` (run evals as a background
  task; cancel on new policy when `eval.cancel_on_new_policy`), new
  `wavelet/entrypoints/rl_evals.py` and `wavelet evals` in `wavelet/cli.py`
  that reuses `envs.py` against a running server.
- Also: unbiased pass@k and pass^k. Reference: `orchestrator/utils.py`
  (search `pass_at_k`), `orchestrator/metrics.py`. Wavelet's current pass@k
  in `wavelet/orchestrator/eval_utils.py` is the biased product form.

### T3-7 Failure accounting and timing metrics (M)
- Missing: errored rollouts are dropped with a warning; no per-error-type
  counts; only whole-rollout elapsed time.
- Reference: `orchestrator/types.py` (`DispatchFailure`),
  `orchestrator/metrics.py` (per-error-type, per-phase timing),
  `orchestrator/dispatcher.py` (promote empty/errored traces to failures).
- Where: `wavelet/orchestrator/envs.py`, `wavelet/orchestrator/rollout_metadata.py`,
  `wavelet/monitor.py` (`fate/*` metrics already exist; add
  `fate/errors/<type>`).

### T3-8 Length penalties (S)
- Missing: prime-rl's linear length penalty with output/input/turn weights
  scaled by group-mean reward, and a max-length (truncation) penalty. Wavelet's
  `advantage.py` has an efficiency bonus with different semantics.
- Reference: `orchestrator/algo/grpo.py`, `configs/algorithm.py`
  `LinearLengthPenaltyConfig`.
- Where: `wavelet/orchestrator/advantage.py`, config under `algo`. Keep the
  existing bonus as its own option; add `linear` and `truncation` penalties.
  Document all three in `docs/algorithms.md`.

### T3-9 Smaller orchestrator items (S each)
- `tasks_per_minute` rate limit and admission burst smoothing. Reference:
  `orchestrator/dispatcher.py` (search `AsyncLimiter`).
- Eval `reasoning_effort` knob. Reference: `configs/orchestrator.py`.
- Staleness metrics split into in-flight vs in-queue. Reference:
  `orchestrator/utils.py` (search `off_policy`).
- Periodic one-line pipeline status log. Reference:
  `orchestrator/periodic_logger.py`.

---

## 7. Tier 4: inference and transport

### T4-1 vLLM version bump (L, design review first)
- Wavelet locks vLLM 0.19.1 (`uv.lock`); prime-rl pins 0.28.0. Many prime-rl
  patches and T2-8 depend on newer APIs (layerwise reload, sampling mask
  return, `/inference/v1/generate`).
- Reference: `ref/prime-rl/pyproject.toml` (pinned wheel URL), `inference/patches.py`
  (each patch is documented with the vLLM version it targets).
- Where: `pyproject.toml`, `wavelet/inference/patches.py`,
  `wavelet/inference/server.py` (`OpenAIServingChatWithTokens` subclasses
  vLLM internals that change between versions),
  `wavelet/transport/policy.py` (`NCCLWeightTransferEngine` import path,
  `load_weights` API), `wavelet/inference/vllm_weight_update.py`.
- Run `tests/test_vllm_server_patches.py` and the GPU smoke examples after
  the bump.

### T4-2 fp32 LM head and fp32 router logits at inference (S after T4-1)
- Reference: `inference/patches.py` (search `fp32_lm_head`),
  `configs/inference.py` (`enable_fp32_lm_head`, `moe_router_dtype` via
  `hf_overrides`).
- Where: `wavelet/inference/patches.py`, `wavelet/inference/server.py`
  argv, config knobs under `inference.vllm`.

### T4-3 NCCL broadcast efficiency (M)
- Missing: `wavelet/transport/policy.py` gathers the full state dict to rank 0
  then broadcasts one tensor at a time.
- Reference: `transports/weights/nccl.py` (per-layer, dtype-grouped
  concatenated buffers; HF conversion per layer), `inference/vllm/worker/nccl.py`.
- Where: sender side in `wavelet/transport/policy.py`
  (`_export_nccl_policy`), receiver in the same module. Keep the
  `NCCL_READY` handshake unchanged.
- Skip FP8 quantized transfer and MLA absorbed-weight recompute for now.

### T4-4 Layerwise filesystem reload (S after T4-1)
- Reference: `inference/vllm/worker/weight_transfer.py`
  (`initialize_layerwise_reload`, `finalize_layerwise_reload`),
  `inference/vllm/worker/filesystem.py`.
- Where: `wavelet/transport/policy.py` worker extension `load_weights` path.

### T4-5 Liveness probe route (S)
- Missing: `/health` is API-process level only. Workers already have a
  `liveness_probe` RPC in `wavelet/transport/policy.py` but no route.
- Reference: `inference/vllm/server.py` `/liveness` (engine no-op RPC with
  timeout).
- Where: `wavelet/inference/server.py`; use it from the orchestrator's
  health wait.

### T4-6 Reasoning parser auto-resolution (S)
- Missing: `tool_call_parser` auto-resolves by model family;
  `reasoning_parser` is manual.
- Reference: `packages/prime-rl-configs/src/prime_rl/utils/parsers.py`.
- Where: `wavelet/inference/server.py` (`MODEL_TOOL_CALL_PARSER` has the
  pattern to extend).

### T4-7 ZMQ batch transport (L, optional)
- Missing: filesystem queue only. prime-rl has a ZMQ PUB/SUB transport with a
  READY barrier and msgpack binary encoding.
- Reference: `transports/batch/zmq.py`, `transports/batch/types.py`,
  `configs/shared.py` (search `zmq`).
- Where: `wavelet/transport/queue.py` behind `transport.type: zmq`. Keep the
  filesystem queue as default; it has manifest and audit features the ZMQ
  path lacks. Only do this if a single-node in-memory path is needed.

### T4-8 Inference process env vars (S)
- Missing: per-role `env_vars` table with protected-variable rejection.
- Reference: `configs/shared.py` (`env_vars`, protected list),
  `entrypoints/rl.py` (applied at spawn).
- Where: `wavelet/configs/config.py` `LauncherConfig`,
  `wavelet/orchestrator/launcher.py` `_run_role_subprocess`.

---

## 8. Tier 5: ops, CLI, monitoring, tooling

### T5-1 Config CLI ergonomics (M)
- Missing: no `--help` from field docs, one `@ file` only, no bare
  `--flag` / `--no-flag`, no kebab-case.
- Reference: `docs/configuration.md` (TOML composition, overrides, booleans,
  optional sub-configs), `utils/config.py`, and the `deps/pydantic-config`
  submodule if present.
- Where: `wavelet/utils/config.py` `load_config`, `wavelet/cli.py`.
  Multiple `@` files deep-merge left to right. Generate `--help` from Pydantic
  field descriptions (add `Field(description=...)` as you go).

### T5-2 Launch attempt bookkeeping (S)
- Missing: `configs/*.yaml` and logs are overwritten/appended per launch.
- Reference: `utils/pathing.py` (`configs/attempt_<n>/`, `latest` symlink,
  `command.txt`, launch config copy).
- Where: `wavelet/orchestrator/runtime.py` (where role YAMLs are written),
  `wavelet/orchestrator/launcher.py` (log path per attempt). Update the
  `webui` config endpoint in `wavelet/orchestrator/state_server.py` to read
  `latest`.

### T5-3 Shared W&B run across roles (M)
- Missing: trainer and orchestrator create two runs linked by `group`.
- Reference: `entrypoints/rl.py` (`WANDB_SHARED_MODE`, run id env),
  `monitors/wandb/monitor.py`.
- Where: `wavelet/monitor.py` (`_init_wandb`, `_wandb_log`),
  `wavelet/orchestrator/runtime.py` (pass the run id to roles). Wavelet
  already persists `wandb_run_id.txt` for resume; reuse it.
- Also: time-keyed rows for metrics with no step, and a saved overview
  workspace. Reference: `monitors/wandb/overview.py`.

### T5-4 Structured logging (S, closes T0-7)
- Reference: `utils/logger.py` (JSON sink, rank filter, `vf_level`).
- Where: `wavelet/monitor.py` `setup_logger`.

### T5-5 Console scripts (S)
- Missing: `[project.scripts]` in `pyproject.toml`. Users must run
  `python -m wavelet <cmd>`.
- Add `wavelet = "wavelet.cli:main"` and keep `python -m wavelet` working.

### T5-6 Deployment surface (L, optional)
- SLURM sbatch generation from Jinja templates for single and multi-node.
  Reference: `entrypoints/rl.py` (search `sbatch`), `templates/*.j2`,
  `docs/scaling.md`. Where: new `wavelet/entrypoints/slurm.py`; document the
  existing Ray backend in `docs/` at the same time.
- Dockerfile. Reference: `ref/prime-rl/Dockerfile.cuda`.
- `pre_download_model` before spawning roles. Reference: `entrypoints/rl.py`.

### T5-7 CI and tests (M)
- Missing: no `.github/workflows`, no unit/integration split, no end-to-end
  subprocess test.
- Reference: `ref/prime-rl/.github/workflows/*.yaml`,
  `ref/prime-rl/tests/integration/test_reverse_text*.py`,
  `ref/prime-rl/tests/utils.py`, `docs/development.md`.
- Where: `.github/workflows/{style,cpu_tests,gpu_tests}.yaml`;
  `tests/integration/` with one end-to-end `reverse_text` run on fake or tiny
  models that asserts loss decreases and resume works.

### T5-8 Tools (S each)
- DCP to HF safetensors converter. Reference:
  `ref/prime-rl/tools/convert_dcp_to_bf16.py`, `utils/weights.py`.
- Traces to HF dataset exporter. Reference:
  `tools/convert_traces_to_hf_dataset.py`.
- Chat client against the inference server. Reference: `scripts/chat.py`.
- Benchmark harness with baselines. Reference: `benchmarks/`.

### T5-9 Dashboard depth (M, optional)
- Missing vs prime-rl: log viewer, per-attempt config view, multi-run
  registry, token-level trace overlays, auto-start on launch.
- Reference: `dashboard/server.py`, `dashboard/README.md`,
  `entrypoints/dashboard.py`.
- Where: `wavelet/orchestrator/state_server.py`, `webui/src/*`.

---

## 9. Do not regress these (Wavelet-only strengths)

- `integrated`, `colocate`, `colocate_sleep` launcher modes and the
  `/sleep` `/wake` choreography.
- Async DCP checkpointing (`ckpt.mode: async | async_with_pinned_mem`).
- Tensor parallel via HF `tp_plan` with LoRA gradient sync.
- Liger kernels, fused LoRA Triton kernels, QLoRA nf4, 8-bit optimizers.
- `sqrt` LR scheduler, sequence-level loss normalisation, explicit global
  seed, synchronized non-finite loss abort.
- In-trainer reference logprobs via `disable_adapter()` when inference
  logprobs are missing.
- Strict configs (`extra="forbid"`), legacy alias conflict detection, port
  and device-group validation.
- Filesystem queue provenance (manifest, claim, consumed), consumed-batch GC,
  resume pruning of policies and batches.
- W&B sample tables, `events.jsonl`, `run_metadata.json`, credential
  redaction.
- `wavelet debug preflight` diagnostics, state server and web UI queue views.
- Local JSONL SFT sources, `max_examples`, per-turn `step_loss_mask`,
  non-packed `pad` mode, epoch-derived `max_steps`.

---

## 10. Suggested order

1. All of Tier 0 (one week, parallelisable, good onboarding).
2. T1-1 entropy, T1-2 passthrough, T1-3 eager default, T1-4 admin retries,
   T1-6 SFT metrics, T1-8 hygiene.
3. T2-4 compile, T2-9 checkpoint features, T2-10 profiling.
4. T3-1 multi-env, T3-6 evals, T3-8 length penalties, T3-2 adaptive
   concurrency.
5. T2-1 FSDP2 (design review), then T2-2 MoE router/metrics, T2-3 CP,
   T2-5 offload, T2-6 selective AC.
6. T4-1 vLLM bump (design review), then T4-2, T4-4, T2-8 sampling replay.
7. T2-7 loss architecture, then T3-4 distillation.
8. Tier 5 as time allows; T5-1 and T5-2 first.

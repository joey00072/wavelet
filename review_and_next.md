# Review of the restructure work (round 2), and what to do next

Reviewed: uncommitted working tree on `master`, 84 files / +2237 / -1146
lines, against `plan.md` sections 4, 7, and 9. Round 1's findings were
addressed almost entirely; this round is about finishing, not redoing.
Date: 2026-09-21.

## Status update (same day, after fixes)

Sections 2.1 through 2.8 and the section 3 items below were applied by a
follow-up pass and re-verified by the reviewer: ruff lint and format clean,
`git diff --check` clean, vLLM-free import of trainer/transport/orchestrator/
launch works, `tests/test_import_layers.py` unchanged and passing, full
suite green. Concretely:

- 2.1 format: all files formatted.
- 2.2 helpers: single `unwrap_model` / `strip_training_wrapper_segments` in
  `wavelet/utils/modules.py`; `_barrier_world` deleted, exporter calls
  `trainer.distributed.barrier`.
- 2.3 conversion helpers: model-side ones in `wavelet/trainer/export_tensors.py`
  (plus `broadcast_model`), wire-format `_convert_layer_to_hf` in
  `wavelet/transport/weights/wire.py`; `nccl.py` owns only the broadcaster.
- 2.4 `PolicyExporter(trainer)` is required; explicit properties `config`,
  `world`, `model`, `step`, `tokenizer`, `output_dir`, `parallel_dims`, and a
  forwarding `offload_after_refit()`. Four tests updated to pass a fake.
- 2.5 six `{"http", "vllm_http"}` sites reverted to `== "vllm_http"`.
- 2.6 empty `inference/sglang/` and `trainer/hf/` removed.
- 2.7 `transport/queue.py` deleted; 29 importers now use
  `wavelet.transport.rollouts.filesystem`; docs updated.
- 2.8 two-replica mismatch test added; `importorskip("vllm")` and
  `object.__new__` comments added.
- Section 3: `run_rl_training` lost its unused `config`;
  `load_and_train_received_batch` returns `RolloutStepTimings` and the two
  runtime callers copy fields; deployment doc section removed.
  `orchestrator/verifiers.py` was **not** trimmed: `tests/test_verifiers_rollouts.py`
  and `tests/test_eval_utils.py` import most of the re-exported private
  names. Leave as is; trimming it means those tests should patch
  `orchestrator.scheduler` instead, which is a separate small PR.

Still owed: the PR split in section 4 and the GPU smoke in section 5.

## Verdict

**Close. Approve in principle; fix the eight items in section 2, split into
the PRs in section 4, run the GPU smoke, and this lands.**

Nothing in this tree is architecturally wrong any more. Phase C is gone,
the real code moved instead of being aliased, the shared loop is the loop
that runs, and the engine talks through the admin client. What remains is
cleanup, one format failure, a handful of leftovers from round 1, and the
still-missing split and GPU verification.

Checks:

| Check | Result |
| --- | --- |
| `uvx ruff check wavelet tests` | clean |
| `uv run ruff format --check wavelet tests` | **13 files would be reformatted** (list in 2.1) |
| `git diff --check` | clean |
| `uv run pytest` | 1460 passed, 10 skipped, 4m03s |
| `import wavelet.trainer/transport/orchestrator/launch.runtime` with vLLM blocked | works |
| `tests/test_import_layers.py` | passes; now catches `import_module("wavelet...")` strings and `sys.modules` aliases; `configs` allows only `utils` |
| `examples/` untouched | yes |
| `pyproject.toml` | only the vLLM plugin entry path changed; extras and `uv.lock` reverted |
| New YAML keys / CLI commands | none |
| GPU smoke | **not run**; no `nvidia-smi` on this machine. Still owed (section 5). |

## 1. What was fixed since round 1

Every blocking item from round 1 is resolved:

- All twelve `sys.modules` shims deleted, and the layer test now forbids them.
- `trainer/hf/`, `trainer/registry.py`, `trainer/megatron/`,
  `inference/registry.py`, `inference/sglang/`, `transport/rollouts/zmq.py`,
  `transport/weights/filesystem.py::FileSystemWeightSender`,
  `configs/roles.py`, all backend config keys, `RoleSpec.num_gpus`, the Ray
  placement-group change, `sglang-server`, and the three extras are gone.
- `transport/policy.py` moved with `git mv` to `transport/weights/nccl.py`
  (975 to 411 lines). The vLLM `Worker` subclasses now live in
  `inference/vllm/weight_update_worker.py` and import vLLM at module scope
  there. The trainer-side exporter lives in `trainer/policy_export.py` and
  imports `trainer.model` normally. No module-level `Any = None` patch
  points, no `import_module("wavelet...")` strings anywhere.
- `transport/queue.py` moved with `git mv` to `transport/rollouts/filesystem.py`;
  record dataclasses in `contracts/queue_records.py`; `rollouts/inspect.py`
  exposes the read-only helpers.
- `trainer/loop.py` is wired into both `trainer/rl.py::main` and
  `launch/runtime.py::_run_rollout_loop`/`_consume_and_train_step`. Per-phase
  timings (`wait_batch`, `load_rollout`, `train`, `export_policy`) preserved.
  Claim/consume are direct calls, not `getattr`.
- `HTTPPolicyInferenceEngine` now routes health/pause/resume/load_policy/
  sleep/wake through `HTTPEngineAdmin`; `base_urls()` is a method.
- `VERIFIER_ROLLOUT_FUNCTION` lives in `configs/constants.py`;
  `contracts/source.py` imports downward. `configs` allowlist is `{utils}`.
- `data/tokenizer.py`, `data/debug_tokenizer.py`, `utils/deepseek_compat.py`
  remove the last `inference -> trainer` and `orchestrator -> trainer` edges.
- `RLTrainer(BaseTrainer)` composes `self.policy_exporter = PolicyExporter(self)`
  instead of inheriting the mixin; `export_policy` / `should_export_policy`
  delegate, so callers are unchanged.
- `AGENTS.md`, `docs/architecture.md`, `docs/deployment.md`,
  `docs/functionality_register.md` describe only what exists.

## 2. Remaining issues (all small, all required)

### 2.1 Format check fails

`uv run ruff format --check` flags 13 files: `wavelet/inference/client.py`,
`wavelet/transport/rollouts/inspect.py`, and eleven tests
(`test_adaptive_concurrency`, `test_agent_trajectory`, `test_context_parallel`,
`test_deepseek_conversion`, `test_eval_resume`, `test_eval_utils`,
`test_live_trace`, `test_rl_loss`, `test_trainer_optim`, `test_vllm_batching`,
`test_vllm_weight_update`). Run `uv run ruff format wavelet tests`. The
eleven test files are the pre-existing formatting drift from round 1 that was
un-applied; put that in its own commit.

### 2.2 Three trainer helpers are still copies, not moves

`transport/weights/nccl.py` lines 25 to 47 define `_unwrap_model`,
`_strip_training_wrapper_segments`, and `_barrier_world`.
`trainer/model.py` still has `unwrap_model` (line 1298) and
`_strip_training_wrapper_segments` (line 1824), and `trainer/distributed.py`
still has `barrier`. Two copies of the same logic will drift.

Fix: these helpers are used by two consumers that are both allowed to
import `trainer`: `trainer/policy_export.py` and
`inference/vllm/weight_update_worker.py` (the worker runs inside vLLM, but
importing `wavelet.trainer.model` from it is a downward edge only if
`inference -> trainer` is allowed, which it is not). So:

- `_unwrap_model` and `_strip_training_wrapper_segments` are pure
  `nn.Module`/string functions with no trainer dependency. Move the single
  definition to `wavelet/utils/modules.py` (or similar under `utils/`),
  delete both copies, and import it from `trainer/model.py`,
  `transport/weights/nccl.py`, and the worker.
- `_barrier_world` is only called from `PolicyExporter._barrier`. Delete it
  from `nccl.py` and call `wavelet.trainer.distributed.barrier(self.world)`
  from `policy_export.py` directly. The copy also changed behaviour: it
  defaults `device` to CPU and `local_rank` to 0 via `getattr` when the
  world object lacks them, where the original fails fast.

### 2.3 HF conversion helpers are in `transport/weights/nccl.py`

`_convert_layer_to_hf`, `_model_named_tensors`, `_materialize_wire_tensors`,
`_iter_layer_state_dicts` are Hugging Face state-dict shaping, not NCCL
transport. Plan Step 2 puts them in `trainer/hf/export.py`; since `trainer/hf/`
is now (correctly) deferred, put them in `trainer/export_tensors.py` for
now. The worker imports them today via `transport.weights.nccl`; after the
move the worker would need `inference -> trainer`, which the layer test
forbids. Resolution: the worker only needs `_iter_layer_state_dicts` and
`_convert_layer_to_hf` to reshape what arrives over the wire; those two
operate on plain `dict[str, Tensor]` and do not touch `nn.Module`, so they
can stay in `transport/weights/` as wire-format helpers under a clearer
module name (`transport/weights/wire.py`). `_model_named_tensors` and
`_materialize_wire_tensors` take an `nn.Module` and belong to the trainer.
Add a one-line note in `nccl.py`'s docstring saying it owns the
broadcaster and wire format, nothing model-side.

### 2.4 `PolicyExporter.__getattr__` forwards everything to the trainer

`trainer/policy_export.py::PolicyExporter.__getattr__` delegates any unknown
attribute to `self._trainer`, so the exporter body still reads
`self.config`, `self.world`, `self.model`, `self.step`, etc. as if it were
the mixin. That works, but it hides the contract: nobody can tell which
trainer attributes the exporter depends on, and a typo becomes an
`AttributeError` at export time instead of at construction. Tests construct
`PolicyExporter()` with `trainer=None`, which then raises on every
attribute, so the tests only pass because they patch around it.

Fix, in this PR since it is small: replace `__getattr__` with explicit
properties for the handful of attributes the exporter actually reads
(`config`, `world`, `model`, `step`, `tokenizer`, whatever the grep shows,
likely under ten) and make `trainer` a required constructor argument.
Update the four tests that construct `PolicyExporter()` bare to pass a
`SimpleNamespace` with those attributes. This is the difference between a
mixin renamed and a real object boundary, and it is what the
`TrainerBackend` protocol in Step 6 will build on.

### 2.5 Leftover `mode in {"http", "vllm_http"}` checks

`"http"` was removed from the `inference.mode` literal, but six call sites
still test for it: `wavelet/debug.py` lines 988, 1956, 2085, 2386,
`wavelet/orchestrator/scheduler.py:3137`, and
`wavelet/launch/runtime.py:159`. Revert them to `== "vllm_http"` so the
code matches the config surface. When Phase B adds `backend`, the legacy
value will be normalized in one `mode="before"` validator, not compared in
six places.

### 2.6 Empty directories

`wavelet/inference/sglang/` and `wavelet/trainer/hf/` still exist with only
`__pycache__` inside. Remove them (`git clean -fdx wavelet/inference/sglang
wavelet/trainer/hf` after confirming nothing tracked is under them).

### 2.7 `transport/queue.py` re-export uses `import *`

The one-release compatibility module is allowed by plan Step 3, but it does
`from wavelet.transport.rollouts.filesystem import *` plus two private
names. 13 test files and 8 package modules still import from
`wavelet.transport.queue`. Two choices, pick one and say so in the PR:
either update all 21 importers now and delete the module (preferred, since
they are all internal), or keep it for one release with an explicit
`__all__` list instead of `import *` and a `TODO(remove after <date>)` line.
Do not leave `import *`.

### 2.8 Test coverage narrowed in two places

- `tests/test_http_inference.py::test_hot_swap_failure_does_not_advance_policy`
  used to return `[{"policy_step": 3}, {"policy_step": 2}]`, i.e. one of two
  replicas reports a stale version and the engine must refuse to advance.
  It now returns a single `{"policy_step": 2}`. The multi-replica mismatch
  path is no longer tested. Add a two-replica variant: set `engine._base_urls`
  to two URLs, give `engine.admin` two `HTTPEngineAdmin` objects whose
  `_request` return 3 and 2 respectively, and assert the engine raises and
  `policy_step` stays 2.
- `tests/test_vllm_weight_update.py` constructs workers with
  `object.__new__(FileSystemWeightUpdateWorker)` instead of calling the
  constructor, because the worker module now imports the real vLLM
  `Worker` at module scope (previously it fell back to `object` when vLLM
  was missing). Acceptable for a unit test of the update methods, but add
  one line in the test docstring saying why `__init__` is bypassed, and
  mark the module `pytest.importorskip("vllm")` so it skips cleanly rather
  than erroring on a vLLM-free machine.

## 3. Minor notes (fix if touching the file anyway)

- `orchestrator/verifiers.py` went from a `sys.modules` alias to an explicit
  re-export of 22 names, 20 of them underscored. The public config value
  `wavelet.orchestrator.verifiers:generate_rollouts` is the only thing that
  needs to resolve. Grep shows no test patches `verifiers._x`, so trim the
  re-export to `generate_rollouts` and `VerifierRolloutScheduler`. If a
  later test needs a private name, it should patch `orchestrator.scheduler`.
- `trainer/loop.py::load_and_train_received_batch` takes `timings: Any |
  None` and writes `load_data`/`train_until`/`export_policy` onto it for the
  integrated launcher's `StepTimes`. Type it as `StepTimes | None` (import
  under `TYPE_CHECKING` if needed) or, better, have the caller copy fields
  from the returned `RolloutStepTimings`. `Any` here defeats the point of
  returning a dataclass.
- `run_rl_training(config, ...)` never uses `config`. Drop the parameter.
- `_run_streaming_rollout_training` in `trainer/rl.py` still has its own
  per-batch body (validate, claim, load, train, consume, export, offload).
  Not a blocker for this round since it existed before, but note it in the
  plan as the Step 3 leftover: the streaming path should call
  `load_and_train_received_batch` once the chunk accumulator hands it a
  materialized batch.
- `HTTPEngineAdmin(request=...)` lets the engine inject its own `_request`
  so retries and per-path timeouts stay in one place. Fine. But then the
  admin's own `_request`/`_request_with_retries` are dead when the override
  is set; either the engine should stop having a `_request` of its own and
  use the admin's, or the admin should not carry a second implementation.
  Resolve in Step 4 follow-up; note it in the plan.
- `docs/deployment.md` gained a short "vLLM inference" section that says
  little. Fold it into the existing inference diagnostics doc or drop it.

## 4. Split into PRs before merging

Still one uncommitted tree on `master`. The split is now easy because
nothing has to be removed, only grouped. Suggested order; each PR carries
the `plan.md` section 9 checklist.

1. **Step 0: import-layer test.** `tests/test_import_layers.py` only. Its
   `KNOWN_VIOLATIONS` must match `master` at that commit, so seed it with
   today's `master` edges and shrink it in the following PRs.
2. **Format drift.** `uv run ruff format` on the eleven test files, nothing
   else.
3. **Step 1: contracts.** `contracts/` renames and `queue_records.py`,
   `configs/constants.py`, `contracts/source.py`, `orchestrator/sources.py`
   and `orchestrator/verifiers.py` changes (with the 3.1 trim), import
   updates, doc rows.
4. **Step 2: weights transport and policy exporter.** `transport/weights/`,
   `trainer/policy_export.py` (with 2.4 done), `inference/vllm/weight_update_worker.py`,
   2.2 and 2.3 helper moves, `RLTrainer` composition, test updates.
   GPU smoke required.
5. **Step 3: rollouts transport and shared loop.** `transport/rollouts/`,
   `transport/queue.py` decision from 2.7, `trainer/base.py`,
   `trainer/loop.py`, both call-site rewires, `tests/test_trainer_loop.py`.
   GPU smoke required.
6. **Step 4: inference/vllm and admin client.** The five renames,
   `inference/base.py`, `inference/client.py`, `data/tokenizer.py`,
   `data/debug_tokenizer.py`, `utils/deepseek_compat.py`, engine routing,
   2.8 test additions, CLI and plugin path. GPU smoke required.
7. **Launch move.** `launch/{roles,placement,runtime}.py` renames, 2.5
   reverts, CLI `rl` entry, `deployment/slurm.py` import update.

Branch naming and PR title format are in `plan.md` section 4.

## 5. GPU verification still owed

For PRs 4, 5, and 6 above, on a clean output directory, against a `master`
run of the same config:

```bash
uv run wavelet debug preflight @ examples/reverse_text/rl.yaml --json
uv run wavelet rl @ examples/reverse_text/rl.yaml --output_dir outputs/review_step_N
```

Compare and paste into the PR: `policies/step-*/policy.json` fields,
`rollouts/step-*/manifest.json` keys, `metrics.jsonl` keys, baseline and
final eval, failed rollout count, queue event kinds, and that all role
processes exited. If you do not have a GPU box, say so in the PR and I will
run it.

## 6. Updates to `plan.md`

- Step 2: replace "`trainer/hf/export.py`" with "`trainer/export_tensors.py`
  until Step 6 creates `trainer/hf/`". Add: `_unwrap_model` and
  `_strip_training_wrapper_segments` live once, under `utils/`.
- Step 2: `PolicyExporter` takes `trainer` as a required argument and
  exposes the attributes it reads as explicit properties; no `__getattr__`
  forwarding.
- Step 3: note the streaming path (`_run_streaming_rollout_training`) as a
  follow-up to route through `load_and_train_received_batch`.
- Step 4: decide the single owner of HTTP request/retry logic (engine or
  admin) and remove the other.
- Section 7: add "Do not use `from x import *` in re-export modules; list
  names explicitly and include a removal date."

## 7. Summary for the intern

This is a real turnaround from round 1. The architecture is now what the
plan asked for, the tests are green, and nothing pretends to be more than
it is. Finish with:

1. `uv run ruff format wavelet tests` and commit the format drift separately.
2. Fix 2.2 through 2.8 (each is under an hour; 2.4 is the largest).
3. Split into the seven PRs in section 4, in order.
4. Attach the GPU smoke comparison to PRs 4, 5, 6, or tell me you need a
   machine.
5. Stop there. Phase B waits for joey's sign-off on the Phase A checkpoint.

# Native SWE comparison

The native bridge runs R2E-Gym training and SWE-bench Verified evaluation with
the same Verifiers v1 taskset, bash harness, Docker runtime, and test reward as
the reference implementation. Wavelet retains its legacy Verifiers dependency;
the bridge starts a subprocess with a separate Python environment containing
Verifiers v1, PrimeRL, `r2e-gym`, and `swebench-verified`.

Run commands from the Wavelet repository root. The native Python environment
must be available on every orchestrator node, along with Docker and the model
cache. Use the same native environment JSON for both frameworks.

Check `docker info --format '{{.Driver}}'` on each sandbox node before increasing
concurrency. The cluster's `vfs` driver exhausted local disk at 128 episodes
because it copies filesystem layers. Use an overlay-backed daemon; the validated
experiment setup uses a separate `fuse-overlayfs` daemon selected with
`DOCKER_HOST=unix:///run/wavelet-docker.sock`. Export that variable in the SLURM
setup on every sandbox node so both frameworks use the same storage backend.
Keep the existing daemon available for unrelated workloads. Inspect both free
disk space and completed sandbox cleanup when increasing concurrency.
The private daemon also needs a working bridge for trusted setup downloads;
`--bridge none` breaks old images that must install a newer uv. Preserve the
native runtime's network cut after setup. Validate both gold patches and the
actual harness bootstrap before scaling concurrency.

Example training environment:

```json
{
  "taskset": {"id": "r2e-gym"},
  "agent": {
    "harness": {"id": "bash"},
    "runtime": {"type": "docker", "cpu": 2, "memory": 4},
    "max_turns": 40,
    "max_output_tokens": 24576,
    "timeout": {"setup": 900, "rollout": 1200, "scoring": 600}
  }
}
```

Create an evaluation environment JSON with `taskset.id=swebench-verified`.
Some task images ship an old `uv` without `uv sync --script`. The bridge applies
`runtime_compat.install()` to trigger the native installer for those images.
For the reference runner, launch its environment servers with the native Python
and `-m examples.qwen30b_swe.runtime_compat` instead of `env-server`, retaining
the same CLI arguments and adding the Wavelet source root to `PYTHONPATH`.
This applies the same bootstrap fix without editing reference files.

Export datasets with the native environment's Python:

```bash
"$NATIVE_PYTHON" -m examples.qwen30b_swe.prepare_native_data \
  --env-config "$RUN_ROOT/configs/train-env.json" \
  --output "$RUN_ROOT/train.jsonl"
"$NATIVE_PYTHON" -m examples.qwen30b_swe.prepare_native_data \
  --env-config "$RUN_ROOT/configs/eval-env.json" \
  --output "$RUN_ROOT/eval.jsonl" --limit 32
```

Use `orchestrator.verifier_env_id=examples.qwen30b_swe.bridge_env`. Its arguments
are `native_python`, `env_config`, `dataset_path`, `bridge_dir`, and `base_model`.
The evaluation environment uses the same module with the evaluation JSON,
dataset, and `training=false`, matching native chat-completions evaluation.
Set `data.path` to the exported training JSONL and `data.shuffle=false`
to retain native source order.

For 16 problem groups with 16 rollouts each, set `examples_per_step: 16`,
`rollouts_per_example: 16`, `filter_zero_advantage: true`, and
`refill_zero_advantage: false`. Match this with native
`constant_trainer_batch_size: false`, `batch_size: 256`, and `group_size: 16`.
These target batches before zero-advantage pruning. Failed-rollout handling
still differs: Wavelet retries missing group members, while the native sink can
admit surviving traces. Report attempted and successful counts for both.

The bridge preserves native token IDs, sampled log probabilities, and tool
masks. Every returned rollout must match a fingerprint of the native training
branches before Wavelet accepts it. Full native episodes are saved under
`bridge_dir/<environment-config-stem>/` for debugging, including failed episodes.
These are generated artifacts; do not commit them.
Failed bridge requests include the native episode identifier in the rollout
error, so the matching saved trace can be located directly.
The bridge checks subprocess liveness while requests are pending and reports
its exit code and log path on failure. Python fault handling is enabled in that
subprocess so native crashes leave a stack trace instead of a silent stall.
The HTTP bridge uses the standard asyncio event loop, matching native environment
workers rather than selecting a loop from optional installed packages.

Training uses [Prime's renderers](https://github.com/PrimeIntellect-ai/renderers)
through the native `TrainClient`. Pin `base_model` to the original model name
so loading a new LoRA policy cannot change tokenizer selection. The optional
`renderer` argument accepts the native renderer config, for example
`{"name": "qwen3"}` for compatible Qwen3 checkpoints. Use the same explicit
renderer in the reference configuration. Unknown model names can fall back to
a renderer without tool parsing; do not assume that a shared model architecture
means compatible chat templates. In particular, the installed renderer map does
not include Qwen3-Coder-30B-A3B-Instruct.

Check the initial prompt against the cached model template and exercise Qwen3
tool-call parsing before allocating GPUs:

```bash
"$NATIVE_PYTHON" -m examples.qwen30b_swe.check_renderer \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --renderer '{"name":"qwen3"}'
```

The installed generic Qwen3 renderer omits the Thinking-2507 template's initial
`<think>` opener. Its initial-prompt check fails for that model. Explicit
`{"name":"default","tool_parser":"qwen3","reasoning_parser":"think"}`
matches that template and parses tool calls, but falls back to full rendering
between turns. This diagnostic checks formatting, not task reward or learning.

The bridge preserves full-precision log probabilities when serializing native
traces. Native JSON's default rounding would change training samples and fail
the fingerprint check. A `taskset.system_prompt` file is applied to reconstructed
tasks as well as native taskset iteration, so both runners see the same override.

The current September comparison uses Qwen3-30B-A3B-Instruct-2507, 32k context,
both attention-only LoRA rank 32/alpha 64 and full finetuning, eight trainer GPUs
with EP8, and two independent TP4 inference replicas per framework. Independent
replicas avoid the idle-DP synchronization slowdown observed with the MoE model.
Full-model runs explicitly select vLLM's `moe_backend=triton`: the automatically
selected Blackwell kernel rejects this model's TP4 intermediate dimensions.
This adapts the stock 131k/full-training recipe to two nodes per framework.
Start with a two-update smoke run and a fixed
four-task evaluation before scaling.

`rl.yaml` is the native two-update, two-node LoRA smoke configuration. Put the
native Python environment at `.venv-native` (or change `native_python`), save
the environment JSON files as `outputs/qwen30b_swe_data/{train,eval}-env.json`,
and export their datasets as `rl_train.jsonl` and `rl_eval.jsonl` in that directory.
Set the shared project directory, SLURM partition, and Python command for your
cluster. All nodes need the same source, dependencies, data, and credentials.

```bash
uv run python -m wavelet debug preflight @ examples/qwen30b_swe/rl.yaml --json
uv run python -m wavelet rl @ examples/qwen30b_swe/rl.yaml
```

Use a fresh output directory for each attempt, including `bridge_dir` under the
native environment arguments. For the larger comparison, increase the batch to
16 × 16 as described above and use the independent frozen baseline instead of
mixing evaluation policies. The smoke checks integration; it does not establish
SWE learning or speed parity.

The smoke config uses `orchestrator.batch_selection: rollouts` with
`refill_zero_advantage: false`. Its target is `examples_per_step ×
rollouts_per_example` clean completed rollouts before zero-advantage pruning.
Failed independent members finish their group slots without replacement; GRPO
scores only clean survivors. A batch can therefore span more distinct problems
when members fail. Completed groups are scored before splitting across batches,
and the remaining scored members retain their original policy provenance.
This selection path needs a fresh run; active runs retain the code they loaded.

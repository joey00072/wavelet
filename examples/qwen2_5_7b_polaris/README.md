# Qwen2.5 7B Polaris GRPO

This example trains `Qwen/Qwen2.5-7B-Instruct` with BF16 LoRA and GRPO on
`POLARIS-Project/Polaris-Dataset-53K`. It keeps difficulty buckets `1/8`
through `6/8`, removes normalized duplicates, and removes exact normalized
overlaps with the held-out AIME 2024 evaluation set. Proof requests are removed
because the binary final-answer rubric cannot validate proofs and some of those
rows contain corrupted target fragments. The environment also removes narrowly
detected incomplete labels such as empty fraction operands; after all default
filters, the current training split contains 29,105 examples.

Every response must be exactly
`<think>...</think><answer>...</answer>`. Training uses eight problems per
optimizer step and eight rollouts per problem. Zero-advantage filtering retries
all-correct and all-wrong groups so the trainer receives mixed groups.

AIME 2024 evaluation uses the pinned 30-problem source from Prime's environment,
eight rollouts per problem, and the same strict response parser and math
verifier. Run the zero-step config first to establish the untrained baseline;
it starts no trainer process.

```bash
uv sync --extra flash-attn
uv pip install --python .venv/bin/python \
  --editable environments/polaris_math_tagged
uv run python examples/qwen2_5_7b_polaris/prepare_rl_data.py
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/eval_baseline.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/eval_baseline.yaml
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_100k.yaml --json
uv run python -m wavelet rl @ examples/qwen2_5_7b_polaris/rl_100k.yaml
```

The long run evaluates AIME 2024 every 100 policy steps. It retains five policy
exports, two checkpoints, five consumed training batches, and two evaluation
rollout sets. The monitor retains at most eight sampled completions every ten
steps for reward-hacking inspection. Async checkpoint staging uses threads and
ordinary CPU tensors instead of allocating one POSIX shared-memory file
descriptor per tensor storage. The smoke config writes a checkpoint at step one
to validate that path before a long run.

## Incorrect synthetic solutions

`generate_incorrect_synthetic.py` selects 100 deterministic, unique Polaris
problems from the hard `1/8` bucket after AIME 2024 decontamination. It samples
eight responses per prompt from a loaded policy adapter. Responses that violate
the exact `<think>...</think><answer>...</answer>` contract or pass Prime's math
verifier are discarded. Reasoning that contains line-start list markers `1.`,
`2.`, and `3.` is also discarded. Only valid-format, non-numbered incorrect
solutions are written to `incorrect.jsonl`. `summary.json` contains aggregate
rejection counts without retaining rejected completions.

With a Wavelet inference server running, generate from a stable adapter using:

```bash
uv run python examples/qwen2_5_7b_polaris/generate_incorrect_synthetic.py \
  --policy-dir outputs/<run>/final_adapter_step-000208 \
  --policy-step 208
```

`generate_wait_recoveries.py` turns those incorrect traces into recovery data.
For each trace it replaces the suffix after the second- and fourth-last newline
inside `<think>` with one of six evenly rotated phrases such as `Wait,` or
`I think I made a mistake`, then continues from that exact assistant-token
prefix. By default it samples four continuations per prefix. It retains only
deduplicated, strict-format, non-numbered traces that Prime's math verifier marks
correct:

```bash
uv run python examples/qwen2_5_7b_polaris/generate_wait_recoveries.py \
  --policy-dir outputs/<run>/final_adapter_step-000208 \
  --policy-step 208 \
  --rollouts 4
```

For midpoint interventions, the generator snaps the center of `<think>` to the
nearest paragraph or line boundary and can expand every source trace with an
explicit phrase set:

```bash
uv run python examples/qwen2_5_7b_polaris/generate_wait_recoveries.py \
  --policy-dir outputs/<run>/final_adapter_step-000208 \
  --policy-step 208 \
  --cut-mode midpoint \
  --recovery-phrases "Alternatively," "Wait," \
  --rollouts 4
```

Browse random verified recovery traces with the read-only viewer:

```bash
uv run python examples/qwen2_5_7b_polaris/serve_recoveries.py --port 8781
```

Fine-tune the step-208 RL adapter on the 312 verified recovery traces for three
epochs at learning rate `1e-5`, then evaluate the resulting adapter on all 30
AIME 2024 problems with eight samples per problem:

```bash
uv run python -m wavelet sft \
  @ examples/qwen2_5_7b_polaris/sft_recoveries.yaml
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/eval_recovery_sft.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/eval_recovery_sft.yaml
```

The SFT config uses full BF16 LoRA, a global batch size of eight, and 117
optimizer steps (`312 / 8 * 3`). Its 2,048-token training window covers every
verified trace; AIME evaluation retains the 8,192-token generation window and
the same strict tagged system prompt and verifier used by the baseline.

To continue that resulting adapter for three more epochs at learning rate
`1e-4`, while preserving the three-epoch output, run:

```bash
uv run python -m wavelet sft \
  @ examples/qwen2_5_7b_polaris/sft_recoveries_plus3_lr1e4.yaml
```

This continuation performs 117 optimizer steps (`312 / 8 * 3`) and writes to
a separate output directory.

Evaluate the continued adapter on the same AIME 2024 pass@8 setup:

```bash
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/eval_recovery_sft_plus3_lr1e4.yaml
```

For a clean comparison that does not inherit the RL adapter, rerun the base
model baseline, train a newly initialized LoRA for three epochs at `2e-4`, and
evaluate it with the identical AIME 2024 settings:

```bash
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/eval_baseline_rerun.yaml
uv run python -m wavelet sft \
  @ examples/qwen2_5_7b_polaris/sft_recoveries_fresh_lr2e4.yaml
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/eval_recovery_sft_fresh_lr2e4.yaml
```

Start the 100,000-step GRPO run from that clean SFT adapter with 128 problems
and eight rollouts per problem (1,024 rollouts per optimizer step). It uses a
global training batch of 128, microbatch size one, learning rate `5e-5`, and no
KL loss:

```bash
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_100k_fresh_sft_lr2e4.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_100k_fresh_sft_lr2e4.yaml
```

The run evaluates AIME 2024 every 100 policy steps, retains five recent policy
exports, two checkpoints, five consumed rollout batches, two evaluation sets,
and bounded sampled rollouts for reward-hacking inspection. The Polaris RL
configs require FlashAttention 2 in the trainer and use chunked selected-token
log-probability computation. Preflight fails before launch if `flash-attn` is
not installed; the trainer also verifies that the extension imports before
loading the model.

For a two-GPU async restart from the latest policy produced by that run, use:

```bash
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step361.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step361.yaml
```

This restart initializes from policy step 361 with a fresh optimizer because
its optimizer batch changes from 128 to 32. GPU 0 trains while GPU 1 serves
vLLM. Four eight-example chunks per optimizer step bound the effective policy
lag to three steps. At most eight chunks may be pending, which keeps enough
generation in flight to saturate the inference GPU without exceeding ordinary
rollout-client resource limits. Policy and consumed-rollout retention remain
larger than the staleness window.

To continue from that run's policy step 24 without sequence packing, use the
unpacked microbatch-eight restart. Ordinary padded batches allow FlashAttention
2 to process eight rows concurrently and use the additional memory on a 96 GB
trainer GPU:

```bash
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step385_unpacked_mb8.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step385_unpacked_mb8.yaml
```

The tuned 96 GB continuation increases the unpacked microbatch to 16 and starts
from the microbatch-eight run's policy step 8:

```bash
uv run python -m wavelet debug preflight \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step393_unpacked_mb16.yaml --json
uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step393_unpacked_mb16.yaml
```

Microbatch 16 reached 97% reserved memory without expandable allocator
segments. Enabling expandable CUDA segments removed that fragmentation and the
same microbatch completed 145 policy steps. Streaming optimizes 32 problems
with eight rollouts each. The Polaris verifier uses a five-second
symbolic-comparison timeout and replaces its process pool after a hard timeout
so a poisoned worker cannot stall all later rewards:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step402_unpacked_mb16_expandable.yaml
```

After policy step 145 of that run, use the step-547 continuation config. It
keeps the same trainer shape and includes the resilient five-second verifier:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step547_unpacked_mb16.yaml
```

If a later policy makes mixed-correctness groups sparse, continue from policy
step 6 of that run with the wider 64-attempt zero-advantage filtering window:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m wavelet rl \
  @ examples/qwen2_5_7b_polaris/rl_100k_async_2gpu_b32_from_step553_unpacked_mb16.yaml
```

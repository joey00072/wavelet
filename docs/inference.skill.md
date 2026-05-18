# Inference Diagnostics Runbook

Use this human- and agent-readable guide when debugging Wavelet inference without
starting the trainer or orchestrator. The goal is to isolate model serving,
policy loading, throughput, token accounting, and logprob behavior before
involving the full RL loop.

Related docs: [documentation index](index.md).

## First Principles

- Debug inference as a standalone system first.
- Separate server health, policy loading, generation, token/logprob attachment,
  and throughput into independent checks.
- Prefer JSON output from diagnostics commands so another agent can compare runs.
- Keep trainer and orchestrator stopped unless the question explicitly requires
  queue or optimizer behavior.
- Never infer RL failure from reward alone until inference logprobs, completion
  lengths, policy step, and token throughput are visible.

## Commands

Inspect the resolved inference configuration:

```bash
uv run python -m wavelet debug inference inspect @ examples/wordle/rl.yaml --json
```

Check live HTTP inference servers:

```bash
uv run python -m wavelet debug inference health @ examples/wordle/rl.yaml --json
```

Run a small end-to-end inference smoke test:

```bash
uv run python -m wavelet debug inference smoke @ examples/wordle/rl.yaml --count 4 --json
```

Run a throughput probe:

```bash
uv run python -m wavelet debug inference benchmark @ examples/wordle/rl.yaml --count 32 --warmup 4 --repeats 3 --json
```

Run a live continuous-batching probe against a running OpenAI-compatible vLLM
server:

```bash
uv run python -m wavelet debug inference continuous-batch @ examples/wordle/rl.yaml --count 96 --concurrency 48 --stagger-ms 100 --max-completion-tokens 512 --json
```

Load a specific exported policy before probing:

```bash
uv run python -m wavelet debug inference smoke @ path/to/rl.yaml --policy-dir path/to/policies/step-000010 --policy-step 10 --json
```

## Live Server Endpoints

The vLLM servers expose these debug endpoints:

- `GET /health`: basic liveness, loaded policy step, and sleep state.
- `GET /debug/state`: resolved inference config plus runtime policy state.

Use the configured `inference.http.host` and `inference.http.port` or the full
`inference.http.ports` list when multiple replicas are running.

## What To Check

- `inference.mode`: `vllm_http` probes a running server; `vllm` boots local vLLM.
- `server_backend`: `openai` should expose `/v1/chat/completions/tokens`.
- `policy_step`: must match the expected trainer export after policy load.
- `policy_adapter_path` or `policy_weight_path`: must point at the intended
  policy snapshot, not an old run.
- `records_with_inference_logprobs`: should equal `records` for RL generation.
- `records_with_loss_mask`: should equal `records` for trainable rollouts.
- `model_input_tokens_per_second`: shows model-side token processing throughput.
- `trainable_tokens_per_second`: useful for comparing rollout configurations.
- `mean_completion_tokens`: catches accidental completion caps or runaway output.
- `max_observed_concurrency`: proves requests overlapped at the client side.
- `per_rank_requests`: should be balanced across DP ranks for HTTP vLLM pools.
- `tensor_parallel_size`, `data_parallel_size`, and CUDA device groups: must match
  the intended placement before running expensive tests.

## Agent Workflow

1. Run `inspect --json` and confirm the model, mode, backend, sampling, LoRA, and
   policy transfer settings.
2. If using HTTP serving, run `health --json` and inspect every replica.
3. Run `smoke --count 2 --json` to verify completions, logprobs, and loss masks.
4. Run `benchmark --count 16` or larger only after smoke passes.
5. Run `continuous-batch` when checking vLLM scheduling, DP rank routing, or
   whether a running server stays busy under staggered arrivals.
6. Compare metrics across configs by keeping `count`, `prompt`, `warmup`, and
   `repeats` fixed.
7. Only start trainer/orchestrator after inference proves it can generate,
   attach logprobs, and load the intended policy.

## Failure Patterns

- Missing logprobs usually means the server request path or vLLM logprob settings
  are wrong.
- Zero trainable tokens means generation returned no completion tokens or the loss
  mask was not attached.
- Correct health but failed smoke usually points to tokenizer/chat-template,
  endpoint, max-length, or sampling issues.
- Correct smoke but bad benchmark usually points to batching, TP/DP placement,
  generation length, or HTTP replica distribution.
- Good continuous-batch overlap but low GPU utilization usually means the request
  wave is too small, generations are too short, or DP rank routing is unbalanced.
- Policy step drift means the inference process is serving an old policy or the
  policy load path was skipped.

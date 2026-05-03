# PrimeRL Adoption Notes

This tracks changes from the PrimeRL `95af011` update that are relevant to
Wavelet.

## Adopted

- Optional `flash-attn` extra for trainer-side packed varlen FlashAttention:
  `uv sync --extra flash-attn`.
- PrimeRL-style correctness-gated length penalty for group-reward advantages:
  `orchestrator.length_penalty.type: tokens` or `turns`.
- Weighted token length cost:
  `completion_weight * completion_tokens + tool_response_weight * tool_response_tokens`.
- Tool-response token accounting from rollout metrics whose key ends with
  `total_tool_response_tokens`.
- Silent migration for legacy config aliases.

## Deferred

- PrimeRL now targets a newer vLLM line. Wavelet is intentionally not bumping the
  lower bound yet because policy hot-load, LoRA adapter routing, and OpenAI batch
  endpoints need a separate compatibility run.

Minimum vLLM bump checklist:

- Run reverse-text RL for at least 100 steps with policy export every step.
- Verify `/load_policy` latency does not regress.
- Verify continuous batching metrics remain nonzero during async generation.
- Verify `chat/completions/tokens` returns token ids and logprobs.
- Verify multi-replica `inference.http.ports` routing still works.

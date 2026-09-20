# Reverse-text learning diagnostic

`rl_full_diagnostic.yaml` is a small full-parameter RL control using
`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`. It uses 8 prompts × 16 rollouts,
128-token completions, IPO loss, FP32 optimization with BF16 FSDP computation,
and 100 updates. Evaluation runs before training, every ten updates, and at
the end. GPU 0 serves inference and GPU 1 trains; check that both are free.

The local environment matches the canonical reverse-text task's system prompt,
dataset, tagged-answer extraction, and `SequenceMatcher` partial-credit reward.
Evaluation uses the same task distribution as training: this is a learning
diagnostic, not a held-out generalization benchmark. The starting checkpoint
has already received supervised training for reversal.

From the repository root, with the `verifiers` extra installed:

```bash
export PYTHONPATH="$PWD/examples/reverse_text${PYTHONPATH:+:$PYTHONPATH}"
uv run python examples/reverse_text/prepare_rl_data.py \
  --env-id reverse-text-diagnostic --examples 1000 \
  --output outputs/reverse_text_diagnostic/rl_train.jsonl
uv run python -m wavelet debug preflight @ examples/reverse_text/rl_full_diagnostic.yaml --json
uv run python -m wavelet rl @ examples/reverse_text/rl_full_diagnostic.yaml
```

W&B is enabled when credentials are available. For an offline smoke, override
`--monitor.wandb.enabled false`. Use a fresh output directory for each retry.
The run's baseline/final evaluations, rollout failures, policy versions, and
checkpoint teardown all matter; rising training reward alone is insufficient.

When substituting an ordinary Qwen3 model, disable thinking in **both** training
and evaluation sampling if retaining the 128-token completion budget:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

Put that block under both `inference.sampling` and `eval.sampling`. A renderer
setting on a separate reference trainer does not automatically configure its
chat-based evaluation requests. Validate prompt-token equivalence when comparing
implementations. The existing `rl.yaml` remains the LoRA example; it is a
different optimization recipe from this full-parameter control.

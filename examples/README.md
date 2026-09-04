# Examples

Wavelet examples cover the local RL, SFT, verifier, and multi-GPU launch paths
that are expected to run on this host.

The FSDP examples keep the default `fsdp.impl: fsdp1` compatibility path.
FSDP2 is opt-in; copy an example and set `impl: fsdp2` when following the
[migration guide](../docs/fsdp2_migration.md). FSDP2-only reshard controls are
intentionally omitted from the baseline examples.
Model configs use `matmul_precision: high` by default. Set it to `highest` for
full FP32 matrix multiplication, including on ROCm where reduced float32
precision can be unsuitable for large-vocabulary softmax calculations.
With `attn_implementation: auto`, Hopper (SM90) GPUs use FlashAttention 3 when
its package is installed; other architectures and Hopper environments without
that package fall back to FlashAttention 2 or SDPA. An explicit
`flash_attention_3` setting fails fast unless both Hopper and the extension are
available.
Activation checkpointing is configured as a block with `mode`, `freq`, and
optional selective operator `targets`; set `activation_checkpointing: null` to
disable it. The default full mode checkpoints every decoder layer.
SFT YAML files do not select a process count. Launch multi-GPU SFT examples
with:

```bash
uv run torchrun --standalone --nproc-per-node=N \
  -m wavelet sft @ <config>.yaml
```

| Example | Status | Notes |
| --- | --- | --- |
| `alphabet_sort` | working | 4B LoRA RL. |
| `equation_builder` | working | Local 3–5 number `+`/`-` environment; 0.6B LoRA plus 7B QLoRA smoke and BF16 LoRA long-run configs with rollout auditing. |
| `reverse_text` | working | 0.6B LoRA RL plus SFT with periodic validation loss; includes a 2-GPU INT4 QLoRA 4B experiment. |
| `moe_reverse_text` | working | Qwen3 MoE INT4 QLoRA SFT-to-RL smoke path on two GPUs. |
| `qwen4b_math` | working | Single-node 4B math adaptation, plus a 2-GPU INT4 QLoRA smoke config. |
| `qwen2_5_7b_polaris` | runnable | 7B LoRA GRPO on filtered Polaris with strict tags, diverse group sampling, a zero-step AIME 2024 baseline, and held-out 100-step evals. |
| `hendrycks_sanity` | runnable | 1.5B math sanity config. |
| `wiki_search` | runnable with env deps | Requires `wiki-search` environment setup. |
| `wordle` | runnable after external env install | Requires the Wordle environment installed outside `uv sync`. |
| `qwen30b_math` | adapted | Single-node prototype constrained to supported DP/TP dimensions. |
| `qwen30b_swe` | adapted | Requires SWE environment; CP/EP remain disabled. |
| `multinode` | adapted | Single-node prototype of a multi-node example. |
| `intellect_3_1` | config only | Large-model workload using AdamW; CP/EP remain disabled. |
| `minimax_m2_5_swe` | config only | Large-model workload; CP/EP remain disabled. |
| `glm5_pd_disag` | config only | Disaggregated/MoE workload with EP disabled. |

Use the shared verifier data helper for environment-backed RL examples:

```bash
uv run python examples/prepare_verifier_rl_data.py \
  --env-id math-env \
  --output outputs/hendrycks_sanity_data/rl_train.jsonl \
  --examples 512 \
  --env-arg dataset_name=PrimeIntellect/Hendrycks-Math \
  --env-arg dataset_subset=default
```

Some verifier packages are Python 3.12-only. Use `uv run --python 3.12
--extra envs ...` for the examples that require `wiki-search`, `code-env`,
`logic-env`, `mini-swe-agent-plus`, or `deepdive`.

The canonical named-algorithm example is `examples/reverse_text/rl.yaml`. Its
`algo` block can be changed to `type: max_rl` to use mean-normalized MaxRL
advantages while keeping the same rollout and trainer path. See
[RL algorithms](../docs/algorithms.md) for the full configuration contract.

Environment-specific blockers found locally:

- `wiki_search`: imports on Python 3.12, but environment construction requires
  `OPENAI_API_KEY`.
- `deepdive`: imports on Python 3.12, but environment construction requires
  `SERPER_API_KEY`.
- `wordle`: the package is not included in the default verifier extras; install
  the environment package externally before running.

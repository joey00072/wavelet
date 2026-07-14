# Examples

Wavelet examples cover the local RL, SFT, verifier, and multi-GPU launch paths
that are expected to run on this host.

| Example | Status | Notes |
| --- | --- | --- |
| `alphabet_sort` | working | 4B LoRA RL. |
| `reverse_text` | working | 0.6B LoRA RL with explicit GRPO plus SFT config; includes a 2-GPU INT4 QLoRA 4B experiment. |
| `moe_reverse_text` | working | Qwen3 MoE INT4 QLoRA SFT-to-RL smoke path on two GPUs. |
| `qwen4b_math` | working | Single-node 4B math adaptation, plus a 2-GPU INT4 QLoRA smoke config. |
| `hendrycks_sanity` | runnable | 1.5B math sanity config. |
| `wiki_search` | runnable with env deps | Requires `wiki-search` environment setup. |
| `wordle` | runnable after external env install | Requires the Wordle environment installed outside `uv sync`. |
| `qwen30b_math` | adapted | Single-node prototype of a larger multi-node config. |
| `qwen30b_swe` | adapted | Requires SWE environment and reduced sequence length for this host. |
| `multinode` | adapted | Single-node prototype of a multi-node example. |
| `intellect_3_1` | config only | Production-scale multi-node/MoE workload. |
| `minimax_m2_5_swe` | config only | Production-scale multi-node/MoE workload. |
| `glm5_pd_disag` | config only | Production-scale disaggregated/MoE workload. |

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

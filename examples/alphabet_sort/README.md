# Alphabet Sort

This example ports the Prime-RL Alphabet Sort setup to Wavelet's RL path. It
trains `Qwen/Qwen3-4B-Instruct-2507` with LoRA rank 32 / alpha 64 to maintain a
cumulative alphabetically sorted list across turns.

This uses the same Prime verifier environment shape as Prime-RL: Wavelet loads
`alphabet-sort` through `verifiers`, runs multi-turn rollouts against an
OpenAI-compatible vLLM endpoint, consumes verifier rewards, and trains on the
model-generated assistant turns.

The verifier asks the model to:

- sort by first or last name, chosen per episode
- maintain the prior sorted list across turns
- tag only names introduced in the current turn with `// new name!`
- return the list inside `<alphabetical_sorted>` or
  `<combined_alphabetical_sorted>` tags

Install the verifier extras:

```bash
uv sync --extra verifiers --extra envs
```

Generate verifier examples:

```bash
uv run python examples/alphabet_sort/prepare_rl_data.py
```

Each row stores one verifier dataset example. During RL, Wavelet calls
`wavelet.orchestrator.verifiers:generate_rollouts`, which runs
`orchestrator.rollouts_per_example` verifier rollouts per example.

Validate the configuration without starting training:

```bash
uv run python -m wavelet rl @ examples/alphabet_sort/rl.yaml --dry_run true
```

Run training on a GPU host:

```bash
uv run python -m wavelet rl @ examples/alphabet_sort/rl.yaml
```

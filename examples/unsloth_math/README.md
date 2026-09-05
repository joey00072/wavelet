# Unsloth Qwen3 Math SFT + RL

Wavelet-native version of the Unsloth Qwen3-4B math reasoning example.

It does a short SFT formatting warmup on `unsloth/OpenMathReasoning-mini`, then
RL on `open-r1/DAPO-Math-17k-Processed` with the same custom tags:

- `<start_working_out>`
- `<end_working_out>`
- `<SOLUTION>`
- `</SOLUTION>`

The configs provide bounded example runs:

- SFT: 100 steps, `3e-5` learning rate
- RL: 1,000 steps, `6e-5` learning rate

Prepare data:

```bash
uv run python examples/unsloth_math/prepare_data.py
```

Run SFT:

```bash
uv run python -m wavelet sft @ examples/unsloth_math/sft.yaml
```

Run RL:

```bash
uv run python -m wavelet rl @ examples/unsloth_math/rl.yaml
```

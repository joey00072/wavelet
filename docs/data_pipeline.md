# Data Pipeline

Wavelet keeps source loading, record normalization, token preparation, packing,
and batch collation as separate stages. This makes malformed data fail near the
boundary that owns it and keeps trainer code independent of file formats.

## SFT Flow

`wavelet.data.loading` reads local JSON/JSONL, Hugging Face datasets, or fake
data and normalizes each row into an `Example`. `wavelet.data.tokenization`
then applies the chat template and constructs token-aligned loss masks.
`wavelet.data.dataset` owns iteration and SFT packing.

## RL Flow

RL records use `RLExample` from `wavelet.data.rl_types`. The stages are:

1. `load_data_payloads` reads and mixes raw rows.
2. `load_rl_records` normalizes RL-specific reward, advantage, logprob, and
   metadata fields.
3. `prepare_rl_sample` tokenizes message rows or validates pre-tokenized rows.
4. `wavelet.data.rl_packing` packs prepared samples and equalizes distributed
   bin counts with explicit zero-loss samples.
5. `collate_rl_batch` pads a local micro-batch and aligns trainable-token value
   streams with the full token sequence.

Existing imports from `wavelet.data.rl_dataset` remain supported. New code
should import serialization types from `wavelet.data.rl_types`, collation from
`wavelet.data.rl_collation`, and packing helpers from
`wavelet.data.rl_packing` when it needs those focused contracts directly.

## Pretokenized RL Rows

A pre-tokenized row must provide `input_ids`, `target_ids`, and `loss_mask` with
matching lengths. Token-level `advantage`, `inference_logprobs`,
`teacher_logprobs`, and `temperature` values align only with `true` entries in
`loss_mask`; they do not include values for masked tokens.

Normal rows with no trainable tokens are skipped. Internally marked dummy and
filtered rollout rows are retained so distributed batch counts and rollout
metrics stay correct, while their empty loss mask prevents optimizer impact.

## Debugging

Use the trainer-input diagnostic before launching a trainer against a saved RL
batch:

```bash
uv run python -m wavelet debug trainer inspect \
  --rollout-path outputs/my_run/rollouts/step-000000/rollouts.jsonl --json \
  @ examples/reverse_text/rl.yaml
```

The diagnostic checks token-stream lengths and trainable-token value alignment.
For local code changes, the fastest focused verification is:

```bash
uv run pytest tests/test_rl_dataset.py tests/test_tokenization_alignment.py
```

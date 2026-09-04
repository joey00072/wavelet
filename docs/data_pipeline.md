# Data Pipeline

Wavelet keeps source loading, record normalization, token preparation, packing,
and batch collation as separate stages. This makes malformed data fail near the
boundary that owns it and keeps trainer code independent of file formats.

## SFT Flow

`wavelet.data.sft` reads local JSON/JSONL, Hugging Face datasets, or fake data,
normalizes each row into an `Example`, applies the chat template, constructs
token-aligned loss masks, and owns SFT iteration and collation.

An optional SFT `val` block uses the same loading, tokenization, packing, and
collation path with its own `data` settings. The trainer evaluates one finite
validation epoch under `torch.no_grad()`, aggregates a token-weighted loss
across ranks, and logs `val/loss`. Set `val.eval_on_start: true` for a step-0
baseline; `val.interval` controls evaluation after completed optimizer steps.
The validation dataset is loaded once and a fresh finite dataloader is built for
each evaluation, so repeated metrics cover the same configured records without
advancing the training cursor.

## RL Flow

RL records use `RLExample` from `wavelet.data.rl`. The stages are:

1. `load_data_payloads` reads and mixes raw rows.
2. `load_rl_records` normalizes RL-specific reward, advantage, logprob, and
   metadata fields.
3. `prepare_rl_sample` tokenizes message rows or validates pre-tokenized rows.
4. RL packing helpers pack prepared samples and pad the bin count to a
   multiple of `data_world_size * micro_batch_size` with explicit zero-loss
   samples, so every rank's epoch splits into whole micro-batches and the last
   micro-batch never pulls bins from the next epoch.
5. `collate_rl_batch` pads a local micro-batch and aligns trainable-token value
   streams with the full token sequence.

Token-level streams (`advantage` lists, `inference_logprobs`, `teacher_logprobs`,
`temperature` lists) must match the trainable token count exactly; a longer
stream is only accepted when the sample was cut at `seq_len`, in which case the
aligned prefix is kept. RL `data.num_workers` is limited to 0 or 1 because rows
are pretokenized and extra workers would split rows and packed bins differently
from the per-rank micro-batch count the trainer expects. SFT messages with
`content: null` are treated as empty strings, and a per-message
`step_loss_mask: 0` is honored regardless of whether the chat template supports
assistant token masks.

Existing imports from `wavelet.data.rl_dataset`, `rl_types`, `rl_collation`, and
`rl_packing` remain supported as compatibility wrappers. New code should import
the canonical RL surface from `wavelet.data.rl`; use `wavelet.data.sft` for SFT
records, tokenization, collation, and datasets.

## Pretokenized RL Rows

A pre-tokenized row must provide `input_ids`, `target_ids`, and `loss_mask` with
matching lengths. Token-level `advantage`, `inference_logprobs`,
`teacher_logprobs`, and `temperature` values align only with `true` entries in
`loss_mask`; they do not include values for masked tokens.

Pretokenized rows are never retokenized from their chat messages, because the
sampled `inference_logprobs` belong to the original token stream. A row whose
trainable tokens all fall beyond `seq_len` is kept as a zero-loss row (with a
warning) rather than skipped: skipping would pull the next epoch's row into the
current optimizer batch and duplicate rollouts. Internally marked dummy and
filtered rollout rows are retained the same way so distributed batch counts and
rollout metrics stay correct, while their empty loss mask prevents optimizer
impact. Only rows that were actually cut at `seq_len` may carry value streams
longer than their remaining trainable tokens.

Packing never places rows with and without `inference_logprobs` (or
`teacher_logprobs`) in the same bin, since a merged bin can only carry a stream
for all of its rows or none. `data.pad_to_multiple_of` must divide
`data.seq_len` so padded bins never exceed the configured length.

For chat data, the generation prompt is rendered whenever the next message is an
assistant turn, so assistant headers are never trainable regardless of whether a
system, user, tool, or assistant message precedes them.

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

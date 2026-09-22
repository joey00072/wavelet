# Model and multimodal training

Text models use Hugging Face `AutoModelForCausalLM`. Setting `model.vlm: {}`
selects `AutoModelForImageTextToText` and requires an `AutoProcessor` with an
image or video processor. The default vision tower path is `model.visual`;
set `model.vlm.vision_encoder_attr` for another architecture. Its parameters
are frozen by default; set `freeze_vision_encoder: false` to train them.
The processor is saved beside the exported model or adapter.

Use structured message content with `type: image` / `image: <local path>` and
`type: text` / `text: <text>` blocks. The processor renders each conversation
prefix, expands image tokens, and aligns role-based loss masks with that exact
token stream. Multimodal sequence truncation fails explicitly instead of
cutting through image tokens. Use SFT `data.pack_function: pad` or RL
`data.pack_sequences: false`; concatenative multimodal packing is unsupported.
Batches must supply consistent processor fields. Image patches and grid data
are concatenated; multimodal token types are padded with the input sequence.

Verifier image rollouts snapshot each `image_url` input to a PNG data URL before
inference. The same bytes feed the processor and the server. Authoritative
response prompt IDs must match the processor's expanded prompt; tensors are
attached to that response's trajectory step and propagated to the matching
merged sample. The queue contains frozen media and processor tensors, so later
changes to an image file or URL cannot change training pixels. Online capture
currently supports image/text parts without tool schemas; it rejects audio,
video and tool schemas explicitly. Video processor timing metadata is preserved
for offline training, but live video capture is not implemented.

Pretokenized RL records carrying `mm_kwargs` must also contain
`metadata.multimodal_input_ids` equal to their input token IDs. Without captured
tensors, the processor rerenders messages and requires exact equality with the
sampled token stream. Use immutable media for that offline path. Tiny local
Qwen3-VL tests cover two-turn and branching verifier trajectories through JSON
queue serialization and backward, processor expansion, SFT/RL training,
LoRA vision freezing, and processor/model export reload without downloads.

`model.experts_implementation` can select Hugging Face `eager` or `grouped_mm`
expert execution, including Wavelet expert-parallel dispatch. Local expert
shards reuse Transformers' eager or grouped implementation; Wavelet owns token
dispatch and collective communication. The installed Transformers implementation and device must
support the requested model and backend. Wavelet recognizes Qwen3.5 packed
experts and Nemotron-H ungated expert projections for expert parallel dispatch.

Expert-parallel sharding preserves each parameter's `requires_grad` flag.
LoRA runs keep base experts frozen while propagating gradients through them to
the adapters. After distributed model setup, Wavelet rejects trainable base
parameters in a LoRA run; only adapter parameters and `modules_to_save` may be
trainable. This prevents adapter-only policy exports from silently omitting
base-weight updates.

FSDP expert gradients use the full DP × CP division factor even though the
expert parameters are sharded over `dp_mod_ep`. Expert dispatch sums contributions
from EP ranks, so using only the smaller group's divisor over-scales expert
gradients relative to dense weights. Router metrics disable both the config and
cached model auxiliary-loss coefficient, keeping the requested training objective.

Packed RL batches split bins at complete sample boundaries to fill the trainer's
data ranks and micro-batches before adding zero-loss padding. Splitting preserves
the original token streams and sequence boundaries; unavoidable dummy bins remain
loss-masked. This avoids processing full-length dummy copies when enough real
samples are available. Rank assignment changes, so reproduce a run with a fixed
code revision as well as a fixed seed.
Distributed bins are assigned in descending-token rounds, placing longer bins
on ranks with less accumulated work while preserving equal bin counts. This
balances token load across multiple micro-batches; it is not a model-specific
FLOP estimate and cannot fix imbalance within a single indivisible bin per rank.

The chunked RL output head extracts target logits and accumulates their gradients
with tensor gather/scatter operations. It avoids Python conditionals on GPU
tensors inside the vocabulary loop; probability normalization and loss masks
remain unchanged.

Verifier rollout generation polls for published policies while agent tools are
running. Full-model and LoRA HTTP updates pause the vLLM replicas instead of waiting for
whole episodes; sampled logprobs and start/end policy versions remain attached
to each rollout. Evaluation is settled before either update. NCCL readiness is published only
after all replicas pause, and final evaluation receives intermediate exports
in order because the trainer blocks until each NCCL export is received.
Shutdown waits for an in-progress transfer before closing its receiver.
NCCL startup transfers version 0 as well, validating the actual weight channel
before the first optimizer update instead of relying on independently loaded weights.

`policy_transfer.nccl_dtype` defaults to `bfloat16`: parameter shards are cast
before gathering for transfer, while trainer master weights remain unchanged.
Buffers retain their original dtype. Models can declare sensitive parameters
through `keep_in_fp32_for_weight_transfer(name)`; those transfer as FP32.
Set `nccl_dtype: model` to retain every parameter's original dtype, or
`nccl_dtype: float32` when the inference implementation needs FP32 parameters.
Reconstructed, nonpersistent buffers such as rotary frequencies are excluded
from the wire, matching checkpoint semantics and the inference weight loader.
Training wrapper segments are removed before Hugging Face weight conversion,
so activation checkpointing does not change inference parameter names.

For verifier runs, `orchestrator.refill_zero_advantage: false` counts complete
valid groups toward `examples_per_step` before zero-advantage filtering.
With `filter_zero_advantage: true`, zero-advantage records remain in reward
metrics but are excluded from training. This gives a bounded problem batch
instead of refilling every uninformative group. Entirely uninformative batches
are retried under `zero_advantage_max_retries`; they do not advance the optimizer.
The default `refill_zero_advantage: true` retains effective-group batching.
Offline and streaming RL progress accumulate globally reduced token/sample
counts once per optimizer update, including unequal packed rank lengths and
ranks containing only padding.
Checkpoint logs report the pending upload path and when its stable marker is
written; a final Hugging Face export can take additional time afterward.

Context parallel SFT and RL require SDPA, token normalization, and a sequence
length divisible by `2 * fsdp.cp`. Losses are normalized over the explicit
`dp_cp` process group, so ranks with unequal supervised-token counts retain the
same objective as a non-CP batch. The experimental ring backend currently
rejects explicit 4D attention bias/masks; packed or otherwise bias-dependent
CP batches (including padded 2D masks) fail early with a configuration error. VLM context parallelism is
also unsupported until the model-specific multimodal position and image
feature sharding path is enabled.

CP gradient accumulation weights microbatches by supervised-token count across
the optimizer batch, including empty local shards. CPU two-rank tests exercise
actual trainer loss and updates; CUDA ring-attention execution requires the
GPU regression suite.

Image snapshots accept regular local files, data URLs and HTTP(S) URLs. Inputs
and canonical PNG snapshots are limited to 32 MiB and 16 megapixels. Remote
fetches use a 10-second network timeout and a 30-second streaming deadline;
a cancelled rollout can wait for its bounded download worker to exit. VLM
capture requires local verifier execution; server-mode environments and missing
generation captures fail explicitly. A change in captured media splits a
trajectory into separate training samples instead of reusing earlier
logprobs under different pixels.

Liger dispatch uses the checkpoint's actual `model_type`, rather than its path
or name. The supported Liger families are Qwen3, Qwen2, Llama and Mistral;
other architectures, including Qwen3.5 and VLMs, require
`loss_impl: torch` and fail early if a Liger mode is requested.

RL loss diagnostics accumulate as detached device scalars during gradient
accumulation. At the optimizer boundary the trainer reads them together, then
applies the existing mean/sum/min/max and cross-rank reductions. This avoids
per-diagnostic device synchronization on each microbatch without retaining the
autograd graph or changing loss normalization. Rollout statistics and finite-loss
checks retain their existing timing. Supplied component-weight tensors are reused;
default RL/CE/reference-KL weights are allocated only when absent.

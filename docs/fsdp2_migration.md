# FSDP2 Migration

Wavelet supports an opt-in FSDP2 execution path while FSDP1 remains the
default for one compatibility release. Select it explicitly:

```yaml
fsdp:
  enabled: true
  impl: fsdp2
  reshard_after_forward: true
```

Run the normal distributed entrypoint. For example, SFT on two local workers
uses:

```bash
uv run torchrun --standalone --nproc-per-node=2 \
  -m wavelet sft @ <config>.yaml
```

`reshard_after_forward` is an FSDP2-only control. Setting it to `false` keeps
unsharded parameters resident after forward, trading memory for less gather
work before backward. Config validation rejects that setting with FSDP1 so it
cannot be silently ignored.

## Wrapper design

The trainer applies model transforms in this order:

1. Load the Hugging Face model and apply LoRA or optional kernel transforms.
2. Apply activation checkpoint placement and optional `torch.compile`.
3. Call `fully_shard` on each transformer block named by the model's
   `_no_split_modules` contract.
4. Call `fully_shard` on the root model to cover embeddings, the language-model
   head, and any parameters outside transformer blocks.

FSDP2 receives the existing `hsdp` device mesh. With `dp_replicate: 1`, that is
the flattened `dp_shard_cp` mesh; with `dp_replicate > 1`, it is a two-dimensional
replicate-by-shard mesh. `MixedPrecisionPolicy` follows the same dtype rules as
the FSDP1 path, and CPU offload maps to `CPUOffloadPolicy`.

Checkpoint save and load use PyTorch Distributed Checkpoint's unified
`get_state_dict` and `set_state_dict` APIs for both implementations. Full policy
exports use a rank-zero, CPU-offloaded full state dict. Lightweight LoRA policy
exports ask DCP for trainable parameters only, so frozen base-model parameters
are not gathered for an adapter snapshot.

## Compatibility boundary

| Feature | FSDP1 | FSDP2 opt-in |
| --- | --- | --- |
| Full-shard data parallel | Supported | Supported; two-rank CPU round trip tested |
| HSDP (`dp_replicate > 1`) | Supported | Uses the existing HSDP mesh; GPU validation pending |
| DCP sync/async checkpoints | Supported | Unified state-dict path; sync round trip tested |
| Lightweight LoRA export | Supported | Trainable-only DCP gather tested |
| Hugging Face tensor parallel plus DP sharding | Supported where the model has a TP plan | Wrapper composition is present; GPU validation pending |
| QLoRA | Replicated DDP only | FSDP remains rejected by preflight |
| Context or expert parallelism | Rejected | Rejected until their model kernels are implemented |
| `colocate_sleep` CPU movement | Supported FSDP1 path | Validation pending |

The first FSDP2 phase deliberately retains the current Hugging Face loading
path: every rank materializes pretrained weights on CPU before sharding. The
next migration phase is meta-device construction followed by direct loading of
Hugging Face safetensors into DCP shards. Until that lands, `model.meta_device_init`
does not provide a zero-materialization FSDP2 load path.

## Validation

The CPU integration test exercises per-block wrapping, forward/backward, an
optimizer step, DCP save/load restoration of model and optimizer shards, and a
lightweight LoRA safetensors export:

```bash
uv run pytest tests/integration/test_fsdp2_checkpoint.py -q
```

Before moving a production configuration from FSDP1, also run preflight and a
short GPU smoke job with the exact HSDP, TP, checkpoint mode, and colocated
settings used by the full run. Keep the output directory clean between the
FSDP1 and FSDP2 trials so stale policies or checkpoints cannot mask a mismatch.

# Wavelet benchmarks

Run a short SFT or RL configuration into a new, empty run directory and save a
hardware-keyed JSON result:

```bash
uv run wavelet benchmark run sft examples/reverse_text/sft.yaml \
  outputs/benchmarks/reverse-text-run \
  benchmarks/results/reverse-text-a100.json
```

The first optimizer step is excluded as warmup by default. The harness merges
multiple monitor rows for each step and summarizes throughput, MFU, step time,
and peak memory when those metrics are present. The result identity includes a
hash of the source config, the Torch version, and accelerator names so results
from unlike workloads or hardware cannot be compared accidentally.

Promote a representative result to `benchmarks/baselines/` after reviewing its
run logs and repeatability. Compare a new result with it either during the run
or afterward:

```bash
uv run wavelet benchmark compare CURRENT.json BASELINE.json \
  --regression-threshold 0.05
```

The command exits with status 2 when any higher-is-better metric drops or any
lower-is-better metric rises beyond the threshold. A missing baseline metric is
also a regression. Run directories and generated result JSON files are artifacts
and should not be committed; only reviewed baseline JSON belongs in
`benchmarks/baselines/`.

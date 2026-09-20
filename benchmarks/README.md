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

The GPU test workflow runs nightly and on manual dispatch on a self-hosted
runner with `linux` and `gpu` labels. The GPU benchmark workflow runs weekly
and on manual dispatch using `benchmarks/configs/sft.yaml`: 20 full-parameter
SFT steps on Qwen3-0.6B with synthetic data, excluding five warmup steps.
The runner needs a CUDA GPU with sufficient free memory and access to the model.
Both workflows retain diagnostics as GitHub artifacts, including on failure.
Scheduled workflows execute the default branch; a runner must be provisioned
before these checks can run.

To enable regression gating, review repeated benchmark results on the chosen
runner hardware and commit one as `benchmarks/baselines/sft-ci.json`. Until that
baseline exists, the workflow records measurements without claiming performance
parity. A baseline from different hardware, Torch, or config fails comparison
and must be remeasured. Do not promote a baseline automatically.

Scheduled GPU jobs are opt-in to avoid consuming scarce GPUs. Set repository
variable `ENABLE_SCHEDULED_GPU_BENCHMARKS=true` to enable the weekly benchmark;
`ENABLE_SCHEDULED_GPU_CHECKS=true` enables nightly GPU tests. Manual dispatch
remains available. Neither variable is enabled by this change, and no GPU run
was launched during implementation.

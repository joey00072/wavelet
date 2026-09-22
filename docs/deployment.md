# Deployment

Wavelet provides a CUDA 13.0 image for SFT and RL runs. The image installs the
locked Python 3.12 environment, including the
matching prebuilt FlashAttention wheel, and retains the CUDA toolkit because
vLLM and attention backends may compile kernels at runtime.

## Build and run the image

Build from the repository root:

```bash
docker build --pull -t wavelet:local .
```

Run preflight against a config included in the image:

```bash
docker run --rm --gpus all \
  -v "$PWD/outputs:/opt/wavelet/outputs" \
  wavelet:local \
  debug preflight @ examples/reverse_text/rl.yaml --json
```

The image entrypoint is `wavelet`, so arguments after the image name are normal
Wavelet subcommands. The container runs as UID and GID 1000 by default. Match a
different host user at build time when the mounted output directory requires
it:

```bash
docker build \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  -t wavelet:local .
```

Mount model and dataset caches instead of putting them in the image. Pass
tokens at runtime; do not copy `.env` or credential files into an image:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -v "$HOME/.cache/huggingface:/home/wavelet/.cache/huggingface" \
  -v "$PWD/outputs:/opt/wavelet/outputs" \
  wavelet:local \
  rl @ examples/reverse_text/rl.yaml
```

The NVIDIA Container Toolkit must be installed on the host for `--gpus all`.
Check GPU visibility before a long run with:

```bash
docker run --rm --gpus all --entrypoint python wavelet:local \
  -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

## Local and Ray launchers

`launcher.backend: local` starts every configured role inside one container.
Give the container all devices named by
`launcher.{trainer,inference}_cuda_visible_devices`, mount a writable
`output_dir`, and stop the outer container with `SIGTERM` so the launcher can
tear down its child processes.

`launcher.backend: ray` submits the same role subprocesses to an already-running
Ray cluster. Ray is intentionally an operator-installed dependency and is not
included in the base Wavelet image. Pin and install the same Ray release in
Wavelet's environment on the head and every worker, then start the head and
workers before launching Wavelet. For example:

```bash
uv pip install 'ray[default]'
ray start --head --port=6379
# On each worker:
ray start --address=ray-head.example:6379
uv run wavelet rl @ run.yaml \
  --launcher.backend ray \
  --launcher.ray-address ray-head.example:6379
```

The Ray workers must see the same repository path, resolved role configs,
model/data inputs, and output directory. Use a shared filesystem or provide an
equivalent `launcher.ray_runtime_env`. The Ray backend does not provision a
cluster, install dependencies, translate GPU IDs between heterogeneous nodes,
or generate scheduler jobs. Verify the resolved commands and paths before the
launch:

```bash
uv run wavelet debug preflight @ run.yaml \
  --launcher.backend ray \
  --launcher.ray-address ray-head.example:6379 \
  --json
```

Role GPU requirements come from the configured trainer and inference device
groups. The launcher does not provision a cluster or create scheduler jobs.

## Native multi-node SLURM

The normal `sft` and `rl` commands submit a job when a config contains both a
multi-node `deployment` block and a `slurm` block. They write the resolved
config, generated `job.sbatch`, submitted job ID, allocation map, and per-role
logs under the run directory. The submission command returns after `sbatch`
accepts the job.

The generated script invokes `python -m wavelet slurm-worker` inside the
allocation; this internal command dispatches through the same CLI as the public
training commands.

Startup prints the allocated trainer/inference hosts, pending endpoints, and
readiness times. OpenAI-compatible inference is ready only after `/health`,
worker `/liveness`, and `/v1/models` checks succeed with the expected model.
This establishes serving readiness; the orchestrator still waits for the
trainer's initial policy export before generating training rollouts. Per-role
logs identify the component to inspect when either phase stalls.

SFT uses one torchrun agent per train node and one trainer process per GPU:

```yaml
deployment:
  type: multi_node
  num_train_nodes: 2
  num_inference_nodes: 0
  gpus_per_node: 8
  trainer_master_port: 29500

slurm:
  job_name: wavelet-sft
  project_dir: /shared/wavelet
  partition: gpu
  account: research
  time_limit: "04:00:00"
  shared_fs: true
  setup_commands:
    - module load cuda
```

Launch it through the same public entrypoint used locally:

```bash
uv run python -m wavelet sft @ examples/multinode/sft.yaml
```

For RL, inference nodes are assigned first from the SLURM allocation and train
nodes follow. Wavelet starts one vLLM replica per inference node, gives each
replica its node's GPUs, publishes the allocated `(host, port)` endpoints to the
rollout client, and starts an elastic multi-node trainer across all train nodes:

```yaml
launcher:
  mode: process
  backend: local

deployment:
  type: multi_node
  num_train_nodes: 2
  num_inference_nodes: 2
  gpus_per_node: 8

slurm:
  job_name: wavelet-rl
  project_dir: /shared/wavelet
  partition: gpu
  shared_fs: true
```

`launcher.trainer_cuda_visible_devices`,
`launcher.inference_cuda_visible_devices`, and
`launcher.trainer_num_processes` remain single-node placement options. SLURM
sets device visibility and the multi-node worker derives trainer process counts
from `deployment.gpus_per_node`.
Preflight validates FSDP against the total allocated trainer GPU count, regardless
of the single-node `launcher.trainer_num_processes` setting.

With online W&B enabled, the SLURM worker gives the trainer and orchestrator one
shared run ID, recorded in `wandb_run_id.txt`. Supply credentials through the job
environment or a private file sourced by `slurm.setup_commands`; do not embed API
keys in YAML. Role environment values are redacted in resolved config artifacts,
so they are not a credential transport across SLURM submission.
The trainer owns shared run metadata. Both trainer and orchestrator flush their
logs without setting final status; the launcher marks success or failure after
joining and cleaning up all roles. This also applies to the local process
launcher and prevents a late checkpoint save from leaving a completed run marked
as crashed. Finalization failures are reported without masking a training error.

Use `dry_run: true` to validate the config and write `job.sbatch` without
calling `sbatch`. Inspect the exact script before allocating GPUs:

```bash
uv run python -m wavelet rl @ examples/multinode/rl.yaml --dry-run
cat outputs/multinode_math_rl/configs/latest/job.sbatch
```

The native backend has these constraints:

- the repository, environment, model/data inputs, and output directory must be
  visible at the same paths on every node (`slurm.shared_fs: true`);
- LoRA split RL uses `policy_transfer.type: filesystem`; full-model runs can
  use `policy_transfer.type: nccl`. The worker resolves a default loopback
  broadcast address to the first trainer host, counts every inference GPU, and
  assigns disjoint NCCL rank ranges to inference replicas. An explicit non-loopback
  `nccl_host` is preserved and must route to the first trainer;
- `tensor_parallel_size * (data_parallel_size_local or data_parallel_size)`
  multiplied by `deployment.inference_replicas_per_node` cannot exceed
  `deployment.gpus_per_node`.

`deployment.inference_replicas_per_node` defaults to one. For two independent
four-GPU servers on an eight-GPU inference node, set it to `2`, set
`inference.vllm.tensor_parallel_size: 4`, and set both `data_parallel_size` and
`data_parallel_size_local` to `1`. SLURM allocates disjoint GPU subsets to the
exclusive server steps; each replica gets its own endpoint and policy-sync rank
range. This differs from one vLLM server with internal data parallelism, whose
MoE workers may share communication across replicas. Measure both layouts for
the intended batch/concurrency, especially when one replica becomes idle.
Multiple replicas also require `slurm.inference_memory_per_replica` (for
example `128G`) so one step cannot reserve the whole node's job memory. Set
`slurm.inference_cpus_per_replica` for each server's CPU needs (default `1`).
The allocation must have enough memory and CPUs for all simultaneous replicas.
Server steps explicitly override the job's GPU-per-node reservation as well as
its GPU-per-task count.

Cluster-specific setup stays declarative. Use the typed partition, account,
QoS, constraint, reservation, node list, CPU, memory, and time fields where
possible. `slurm.setup_commands` is for site modules or container setup, and
`slurm.extra_directives` accepts less common single-line `--...` directives.
Wavelet invokes allocation workers with `uv run --no-sync python` by default so
the job does not mutate a shared environment; override `slurm.python_command`
when the site uses a different environment wrapper.

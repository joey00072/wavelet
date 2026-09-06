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

## Native multi-node SLURM

The normal `sft` and `rl` commands submit a job when a config contains both a
multi-node `deployment` block and a `slurm` block. They write the resolved
config, generated `job.sbatch`, submitted job ID, allocation map, and per-role
logs under the run directory. The submission command returns after `sbatch`
accepts the job.

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

Use `dry_run: true` to validate the config and write `job.sbatch` without
calling `sbatch`. Inspect the exact script before allocating GPUs:

```bash
uv run python -m wavelet rl @ examples/multinode/rl.yaml --dry-run
cat outputs/multinode_math_rl/configs/latest/job.sbatch
```

The initial native backend intentionally has three explicit constraints:

- the repository, environment, model/data inputs, and output directory must be
  visible at the same paths on every node (`slurm.shared_fs: true`);
- split RL uses `policy_transfer.type: filesystem`;
- `tensor_parallel_size * (data_parallel_size_local or data_parallel_size)`
  for one vLLM replica cannot exceed `deployment.gpus_per_node`.

Cluster-specific setup stays declarative. Use the typed partition, account,
QoS, constraint, reservation, node list, CPU, memory, and time fields where
possible. `slurm.setup_commands` is for site modules or container setup, and
`slurm.extra_directives` accepts less common single-line `--...` directives.
Wavelet invokes allocation workers with `uv run --no-sync python` by default so
the job does not mutate a shared environment; override `slurm.python_command`
when the site uses a different environment wrapper.

# RL Algorithms

Wavelet algorithms turn finalized rollouts into per-token training signals.
The same runtime is used for native and Verifiers-backed rollouts, and an
algorithm may be selected globally or per training source.

The top-level `algo` block selects a built-in algorithm or a user-owned Python
file:

| Configuration | Scope | Behavior |
| --- | --- | --- |
| `type: passthrough` | rollout | Keep advantages already present in the data. |
| `type: reward` | rollout | Copy each reward into its missing advantage. |
| `type: grpo` | group | Center rewards within each prompt group. |
| `type: max_rl` | group | Center rewards and divide by the group mean. |
| `type: opd` | rollout | Score policy rollouts under a frozen teacher and train with reverse KL. |
| `file` + `algorithm` | explicit | Load a registered class or factory from a local file. |

Algorithm and length-penalty configs reject unknown keys so misspellings fail
during configuration loading rather than being silently ignored.

## Built-in Algorithms

### Passthrough

Passthrough is the default. It leaves each `RLExample.advantage` unchanged,
which is useful when a dataset or custom rollout function has already assigned
credit.

```yaml
algo:
  type: passthrough
```

### Reward

Reward mode copies `RLExample.reward` to `RLExample.advantage` only when the
advantage is missing. Existing advantages are preserved.

```yaml
algo:
  type: reward
```

### GRPO

GRPO requires a reward on every rollout and centers rewards within each prompt
group. Set `normalize_advantages` to divide centered advantages by their
population standard deviation when that deviation exceeds `epsilon`.

```yaml
algo:
  type: grpo
  normalize_advantages: false
  epsilon: 1.0e-6
```

GRPO can also favor efficient correct rollouts. Token cost is
`completion_weight * completion_tokens + tool_response_weight *
tool_response_tokens`:

```yaml
algo:
  type: grpo
  length_penalty:
    type: tokens
    completion_weight: 1.0
    tool_response_weight: 0.0
```

Use turn count instead:

```yaml
algo:
  type: grpo
  length_penalty:
    type: turns
```

Length cost is not subtracted as a fixed linear penalty. Among rollouts tied
for the best positive reward, the shorter-than-average rollouts receive an
efficiency bonus before the group is centered. If the group has no positive
reward or no usable cost, Wavelet falls back to ordinary reward centering.

### MaxRL

MaxRL assigns `(reward - group_mean) / group_mean`. It is intended for
non-negative, commonly binary rewards. A group with mean reward at or below
zero receives zero advantages.

```yaml
algo:
  type: max_rl
```

### On-policy distillation (OPD)

OPD samples rollouts from the live policy and prefill-scores the exact shifted
token sequence under a frozen teacher. It writes `ref_logprobs` and selects the
`ref_kl` trainer component; it does not assign a scalar advantage.

```yaml
algo:
  type: opd
  teacher:
    name: PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL
    base_url: http://localhost:8001/v1
    api_key_var: OPENAI_API_KEY
    timeout_seconds: 120
```

The teacher is an external vLLM endpoint. Wavelet manages only the trainable
policy. For the reverse-text example, start the teacher separately:

```bash
CUDA_VISIBLE_DEVICES=2 uv run vllm serve \
  PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
  --port 8001 --gpu-memory-utilization 0.5 --enforce-eager
```

OPD calls vLLM's `/inference/v1/generate` token endpoint with
`prompt_logprobs: 1`. The endpoint must return one aligned prompt logprob per
token. Verifier rollouts must use `openai_chat_completions_token` so Wavelet can
prove that teacher, inference, and trainer token streams are identical.

The reverse-KL component uses the sampling-policy importance ratio and the same
one-sided mismatch guard and drift regularizer configured under `loss`. OPD is
currently published as a complete optimizer batch rather than streaming
partial chunks, which lets `rl`, `ce`, and `ref_kl` use independent exact
denominators. Distributed trainers sum each component's token count across the
data-parallel group before scaling the rank-local loss. OPD therefore requires
`data.num_workers: 0`; config validation rejects worker-local dataset state
that cannot prove the upcoming optimizer-batch denominator.

## Mixed per-source algorithms

`orchestrator.train_sources` overrides the global `algo` for named training
sources. Every source may select its own verifier environment, data config,
algorithm, weight, and OPD teacher. All records are packed by the same trainer
and update one student policy and one LoRA adapter.

```yaml
algo:
  type: grpo

orchestrator:
  custom_rollout_function: wavelet.orchestrator.verifiers:generate_rollouts
  verifier_client_type: openai_chat_completions_token
  examples_per_step: 8
  rollouts_per_example: 4
  train_sources:
    - name: reverse-text-grpo
      verifier_env_id: reverse-text
      algo:
        type: grpo
    - name: reverse-text-opd
      verifier_env_id: reverse-text
      algo:
        type: opd
        teacher:
          name: PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL
          base_url: http://localhost:8001/v1
```

`weight` defaults to `1` and controls each source's share of the configured
groups. When there are enough groups, every source receives at least one.
Different OPD sources can point to different teacher endpoints, providing
multi-teacher distillation without multiple students or multiple LoRA
adapters. See `examples/reverse_text/rl_mixed_grpo_opd.yaml` for a complete
debug configuration.

### API and Web UI observability

When the orchestrator state server is enabled, `GET /algorithms` exposes the
resolved source-to-algorithm graph. `GET /state` embeds the same payload under
`algorithms`. Both responses omit teacher credential environment variable names
and values. The payload reports one student, at most one trainable adapter,
unique teachers, teacher endpoint replica counts, source weights, active loss
components, and the latest per-source rollout observations.

The orchestrator emits source-local `batch/source/<name>`,
`reward/source/<name>`, and `fate/source/<name>` metrics. Fate metrics include
reference-logprob coverage and whether RL, CE, or REF-KL was active for each
rollout. The Web UI presents these fields in the overview and annotates compact
rollout samples with their source and active loss streams.

The trainer understands three independently normalized token components:

- `rl_weights`: policy-gradient credit from GRPO, MaxRL, reward, or a custom
  algorithm.
- `ce_weights`: cross-entropy tokens reserved for distillation and ECHO-style
  algorithms.
- `ref_kl_weights`: reverse-KL tokens with aligned `ref_logprobs`, used by OPD.

Components may coexist in one packed micro-batch. A zero weight removes a token
from that component's numerator and denominator; adding OPD tokens therefore
does not reduce the effective GRPO learning rate.

## Custom Algorithm Files

A custom algorithm can live anywhere on the local filesystem. It does not need
to be copied into Wavelet or added to Wavelet's built-in configuration union.

### 1. Write and register the algorithm

Create a Python file such as `/home/user/my_training/algorithms.py`:

```python
from dataclasses import dataclass, replace

from wavelet.data.rl_dataset import RLExample
from wavelet.orchestrator.algorithms import BaseAlgorithm, register_algorithm


@register_algorithm("centered_reward")
@dataclass(frozen=True, slots=True)
class CenteredRewardAlgorithm(BaseAlgorithm):
    scale: float = 1.0

    def score_group(self, records: list[RLExample]) -> list[RLExample]:
        rewards = [float(record.reward) for record in records]
        mean_reward = sum(rewards) / len(rewards)
        return [
            replace(record, advantage=self.scale * (reward - mean_reward))
            for record, reward in zip(records, rewards, strict=True)
        ]
```

`BaseAlgorithm` supplies no-op implementations of both hooks, so the class only
overrides the hook it needs. Using `dataclasses.replace` keeps input records
unchanged and makes the returned state explicit.

Set `action_loss_type` to `"rl"`, `"ce"`, or `"ref_kl"` to select the default
trainer component for every trainable action token returned by the algorithm.
It defaults to `"rl"`. An algorithm may instead set `rl_weights`, `ce_weights`,
and `ref_kl_weights` directly on a record for token-level mixtures; explicit
weights take precedence over the class default. A nonzero `ref_kl` stream must
also attach aligned `ref_logprobs`.

### 2. Select it in YAML

```yaml
algo:
  file: /home/user/my_training/algorithms.py
  algorithm: centered_reward
  scope: group
  kwargs:
    scale: 2.0
  epsilon: 1.0e-6
```

The `file` or `algorithm` field is enough for Wavelet to infer the custom
variant. `type: custom` is accepted but redundant. Relative paths are resolved
from the process working directory; an absolute path is less ambiguous for
distributed launches.

`algorithm` normally refers to the name passed to `register_algorithm`. It may
also name an undecorated class or factory attribute in the file. `kwargs` are
passed directly to that class or factory. For example, a registered factory is
valid too:

```python
@register_algorithm("my_factory")
def build_algorithm(*, scale: float) -> BaseAlgorithm:
    return CenteredRewardAlgorithm(scale=scale)
```

Registration names must be non-empty, must not contain surrounding whitespace,
and must be unique within the file.

### Scope

`scope` is required because scheduling and zero-advantage filtering must be
known before Wavelet executes the user file.

| Scope | Hooks run | Orchestration behavior |
| --- | --- | --- |
| `rollout` | `score_rollout` | Each rollout can be scored independently. |
| `group` | `score_group` | Wavelet waits for complete prompt groups and enables group-level zero-advantage filtering. |
| `both` | `score_rollout`, then `score_group` | Per-rollout signals can be refined using the completed group. |

For `group` scope, Wavelet skips the group hook when every input record already
has an advantage. For `both`, both hooks always run because the group hook may
intentionally refine rollout-local credit.

`epsilon` is the threshold used by Wavelet's zero-advantage group filter. It is
not passed to the custom class; include an identically named entry under
`kwargs` if the algorithm itself needs one.

### Hook contract

Custom hooks are synchronous:

```python
def score_rollout(self, record: RLExample) -> RLExample: ...

def score_group(self, records: list[RLExample]) -> list[RLExample]: ...
```

Relevant `RLExample` fields include:

- `reward`: scalar reward assigned by the reward or verifier path.
- `advantage`: scalar or token-level advantage; custom scoring usually assigns
  a scalar with `dataclasses.replace`.
- `prompt` and `completion`: rendered message lists when available.
- `input_ids`, `target_ids`, and `loss_mask`: pre-tokenized training artifacts
  when available.
- `rl_weights`, `ce_weights`, and `ref_kl_weights`: optional per-token component
  membership. Supplying no component streams retains the legacy all-RL path.
- `ref_logprobs`: frozen-reference logprobs required wherever
  `ref_kl_weights` is non-zero.
- `metadata`: group identity, rollout identity, token counts, tool-response
  token counts, turn count, and rollout provenance when supplied by the source.
- `source`: dataset or environment name.

The runtime validates these invariants:

- `score_rollout` returns one `RLExample`.
- `score_group` returns a `list[RLExample]` with exactly one output per input.
- Group output order corresponds to group input order.
- Wavelet restores the original cross-group record order before writing the
  rollout batch.

Do not mutate input records or change group cardinality. Return replacement
records instead. Raise a specific exception when required fields such as
`reward` or metadata are missing.

### Loading and lifecycle

Wavelet resolves `~`, converts the configured path to an absolute path, executes
the file as a uniquely named Python module, locates the registered class or
factory, passes `kwargs`, and validates both hooks. A custom file may therefore
import normal installed dependencies, including Wavelet itself.

The file must be readable from every process that generates or scores rollouts.
Factories should be lightweight because preflight and rollout workers may
construct them more than once. If present, synchronous `setup()` and `close()`
lifecycle hooks run around each scoring call. Scoring hooks remain synchronous;
async custom hooks are not yet part of this interface.

Configuration executes only the explicitly named local file. Wavelet does not
download custom algorithm code.

### Validate before launch

Run preflight after creating or changing the custom file:

```bash
uv run python -m wavelet debug preflight @ path/to/rl.yaml --json
```

The `algorithm` check reports missing files, Python syntax/import errors,
duplicate registrations, constructor argument errors, missing hooks, and
non-callable algorithm names before GPU processes start.

## Runtime Flow

For each scoring batch, Wavelet:

1. Resolves the global or source-local algorithm.
2. Builds it and runs its lifecycle and declared scoring hooks.
3. Partitions group-scoped records by prompt group.
4. Validates hook return types, token alignment, and group cardinality.
5. Restores original cross-source record order.
6. Applies group-advantage filtering only to sources that use it.
7. Serializes advantages, component weights, and reference logprobs.
8. Packs heterogeneous records and normalizes each trainer component
   independently.

The `loss` block still configures off-policy correction and optimization math;
custom optimizer losses are separate from the algorithm layer. Wavelet does
not download custom algorithm code or manage frozen teacher processes.

## Legacy Configuration

Existing configs using `orchestrator.advantage_mode` remain accepted and are
normalized at the configuration boundary:

- `passthrough` becomes `algo.type: passthrough`.
- `reward` becomes `algo.type: reward`.
- `group_reward` becomes `algo.type: grpo`.

Legacy GRPO normalization, epsilon, and length-penalty fields are carried into
the named configuration. New configs should use `algo`. If both forms are
present, the explicit `algo` block takes precedence.

## Inspect the Resolved Configuration

```bash
uv run python -m wavelet debug orchestrator inspect \
  @ examples/reverse_text/rl.yaml --json
```

The JSON response includes the resolved `algo` block and all default values.

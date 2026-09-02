# RL Algorithms

Wavelet algorithms assign advantages to rollout records. The same runtime is
used for native and Verifiers-backed rollouts, so an algorithm behaves the same
regardless of where its completions came from.

The top-level `algo` block selects a built-in algorithm or a user-owned Python
file:

| Configuration | Scope | Behavior |
| --- | --- | --- |
| `type: passthrough` | rollout | Keep advantages already present in the data. |
| `type: reward` | rollout | Copy each reward into its missing advantage. |
| `type: grpo` | group | Center rewards within each prompt group. |
| `type: max_rl` | group | Center rewards and divide by the group mean. |
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

Training groups must contain genuinely sampled alternatives. When a group has
more than one rollout, Wavelet rejects greedy decoding, zero temperature, and a
fixed generation seed because repeated completions produce zero group-relative
advantages. Use `data.seed` for reproducible task ordering while leaving the
generation seed unset.

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
Factories should be lightweight and stateless because preflight and rollout
workers may construct them more than once. Hooks are currently synchronous;
async hooks and long-lived external model clients are not part of this
interface.

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

1. Builds the selected built-in or custom algorithm.
2. Runs the hooks declared by the algorithm scope.
3. Partitions group-scoped records by prompt group.
4. Validates hook return types and group cardinality.
5. Restores original record order.
6. Applies zero-advantage filtering when configured.
7. Serializes advantages into the existing trainer rollout format.

Algorithms only assign credit. The trainer loss remains configured separately
under `loss`; this interface does not add custom optimizer losses, asynchronous
model calls, or remote code loading.

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

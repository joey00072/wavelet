# AGENTS.md

Guidance for coding agents working in this repository.

## Scope and Intent

- This repository is a Python post-training library for SFT and RL experiments.
- Keep changes lightweight, explicit, and easy to review.
- Prefer practical improvements over speculative architecture.
- Treat `ref/` as read-only reference material.
- Do not edit files under `ref/`.
- Keep `ref/` out of ordinary search, lint, test, and formatting commands unless
  the user explicitly asks to inspect it.
- Optimize changes for debuggability: one obvious run path, explicit state,
  structured metrics, and small testable functions.
- When code changes alter commands, config defaults, examples, diagnostics,
  public behavior, or agent workflows, update the relevant docs in the same
  change. Do not leave documentation for a later cleanup pass.

## Repository Layout

- `pyproject.toml`: project metadata and Python requirement.
- `wavelet/`: package directory for app code, CLI, trainer, inference, and orchestration.
- `wavelet/entrypoints/`: CLI entrypoints for RL, SFT, trainer, inference, and
  vLLM server processes.
- `wavelet/configs/`: Pydantic config models and legacy normalization.
- `wavelet/orchestrator/`: rollout scheduling and sources, verifier environments,
  rewards, algorithms, metrics, state inspection, and launcher utilities.
- `wavelet/transport/`: canonical filesystem queue and filesystem/NCCL policy
  transport implementations. Legacy queue imports remain compatibility aliases.
- `wavelet/trainer/`: model and LoRA/QLoRA setup, distributed world/mesh state,
  SFT/RL trainers, losses, optimization, and checkpointing.
- `wavelet/inference/`: vLLM integration, policy adapter loading, and inference
  serialization.
- `wavelet/data/`: canonical `sft.py` and `rl.py` data pipelines; historical
  fine-grained module paths remain compatibility wrappers.
- `examples/`: runnable example configs and data-preparation helpers.
- `tests/`: pytest suite.
- `webui/`: lightweight run-state UI.
- `outputs/` and run directories: generated data, rollouts, policies,
  checkpoints, logs, and metrics. Do not commit generated run artifacts.
- `README.md`: project overview and quick-start notes.
- `ref/`: external/reference repos; read-only and excluded from default tooling.

## Tooling Baseline

- Python version: `>=3.11` (from `pyproject.toml`).
- Use `uv` for environment and command execution.
- Run commands from repository root: `/home/joey/workspace/wavelet`.
- Prefer deterministic CLI invocations that can run in CI.

## Git and Credentials

- Source `.env` before commit or push commands so local identity and tokens are
  available without hard-coding them in git config or command history.
- Use `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME`, and
  `GIT_COMMITTER_EMAIL` from `.env` for commits when present.
- Use GitHub token variables from `.env` for push authentication when needed.
- Never print, commit, or otherwise expose `.env` contents.

## Build, Run, Lint, and Test Commands

### Environment Setup

- Sync dependencies: `uv sync`
- Sync with optional groups (if later added): `uv sync --all-extras`
- Run any command inside env: `uv run <command>`

### Run / Build

- Preferred app execution path: `uv run python -m wavelet`
- CLI commands use `uv run python -m wavelet <subcommand>`.
- Available commands:
  - `uv run python -m wavelet sft @ examples/<example>/sft.yaml`
  - `uv run python -m wavelet rl @ examples/<example>/rl.yaml`
  - `uv run python -m wavelet rl-inference @ <config>.yaml`
  - `uv run python -m wavelet rl-trainer @ <config>.yaml`
  - `uv run python -m wavelet rl-orchestrator @ <config>.yaml`
  - `uv run python -m wavelet inference-server @ <config>.yaml`
  - `uv run python -m wavelet debug preflight @ <config>.yaml --json`
  - `uv run python -m wavelet debug inference inspect @ <config>.yaml --json`
  - `uv run python -m wavelet debug orchestrator inspect @ <config>.yaml --json`
- Build source/wheel packages: `uv build`

### Known Runnable Examples

- `examples/reverse_text/`: small RL/SFT path; use this for fast pipeline checks.
- `examples/alphabet_sort/`: working 4B LoRA RL path.
- `examples/qwen4b_math/`: single-node 4B math adaptation.
- `examples/unsloth_math/`: SFT and RL math quick-start.
- `examples/qwen30b_math/`: larger-model configs, including colocate/sleep
  smoke configs; expect GPU-memory-sensitive runs.

When adding or changing an example, update `examples/README.md` and prefer a
small smoke config that can prove the pipeline before long runs.

### Lint and Format

- Lint (Ruff): `uv run ruff check .`
- Auto-fix lint issues: `uv run ruff check . --fix`
- Format (Ruff formatter): `uv run ruff format .`
- Type check (if mypy/pyright configured): `uv run mypy .`
- Project size snapshot: `uv run python scripts/project_size.py --write docs/project_size.jsonl`
  This counts tracked source/docs/config files while excluding `ref/`, run outputs,
  caches, virtualenvs, and WandB artifacts.

### Test

- Run full test suite: `uv run pytest`
- Run full test suite with dev extras: `UV_LINK_MODE=copy uv run --extra dev pytest`
- Run with verbose output: `uv run pytest -vv`
- Run a single test file: `uv run pytest tests/test_example.py`
- Run a single test function: `uv run pytest tests/test_example.py::test_case_name`
- Run tests by keyword expression: `uv run pytest -k "keyword"`
- Stop on first failure: `uv run pytest -x`
- Re-run last failed tests: `uv run pytest --lf`
- `pytest` is configured to use `tests/` and skip `ref/`; do not remove that
  guard without a specific reason.

## Debugging and Run Hygiene

- Prefer short deterministic smoke runs before long GPU runs.
- Run `uv run python -m wavelet debug preflight @ <config>.yaml --json` before
  expensive RL launches to catch missing local data, port conflicts, device
  placement mismatches, stale output directories, and resolved role commands.
- Use the smallest working example that exercises the subsystem being changed.
- Treat a completed process as only one success criterion. For training runs,
  also verify baseline eval, final eval, failed rollout count, queue health,
  policy freshness, and process teardown.
- Do not claim reward improvement from noisy train batches alone. Prefer eval
  metrics at fixed policy steps, plus a short comparison of early-window and
  late-window training reward.
- Every new long-running path should make it easy to answer:
  - Which config was resolved?
  - Which GPUs are assigned to each process?
  - Which policy version is being generated, trained, exported, and loaded?
  - How many rollouts are queued, scored, consumed, stale, or dropped?
  - What was the last error per worker?
- Run state should live under the configured output/run directory. Keep names
  predictable: `rollouts/`, `policies/`, `checkpoints/`, `logs/`, and metrics
  JSONL files.
- Keep policy metadata adjacent to policy artifacts. Do not infer policy version
  from directory names alone when writing new code.
- For async RL bugs, inspect queue depth, policy freshness, trainer step time,
  inference generation rate, and reward distribution before changing
  hyperparameters.
- Async throughput is only useful when data freshness is bounded. Generation
  must respect the configured off-policy window even when inference has spare
  capacity; otherwise high utilization can silently produce stale rollouts.
- Queue observability should expose both counts and causality: published,
  claimed, consumed, stale, abandoned, producer id, consumer id, optimizer step,
  policy step, and parse errors.
- Check rank ownership for side effects in distributed runs. Filesystem events,
  queue lifecycle markers, metrics that represent global progress, and policy
  exports should be written by the intended rank only.
- If a server accepts both rendered chat messages and pre-tokenized prompts,
  validate context length against the representation actually used for
  generation. Leave a small token safety margin instead of fitting exactly to
  the model limit.
- When source code changes affect a running service, restart the service before
  treating the run as validation. A live process proves the code it loaded, not
  the current working tree.
- When retrying a failed run, use a clean output directory unless the test is
  explicitly about resume behavior. Old policies, rollouts, checkpoints, and
  caches can make failures look fixed or make fixes look broken.
- If utilization looks wrong, first determine whether the bottleneck is
  intentional backpressure, policy loading, generation, training, data loading,
  or teardown. Idle GPUs are not automatically a bug when freshness limits are
  holding the system back.
- Prefer fixing instrumentation gaps before guessing at hyperparameters. A
  system that cannot explain queue state, policy lag, and worker errors is not
  ready for tuning.
- Keep WandB logging available when the config enables it, but never require
  WandB for tests or local smoke validation.
- Do not leave background training, vLLM, web UI, or monitor processes running
  after a debugging task unless the user explicitly asked to keep them alive.

## GPU and Distributed Runs

- Honor `CUDA_VISIBLE_DEVICES` when the user specifies GPU placement.
- Do not assume all eight GPUs are free. Check active processes before starting
  multi-GPU jobs.
- Use `torchrun` only for trainer entrypoints that need distributed workers.
  Do not wrap the combined `wavelet rl` launcher in `torchrun`.
- Keep trainer, inference, and vLLM process responsibilities separate unless a
  colocated config explicitly says otherwise.
- If a distributed run fails, capture the exact command, resolved config path,
  rank-local traceback, and output directory before retrying.

## Coding Standards

### Design Bias

- Make state explicit instead of hidden in process-local globals.
- Prefer one shared helper over repeated inline math for scheduling, chunking,
  policy freshness, and metric naming.
- Prefer typed dataclasses or Pydantic models at subsystem boundaries.
- Keep launch glue thin; put behavior in importable modules with focused tests.
- Avoid compatibility aliases unless they are normalized at the config boundary.
- Do not add broad abstractions for one call site.

### Imports

- Group imports in this order: standard library, third-party, local.
- Separate groups with one blank line.
- Prefer explicit imports over wildcard imports.
- Avoid importing unused symbols; keep import lists minimal.

### Formatting

- Follow PEP 8 and keep formatting tool-driven.
- Max line length: 88 (Ruff/Black-compatible default).
- Use double quotes consistently unless project style changes.
- Keep functions short and single-purpose where practical.

### Types

- Add type hints for all public functions and methods.
- Prefer concrete built-in generics (`list[str]`, `dict[str, int]`).
- Use `| None` for optionals (Python 3.11+ style).
- Avoid `Any` unless unavoidable; narrow types as soon as possible.

### Naming Conventions

- Modules/files: `snake_case.py`
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `tests/test_<unit>.py`
- Test functions: `test_<behavior>`

### Error Handling

- Fail fast on programmer/configuration errors.
- Do not add broad `try/except` blocks without clear recovery behavior.
- Catch specific exception classes only when you can handle them meaningfully.
- Avoid silently swallowing exceptions.

### Logging and Output

- Keep user-facing CLI output concise and informative.
- Do not print secrets, tokens, or credentials.

### Comments and Docstrings

- Add comments only for non-obvious intent or tricky constraints.
- Do not add process comments describing prior code states.
- Keep docstrings concise and focused on behavior and contracts.
- Update docstrings when behavior changes.

### Testing Expectations

- Use `pytest` with plain test functions.
- Prefer fixtures for shared setup; keep fixtures scoped and minimal.
- Include negative-path tests for error-prone logic.
- For bug fixes, add or update tests that fail before and pass after.

## Agent Behavior Rules

- Read existing code before editing.
- Make the smallest change that fully solves the task.
- Preserve unrelated local modifications.
- Do not perform destructive git actions.
- If adding dependencies, update `pyproject.toml` and document rationale.
- Do not edit generated artifacts, run outputs, checkpoints, policy exports, or
  vendored/reference files as part of ordinary source changes.
- Before committing, run a targeted test for the changed behavior and at least:
  `uvx ruff check wavelet tests` and `git diff --check`.
- If touching RL scheduling, rollout queues, policy sync, trainer stepping, or
  reward calculation, add or update tests in `tests/` that exercise the invariant.
- If touching docs or agent guidance, ensure there are no stale project names,
  stale paths, or instructions that point outside `/home/joey/workspace/wavelet`.
- Keep commit messages focused on Wavelet behavior. Do not mention external
  reference repositories unless the user explicitly asks.

## Cursor/Copilot Rules

- No `.cursorrules` file detected at repository root.
- No `.cursor/rules/` directory detected at repository root.
- No `.github/copilot-instructions.md` detected at repository root.
- If these files are added later, treat them as higher-priority constraints
  and merge their requirements into this document.

## Zen of Python

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

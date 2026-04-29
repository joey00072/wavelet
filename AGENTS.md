# AGENTS.md

Guidance for coding agents working in this repository.

## Scope and Intent

- This repository is currently a small Python project scaffold.
- Keep changes lightweight, explicit, and easy to review.
- Prefer practical improvements over speculative architecture.
- Treat `ref/` as read-only reference material.
- Do not edit files under `ref/`.

## Repository Layout

- `main.py`: temporary scaffold only; do not treat as long-term entry point.
- `pyproject.toml`: project metadata and Python requirement.
- `fuchsia/`: package directory; expected home for app code and CLI/module entry points.
- `README.md`: currently empty.
- `ref/`: external/reference repos for style and workflow inspiration.

## Tooling Baseline

- Python version: `>=3.11` (from `pyproject.toml`).
- Use `uv` for environment and command execution.
- Run commands from repository root: `/home/joey/workspace/fuchsia`.
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

- Preferred app execution path: `uv run python -m fuchsia`
- If a CLI script is added later: `uv run fuchsia <subcommand>`
- `main.py` may be used only for temporary local scaffolding.
- Build source/wheel packages: `uv build`

### Lint and Format

- Lint (Ruff): `uv run ruff check .`
- Auto-fix lint issues: `uv run ruff check . --fix`
- Format (Ruff formatter): `uv run ruff format .`
- Type check (if mypy/pyright configured): `uv run mypy .`

### Test

- Run full test suite: `uv run pytest`
- Run with verbose output: `uv run pytest -vv`
- Run a single test file: `uv run pytest tests/test_example.py`
- Run a single test function: `uv run pytest tests/test_example.py::test_case_name`
- Run tests by keyword expression: `uv run pytest -k "keyword"`
- Stop on first failure: `uv run pytest -x`
- Re-run last failed tests: `uv run pytest --lf`

## Coding Standards

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

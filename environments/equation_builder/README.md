# Equation Builder environment

This local Verifiers environment generates deterministic, guaranteed-solvable
arithmetic tasks. Every task contains 3, 4, or 5 unique two-digit integers, a
target between 0 and 99 by default, and no answer. The default is 4 numbers.
Generation constructs a valid equation to guarantee that the target is solvable,
then discards that equation before creating the dataset row.

The model may reorder the numbers and add parentheses. A submission receives a
reward of `1.0` only when the content of `<answer>...</answer>`:

- contains exactly one equation and integer target
- uses every supplied number exactly once, including duplicate-aware counting
- uses only integer literals, `+`, `-`, and parentheses
- evaluates to the requested target

Evaluation uses a restricted Python AST interpreter. Model output is never passed
to `eval()` or `exec()`.

The environment accepts `num_examples`, `eval_examples`, `seed`, `num_numbers`,
`target_min`, and `target_max` arguments. `num_numbers` must be `3`, `4`, or `5`.

Install it into Wavelet's project environment after syncing the Verifiers extra:

```bash
uv sync --extra verifiers
uv pip install --python .venv/bin/python \
  --editable environments/equation_builder
```

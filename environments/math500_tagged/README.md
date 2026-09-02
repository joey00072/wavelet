# Tagged MATH-500 environment

This local Verifiers environment uses the 500-row `test` split from
`HuggingFaceH4/MATH-500` for both RL sampling and periodic evaluation.

Responses must consist of exactly:

```text
<think>step-by-step reasoning</think>
<answer>final answer</answer>
```

The answer is scored with `math-verify`. Missing, duplicated, reordered, or
unclosed tags receive zero reward. Because training and evaluation use the same
500 problems, the periodic metric measures training-set improvement rather than
held-out generalization.

Install after syncing the Verifiers dependency:

```bash
uv pip install --python .venv/bin/python \
  --editable environments/math500_tagged
```

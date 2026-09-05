# Polaris math with tagged AIME 2024 evaluation

This Verifiers environment uses pinned revision
`296f8e34132e63f4a1d70e0dcc8bddebb43f03e4` of
`POLARIS-Project/Polaris-Dataset-53K` for training and the pinned 30-problem
`HuggingFaceH4/aime_2024` source used by the upstream AIME 2024 environment for
held-out evaluation. Pinning both sides keeps deterministic row IDs, epoch
order, and baseline comparisons stable across resumed runs.

By default it keeps Polaris difficulty buckets `1/8` through `6/8`, removes
duplicate normalized problems, and removes exact normalized overlaps with AIME
2024. It also removes proof requests because final-answer equivalence cannot
grade proof validity and some Polaris proof rows contain corrupted target
fragments. Narrow structural checks also remove incomplete targets such as a
leading exponent marker, an empty `\\frac` operand, or an operator-only answer.
Pass `exclude_proof_problems=false` only for a rubric that can grade proofs, or
`exclude_malformed_answers=false` to inspect the original malformed labels.
Both training and evaluation require exactly one
`<think>...</think><answer>...</answer>` response and use `MathRubric` for
binary correctness.

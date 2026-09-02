# Polaris math with tagged AIME 2024 evaluation

This Verifiers environment uses `POLARIS-Project/Polaris-Dataset-53K` for
training and the pinned 30-problem `HuggingFaceH4/aime_2024` source used by
Prime's AIME 2024 environment for held-out evaluation.

By default it keeps Polaris difficulty buckets `1/8` through `6/8`, removes
duplicate normalized problems, and removes exact normalized overlaps with AIME
2024. It also removes proof requests because final-answer equivalence cannot
grade proof validity and some Polaris proof rows contain corrupted target
fragments. Pass `exclude_proof_problems=false` only for a rubric that can grade
proofs. Both training and evaluation require exactly one
`<think>...</think><answer>...</answer>` response and use `MathRubric` for
binary correctness.

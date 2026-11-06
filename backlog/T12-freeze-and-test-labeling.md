# T12 · Freeze and test labeling

Tier: 1 · Owner: _ · Depends: T10, T11

## What

1. **Freeze:** router checkpoint, θ, budgets, ε, the effect-size target, the equivalence margin, `primary_comparator`, prompts, and the generator version. Commit and tag `v1-frozen`.
2. Build test indexes: F-100, PIC-capped and the comparator (seeds as decided).
3. Label the test set: all 9 configs × runs A/B for each of the 3 indexes. Use the full split, or the pre-registered document subset decided in T08.
4. Run the router on the test questions (F-100 index) and record its chosen configs.

## Done when

- The `v1-frozen` tag exists, and every test config validates (`--validate-only` with `dataset.split=test`).
- The label store has complete test rows; the collector reports 0 missing.
- Spend logged; cumulative ≤ $50.

## Description

After the freeze, nobody changes prompts, settings or models for Tier 1. That is what makes the test numbers honest.

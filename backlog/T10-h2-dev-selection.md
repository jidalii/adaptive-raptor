# T10 · H2 on dev: index variants and comparator selection

Tier: 1 · Owner: _ · Depends: T08, T09

## What

- Build all 5 variants (F-100, F-μ, PIC-capped, permuted, paragraph) on the dev documents chosen for H2, with the seeds decided in T08.
- Label all 9 configs × runs A/B on those questions for each variant.
- Compute the cross-fitted oracle F1 at the pre-set `matched_budget`, and the best deployable fixed config per variant.
- **Select the strongest non-PIC control** by the H2 primary metric on dev; write it into `chunking.primary_comparator` and freeze it.
- E5 (part): build cost per variant, with PIC's summary cost as a fraction of RAPTOR's build.
- Tree-shape diagnostics per variant: levels, nodes per level, fanout, summary tokens, within-cluster similarity.

## Done when

- `configs/experiments/h2.yaml` has `primary_comparator` set, and validates with `dataset.split=test`.
- A `NOTES.md` section with the dev table (variant × oracle F1 × best-fixed F1 × build cost) and the seed spread.
- Spend logged.

## Description

Picking the comparator on dev, then never changing it, is what keeps the test comparison from being tuned to the result.

## Notes

- To save budget, build the 5 variants and 3 seeds **only** on dev/pilot documents. Test needs only F-100, PIC-capped and the chosen comparator (proposal §6.4).

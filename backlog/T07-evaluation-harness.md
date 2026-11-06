# T07 · Evaluation harness: frontiers, oracle, matching, bootstrap

Tier: 1 · Owner: _ · Depends: T06

## What

- `evaluation/research.py` (frontiers):
  - fixed-config frontier: (config, mean tokens, mean F1) and its upper envelope;
  - **cross-fitted oracle**: pick each query's config on run A, score it on run B (proposal §6.2).
- `evaluation/research.py` (`match_fixed_baseline`): the conservative "≥ router tokens" rule and the interpolation variant (proposal §5.4).
- `evaluation/research.py` (`paired_document_bootstrap`): paired, document-level bootstrap CI (n from config), and the equivalence check against `equivalence_margin`.
- `evaluation/traces.py` (`write_rows`, already implemented): tables in csv / json / markdown, per `output_formats`.

## Done when

- Unit tests on synthetic label tables, with known answers for each: frontier, oracle with and without cross-fitting, matching rule, and bootstrap coverage (a CI built on a zero-effect simulation contains 0 about 95% of the time).
- No API calls anywhere in this ticket.

## Description

These are the decision rules we promised to fix before seeing test data. Writing and testing them on synthetic data now is what makes the later test evaluation mechanical.

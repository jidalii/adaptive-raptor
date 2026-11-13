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

## Notes

- Nov 9 (`evaluation/research.py`): done, with no API calls.
  - **Fixed configs:** `per_query_table` builds the common query set. `fixed_config_points` and `upper_envelope` give `fixed_config_frontier`.
  - **Oracle:** `oracle_points` / `oracle_frontier` run a Lagrangian sweep, choosing per query `argmax F1_A − λ·tokens` and scoring on run B when cross-fitting. `oracle_at_budget` gives the H2 and E0 numbers.
  - **Matching:** `match_baseline` supports `conservative_at_least` and `interpolate`. `baseline_scores` gives per-query baseline F1 for paired deltas.
  - **Statistics:** `paired_document_bootstrap` is vectorized and query-weighted, with percentile CIs. `verdict()` returns supported / worse / equivalent / inconclusive.
  - Tests (`tests/test_labels_and_research.py`) have known answers for every rule. The zero-effect bootstrap simulation (200 sims, clustered data) covers 0 at about 95%.
  - **Decision to confirm before T12:** the reading of the conservative matching rule (see `NOTES.md`).


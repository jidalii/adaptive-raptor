# T11 · Router: features, acceptability, selection, calibration

Tier: 1 · Owner: _ · Depends: T06, T08 (label decisions). Code can start during T08.

## What

All in `optimizer/__init__.py` (proposal §5.2–5.3); the label function is in `profiling.py`. Remove `Stub` from each class when it's done:

- `FeatureExtractor`: PCA(`pca_dim`) fitted on train embeddings only, plus lexical features (numbers, dates, entities, wh-type, length).
- `AcceptabilityModel`: one regularized logistic regression per config, with `GroupKFold` by document for hyperparameters. Use the ordinal min-sufficient-budget variant instead if T08 found monotonicity.
- `CostModel`: measured mean `tokens_used` per config from dev labels.
- `PlanSelector`: the cheapest config with P ≥ θ, else the fallback.
- `calibrate_theta` / `theta_sweep`: θ calibrated on dev to `target_mean_tokens`, plus a θ sweep that gives the router's own frontier.
- `train_router`, `save_checkpoint`, `load_checkpoint`, `Router`: label store → frozen checkpoint (models, PCA, θ, config hash).
- Collect train labels on the F-100 index (about 1k train questions × 9 configs, run A).

## Done when

- `checkpoint` loads and routes a dev question in < 1 ms (excluding embedding), measured and logged.
- Dev θ sweep frontier plotted against the fixed-config frontier (`results/dev/`).
- Train labels' spend logged.

## Description

The router never sees test documents. Everything it knows, including θ, comes from train and dev, and it is frozen in T12.

## Notes

- **Nov 9: rescoped to the LLM-profiler router (Plan A).** The design is in `docs/OPTIMIZER_PLAN.md`. The embedding classifier described above is deferred and built only if the LLM router shows no dev improvement.
- **Done (offline, tested; `tests/test_optimizer.py`):**
  - the profiler, with mock / LLM / cached variants;
  - the profile-table model (A1) and the rule-based mapping (A2);
  - the cost model, selector, θ calibration on total cost, θ sweep and checkpoints;
  - the `Router` and `RouterPolicy`;
  - the `query_profiling` and `router_train` experiments.
- **Live:** 87 pilot questions were profiled ($0.006 per run). Findings (unusable t=0 logprobs, 332-token overhead, 16% rerun instability) are in `NOTES.md`.
- **Left:**
  1. Collect train labels: F-100 indexes for train papers, plus the full catalog on about 1k train questions, run A. Profile the train and dev questions.
  2. Run `router_train`. Its `dev_report.json` is the go/no-go.
  3. WP10, the H1 runner (T13).
- Train labels need the pilot (T08) decisions first: budgets, ε, and the summary cap.
- **Nov 9 (later):**
  - Prompt `profile_v3` is frozen (hash `e22d6935b289c85d`), and the confidence fallback is off (decision a).
  - WP10 (the H1 replay runner) is done and smoke-tested.
  - Train labeling waits for the T08 parameters.


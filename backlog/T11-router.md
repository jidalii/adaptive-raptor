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

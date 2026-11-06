# T13 · Test experiments E1–E5

Tier: 1 · Owner: _ · Depends: T12

## What

Replace the `PlaceholderExperiment` runners in `experiments/__init__.py` with real ones (use `QAPipeline`, plus `RouterPolicy` for E1), and implement the statistics in `evaluation/research.py`. Runners read the T12 label store and make no new API calls (E2's restricted routers reuse those labels):

- **E1 (H1):** router vs. the matched best fixed config; paired document bootstrap of Δ F1.
- **E2:** 2×2 ablation of strategy × budget, each cell with its own router trained on train labels and calibrated on dev.
- **E3 (H2):** PIC-capped vs. the comparator; cross-fitted oracle F1 at the matched budget plus the best fixed config, with bootstrap and equivalence check.
- **E4 (exploratory):** oracle frontiers and acceptability shares per index.
- **E5:** build cost, routing latency, retrieval latency.

## Done when

- `results/test/` holds the main table (H1, H2 with CIs), the ablation table, the frontier figure, and the cost table.
- Each runner reports `succeeded`, and its resolved config is saved with the outputs.
- Results are interpreted using the H2 rules written in advance (proposal §6.2), including inconclusive and negative outcomes.

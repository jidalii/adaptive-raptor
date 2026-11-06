# T06 · QASPER F1, label store and collection

Tier: 1 · Owner: _ · Depends: T02, T03, T05

## What

- `evaluation/metrics.py::answer_f1`: official QASPER answer F1 (max over annotators; type of the best reference).
- `evaluation/metrics.py::evidence_recall`: diagnostic only; a summary is credited only for evidence it contains (proposal §6.3).
- `profiling.py`:
  - `LabelRecord`, `from_traces`, and `LabelStore` (append-only JSONL);
  - `acceptability_labels(records, epsilon)`: per-query best F1 − ε, `None` for all-fail queries.
- Label collection = the `raptor_baseline` runner over the full catalog (`configs/experiments/pilot_catalog.yaml`). Each run writes `labels.jsonl`; run B uses `--set experiment.run_label=B --set generation.prompt_template=qasper_paraphrase`.

## Done when

- `answer_f1` matches the official evaluator on dev.
- A killed collection run can be restarted without repaying for API calls and without duplicate rows in the merged label table.
- One pilot paper × 9 configs × runs A/B gives 18 × (#questions) label rows.

## Description

Every later number is computed from these records: what was asked, which config, how many tokens, which model version, and the score.

## Notes

- Nov 6: `answer_f1` matches the official `qasper_evaluator.py` on the real dev set (integration test, three prediction sets). Traces, `LabelStore` and `from_traces` done.
- Restarts already cost nothing (the generation and summary caches), but each run writes a new run directory, so merging label tables across runs still needs deduplication by (question, variant, seed, config, run).
- **Left:** `acceptability_labels` (stub in `profiling.py`), `evidence_recall` (stub), and the dedup/merge step.

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
- Nov 9 (`profiling.py`):
  - **`acceptability_labels(records, epsilon, configs=None, report=...)`**
    - Records must come from one (variant, seed, run) cell; `select_records` filters to one.
    - All-fail queries map to `None`. Queries missing a config are excluded and listed in `LabelReport.incomplete`.
    - The report gives per-config acceptable rates and per-answer-type counts.
  - **`merge_label_tables(paths)`**
    - Deduplicates by (question, variant, seed, config, run); the first row wins.
    - A differing F1 is reported as a conflict, never averaged.
  - Tests are in `tests/test_labels_and_research.py`.
  - **Live check passed:** 1 pilot paper × 9 configs × runs A/B = 54 rows (18 × 3 questions). A rerun of A made 0 new calls, and the merge gave 81 → 54 rows with 0 conflicts.
  - **Done.**
- `evidence_recall` stays a stub. Its definition (how a summary is "credited only for evidence it contains") is a research choice for the E6 diagnostics, so it moves to **T14**.

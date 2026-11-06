# T14 · Error analysis and write-up

Tier: 1 · Owner: _ · Depends: T13

## What

- Router errors:
  - topic vs. information need: compare a lexical-only router with an embedding-only router;
  - cases where the fallback fired;
  - queries where the oracle succeeded and the router failed.
- Chunking errors:
  - gold evidence split across chunk boundaries, per variant;
  - summary failures (leaf-only correct, summary-only wrong).
- E6 diagnostics: evidence recall (`evaluation/metrics.py::evidence_recall`), split-evidence rate, tree shape.
- Final report and presentation: setup from `NOTES.md` (pilot section), results from `results/test/`, the threats table from proposal §7.

## Done when

The report states H1 and H2 outcomes with CIs, every confirmatory vs. exploratory label is correct, and limitations include generator dependence and any budget-driven subsetting.

# Backlog

Build tickets, worked **in order**. Each one depends on the tickets in its `Depends:` line. Set `Owner:` in the ticket when someone picks it up, and update the Status column below when its "Done when" check passes.

This is the **only place ticket status is tracked**. *Infra done* = code and offline tests are in place, but the ticket's live "Done when" check (real models / API / pilot data) has not run yet; each ticket's Notes say what is left. The design behind the tickets is in the [proposal](../../Adaptive_RAPTOR_Proposal.md) (§3 architecture, §4 chunking, §5 router, §6 experiments, §7 threats, §8 schedule).

**Budget:** $50 total for generation and index construction ([proposal §6.4](../../Adaptive_RAPTOR_Proposal.md#64-generator-configuration-and-compute-budget)). Every ticket that calls an API logs its spend in `NOTES.md` (Spend log).

## Resolved blocker (Nov 6 → fixed Nov 9)

`deepseek-flash` thinks by default, and its hidden reasoning used up `summarization_length`, so every summary came back empty. Fixed in T03: every request sends `thinking: disabled`, and empty completions are errors. Indexes were rebuilt under a new key in T04. Evidence and spend are in [`NOTES.md`](../NOTES.md).

## Tier 1: H1 + H2 on QASPER (must-have)

Minimum viable deliverable = E0 + E1 + E2 + E3 + E5.

| # | Ticket | Target week | Status |
|---|---|---|---|
| T01 | [Repo, environment and config system](T01-repo-and-environment.md) | Nov 6 | Done |
| T02 | [QASPER download, splits and pilot set](T02-qasper-data.md) | Nov 6 – 9 | Done |
| T03 | [Generator backend (DeepSeek)](T03-generator-backend.md) | Nov 6 – 9 | Done |
| T04 | [RAPTOR index from custom leaves](T04-raptor-index.md) | Nov 7 – 10 | Done |
| T05 | [Retrieval executor (strategy × budget)](T05-retrieval-executor.md) | Nov 9 – 11 | Done |
| T06 | [QASPER F1, label store and collector](T06-metrics-and-labels.md) | Nov 9 – 11 | Done (evidence recall → T14) |
| T07 | [Evaluation harness: frontiers, oracle, matching, bootstrap](T07-evaluation-harness.md) | Nov 10 – 12 | Done |
| T08 | [Pilot (E0) and go/no-go](T08-pilot-and-go-no-go.md) | Nov 11 – 13 |  |
| T09 | [PIC-capped and chunking controls](T09-chunkers.md) | Nov 12 – 16 |  |
| T10 | [H2 on dev: index variants and comparator selection](T10-h2-dev-selection.md) | Nov 16 – 19 |  |
| T11 | [Router: features, acceptability, selection, calibration](T11-router.md) | Nov 14 – 19 | Code done (Plan A); needs train labels |
| T12 | [Freeze and test labeling](T12-freeze-and-test-labeling.md) | Nov 20 – 23 |  |
| T13 | [Test experiments E1–E5](T13-test-experiments.md) | Nov 23 – 27 |  |
| T14 | [Error analysis and write-up](T14-error-analysis-and-report.md) | Nov 27 – Dec 3 |  |

**Tier 1 finish line (~Nov 27):** every confirmatory number (H1, H2) on the test set. Nov 26 is Thanksgiving; plan around it.

## Tier 2: stretch (only after T13)

| # | Ticket | Target week | Status |
|---|---|---|---|
| T15 | [Router on the PIC index (Option C)](T15-router-on-pic.md) | Nov 28 – Dec 1 |  |
| T16 | [QuALITY as a second dataset](T16-quality-dataset.md) | Nov 28 – Dec 2 |  |
| T17 | [Traversal plans in the catalog](T17-traversal-plans.md) | Dec 1 – 3 |  |

**Cut order if anything slips:** T17 → T15 → T16 → reduce seeds in T10 (keep a reduced seed check) → the permuted/paragraph controls (cut last). Keep H1 before H2, and shrink H2's scale before removing its controls.

## Where teammates can overlap

These tickets don't touch each other's files and can go to different people in the same week:

- **Nov 6 – 11:** T02 ∥ T03 ∥ T04; then T05 ∥ T06.
- **Nov 12 – 19:** T09 (chunkers) ∥ T08 (pilot); then T10 ∥ T11.
- **Nov 27 – Dec 3:** T14 ∥ Tier 2.

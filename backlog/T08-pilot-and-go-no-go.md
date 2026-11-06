# T08 · Pilot (E0) and go/no-go

Tier: 1 · Owner: _ · Depends: T04, T05, T06, T07

## What

On the **pilot set only** (T02), with F-100:

1. **Cost and throughput:** about 50–100 questions × 9 configs × runs A/B. Measure real tokens per call, latency, throughput, and $ spent. Re-estimate the full budget table (proposal §6.4) and decide whether the test set is the full split or a pre-registered document subset.
2. **Oracle headroom:** cross-fitted oracle F1 vs. the best fixed config at a matched budget.
3. **Label quality:**
   - the acceptability label distribution by answer type, and its sensitivity to ε;
   - whether quality is monotone in budget within each strategy (this decides between the logistic and ordinal router in T11);
   - the run A vs. run B disagreement rate.
4. **Seed variance:** 3 seeds on about 30 documents; the F1 spread across seeds.
5. **Decide and write down:** budget levels, ε, the H1 effect-size target, the equivalence margin, and the go/no-go for H1 (the threshold, about 2× the H1 target, is written **before** step 2 runs).
6. Start `NOTES.md` with a **Pilot** section (what was tried, what was chosen, why) and a **Spend log**.

## Done when

- `NOTES.md` has the pilot section, the decisions above, and a go/no-go verdict.
- The example configs are updated with the chosen budgets, ε and margins.
- Cumulative spend is recorded and the projected total is ≤ $50 (or a cut is chosen from the README's cut order).

## Description

Every "decide in the pilot" from the proposal gets decided here, on documents that are never reported. If the oracle gap is too small, H1 becomes a headroom/negative result and effort moves to H2/E4. That is a planned outcome, not a failure.

## Notes

- Run it with `configs/experiments/pilot_catalog.yaml` once T02's pilot ids exist and the T03/T04 blocker is fixed.
- Spend so far (Nov 6), to copy into the Spend log: first real baseline, 2 papers (18 summary + 8 answer calls, about 22.5k input / 2.8k output tokens, about $0.005; results invalid because summaries were empty), plus one integration-test call (under $0.001).

- A `pilot` experiment type may be cleaner than reusing `raptor_baseline`. Add it to the schema and registry if needed.

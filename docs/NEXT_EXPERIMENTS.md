# Next experiments: summary-cap ablation and sufficiency-triggered cascade

Prepared 2026-11-09. **Nothing here has been run; every paid step needs approval.** The test split stays untouched. The frozen profiler and the vendored RAPTOR are unchanged.

Prices are deepseek-flash, from the official pricing page (checked 2026-11-09):

| | Off-peak (per 1M tokens) | Peak |
|---|---|---|
| Input, cache miss | $0.15 | 2× |
| Input, cache hit | $0.003 | 2× |
| Output | $0.60 | 2× |

Peak hours are 01–04 and 06–10 UTC, Monday to Friday. Run every step off-peak. Cost estimates come from `scripts/estimate_cost.py` and ignore provider cache hits, so input cost is an upper bound.

## Background

`outputs/diagnostics/t12_redesign/report.json` and the NOTES section "Optimizer redesign" record what led here:

- No router beats fixed leaf_only at matched cost.
- The abstention cascade comes closest: +0.3 [−1.0, +1.6] F1.
- 94% of cap-200 summaries hit the cap, and 90% end mid-sentence.
- With the cap-200 indexes, collapsed@1000 already covers less leaf evidence than leaf_only@1000 (0.83 vs 0.95).

Retrieval-score routing is stopped.

## Experiment 1 (priority): summary cap 200 vs 512

**Hypothesis.** Truncated summaries add little evidence and displace leaf chunks. Complete summaries would make hierarchical retrieval (collapsed, original RAPTOR) competitive with leaf_only at the same budget.

**Design.** A controlled A/B on the same questions, with no tuning:

- **Control:** cap 200, the existing runs.
- **Treatment:** cap 512, the only change. Config: `configs/experiments/summary_cap512_dev.yaml`.

Everything else is identical: chunks, SBERT model, clustering, generator, prompt, budgets and seed. The vendored RAPTOR code is unchanged, because the cap is a config value passed to upstream `summarize(max_tokens=...)`. Upstream's summary prompt has no length instruction ("including as many key details as possible"), so **summaries may still hit 512**. The pilot checks this before any larger spend.

Configs run: leaf_only, collapsed and summary_only at 1000 and 2000, plus original RAPTOR.
- leaf_only@1000 and leaf_only@2000 are the strong baselines. They are also the **control check**: leaf nodes don't depend on the cap, so their answers should be identical, served from the response cache.
- The 400 budget is dropped, because it isn't competitive.

**Separating summary quality from context length:**
1. **Same context length.** Budgets are caps on retrieved context, so both arms answer from about the same amount of context. Actual context tokens are reported for each arm.
2. **Composition.** For each arm, report nodes per context, summary share, leaf-only evidence coverage (the displacement measure) and full-context evidence coverage.
3. **Summary quality on its own.** Report the share of summaries at the cap, the share ending mid-sentence, and the best gold-evidence word recall of any summary node, paired by question. This measures summaries independently of retrieval.
4. **Difference-in-differences.** Compare (strategy − leaf_only)@B across arms. It removes question difficulty and anything common to both arms.

**Run order:**

| Step | Command | Est. cost (off-peak) |
|---|---|---|
| 1a. Pilot: 30 papers, 87 questions | `uv run python -m adaptive_raptor.run --config configs/experiments/summary_cap512_dev.yaml --set dataset.doc_ids_file=data/subsets/qasper/v0.3/dev_pilot.txt --set experiment.name=summary_cap512_pilot` | $0.18–0.28 |
| 1a analysis | `uv run python scripts/analyze_summary_cap.py --control outputs/pilot_catalog/20261109T195138Z --treat outputs/summary_cap512_pilot/<run> --out outputs/diagnostics/summary_cap_pilot` | $0 |
| **Gate** | Continue only if treatment summaries are complete: share ending mid-sentence < 20% and share at the cap < 20%. Otherwise report and stop; a length instruction would need a prompt change to the vendored summarizer, which needs approval. | |
| 1b. Full: dev_rest, 251 papers, 918 questions | `uv run python -m adaptive_raptor.run --config configs/experiments/summary_cap512_dev.yaml` | $1.71–2.60 |
| 1b analysis | `uv run python scripts/analyze_summary_cap.py --control outputs/labels_dev/20261109T195548Z --treat outputs/summary_cap512_dev/<run>` | $0 |

**Outputs:** `summary_cap_report.json` with sections `summaries_control/treat`, `best_summary_evidence_recall`, `per_config` (F1, tokens and USD, context tokens, summary share, leaf and context evidence coverage, response-cache hit rate, and the treatment−control Δ F1 with CI), `control_check_leaf_identical_answers`, `gap_to_leaf` (with difference-in-differences), `original_treat_vs_matched_leaf_mix`, `index_build_cost` and `decision`.

**Pre-registered decisions** (each a paired document bootstrap, 95%):
- **Primary endpoints:** the difference-in-differences for collapsed@1000 and collapsed@2000. A gain is "supported" only if the 97.5% CI (Bonferroni over the two endpoints) excludes 0. Plain 95% CIs are reported alongside.
- **Hierarchical retrieval is competitive** if, under treatment, the lower bound of (collapsed − leaf_only)@B is above −1 F1 at either budget. If so, add hierarchical stage-2 targets to the cascade. That requires collapsed@2000 cap-512 answers on **train** too (about $1.2–1.6 more), so calibration still happens on train.
- **Cost is justified** if the extra index-build cost per paper, spread over that paper's questions (about 3.7 per paper), is below the F1-equivalent saving at matched inference cost. Both are reported, and the call is yours.
- If the leaf control is not about 100% identical, report it as drift, and interpret only the difference-in-differences.

## Experiment 2: sufficiency-triggered cascade

**Hypothesis.** A one-call evidence-sufficiency flag predicts *which questions gain from a larger budget* better than abstention does. That would let a cascade beat the leaf_only mix at matched cost.

**First-stage prompt** (`qasper_sufficiency_v1`, defined in `src/adaptive_raptor/generation/__init__.py`):
- It is the qasper_default prompt plus one final instruction to append "Evidence: SUFFICIENT / PARTIAL / INSUFFICIENT".
- The answer comes first, so the flag can't change how the answer starts.
- Everything up to the question matches qasper_default. A retry with a larger leaf budget therefore shares the instructions and the first chunks as a prompt prefix: about 370–410 tokens of a 2,000-token retry (`scripts/probe_prompt_cache.py`, plan-only output).
- `split_sufficiency()` removes the flag line before F1 is computed.
- A missing flag counts as INSUFFICIENT (escalate). The share of missing flags is reported.

**Design:**
- **Stage 1:** run the new prompt at leaf_only@400 and @1000, on train (`cascade_stage1_train.yaml`) and dev (`cascade_stage1_dev.yaml`). Run label S keeps these rows separate.
- **Stage 2:** reuse the stored qasper_default answers at leaf_only@1000 and @2000 from `labels_train` / `labels_dev`. That is the same prompt as the fixed baselines.
- **Costs:** count both calls in full, including the extra input (~40 tokens) and output (~6–8 tokens) for the flag.

**Candidates** (fixed list):
- Stage 1 at {400, 1000}.
- Triggers: {insufficient, insufficient|partial, abstain, abstain|insufficient, abstain|insufficient|partial}.
- Escalation targets: the larger leaf budgets.
- One two-level option: PARTIAL → 1000, INSUFFICIENT → 2000.

**Selection on train only:** the primary candidate is the one with the best train Δ against the leaf mix at matched tokens. Dev evaluates **only that primary** for the main claim; every other candidate is reported as exploratory.

**Inference-time availability:** the trigger reads only the stage-1 flag and the stage-1 answer text. Gold answers, types and evidence are used only to compute F1.

**Comparators on dev:**
- fixed leaf_only@1000;
- fixed leaf_only@2000;
- the leaf mix at matched tokens;
- the leaf mix at matched USD;
- the earlier abstention cascade (stored run-A leaf@400 answers, 400 → 2000).

All use a paired document bootstrap, 95%.

**Signal quality**, reported for train and dev:
- AUC for "escalating improves F1", for the flag, for abstention, and for the two combined;
- mean gain and positive-gain share for each flag level;
- the share of missing flags;
- the stage-1 format effect: F1 of the new prompt vs qasper_default at the same budget.

**Caching (sensitivity only):**
- DeepSeek caching is documented as best-effort.
- `probe_prompt_cache.py --execute` measures the retry's cache-hit share (10 calls, < $0.01).
- The primary USD figure assumes no caching. A with-cache figure uses the measured median: `analyze_cascade.py --cache-fraction <median>`.
- At most ~18% of the retry's input is cacheable, so caching can't rescue a cascade that loses at no-cache cost.

**Run order:**

| Step | Command | Est. cost |
|---|---|---|
| 2a. Stage 1, train | `uv run python -m adaptive_raptor.run --config configs/experiments/cascade_stage1_train.yaml` | $0.25 |
| 2b. Stage 1, dev | `uv run python -m adaptive_raptor.run --config configs/experiments/cascade_stage1_dev.yaml` | $0.23 |
| 2c. Cache probe | `uv run python scripts/probe_prompt_cache.py --execute` | < $0.01 |
| Analysis | `uv run python scripts/analyze_cascade.py --stage1-train outputs/cascade_stage1_train/<run> --stage1-dev outputs/cascade_stage1_dev/<run> [--cache-fraction X]` | $0 |

`analyze_cascade.py --simulate` is an offline dry run: abstention stands in for the flag. It reproduces the existing abstention cascade (0.4193 F1 at 1,527 tokens; +0.3 [−1.0, +1.6] vs the matched mix), which checks the code and cost accounting.

**Success criterion:** the primary cascade's 95% CI against the leaf mix at matched tokens **and** at matched USD lies above 0 on dev. Then the final test run is planned once, with the configuration frozen.

## Methodology notes

- **Dev reuse.** Experiment 1 is a fixed A/B with no tuning, so selection bias doesn't apply. In Experiment 2 all fitting and selection happen on train, and dev is read once for the primary. The 61-candidate search from the redesign diagnostics is not reused.
- **Generator noise.** About 20% of (question, config) pairs change F1 on a rerun. Paired designs, the cached leaf control and difference-in-differences limit its impact. CIs are paired document bootstraps with 2,000 resamples.
- **Reporting.** F1; input and output tokens; profiler tokens (none in either experiment); USD with and without caching; index-build USD. Point estimates and CIs are always given together.

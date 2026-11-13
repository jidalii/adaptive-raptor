# Project notes

Committed on purpose. Two sections: decisions (with the evidence behind them) and the spend log. Every ticket that calls a paid API appends a spend row here.

## Decisions and findings

### 2026-11-09: DeepSeek thinking mode (T03)

- `deepseek-flash` runs with **thinking enabled by default**. Measured live on the same 1-sentence summary request:

  | Request | `reasoning_content` | `reasoning_tokens` | Prompt tokens | Completion tokens |
  |---|---|---|---|---|
  | No `thinking` field | 159 chars | 36 | 57 | 58 |
  | `extra_body={"thinking": {"type": "disabled"}}` | none | not reported | 31 | 21 |

- Every DeepSeek request now sends `thinking` explicitly (`models/chat.py`). `generation.thinking` controls it for answers; RAPTOR summaries always disable it.
- Empty completions raise `EmptyCompletionError`. They are retried, then reported as an error, never stored as an empty answer or summary.
- **Cache invalidation:**
  - `CHAT_PROTOCOL_VERSION=2` is part of every generation and summary cache key.
  - `INDEX_FORMAT_VERSION=2`, and the RAPTOR index key now includes `chat_protocol`.
  - Cached thinking-mode outputs from Nov 6 can't be reused. The stale `indexes/qasper` and `.cache/summaries` were deleted.
  - `.cache/generations` still holds the unreachable Nov 6 entries (32 KB); they're safe to delete.
- **WP0 probe (for the LLM-profiler plan):** with thinking disabled, `response_format={"type": "json_object"}` and `logprobs=True` are both accepted, and logprobs come back.
  - **Correction (same day): "logprob confidence is feasible" was wrong.** See the optimizer section below. At temperature 0 every returned logprob is 0.0.
- The API still reports only the alias `deepseek-flash` as the model version. Record run dates.

### 2026-11-09: Pilot subset (T02)

- `dev_pilot`: 30 papers / 87 questions, `--seed 0`. Answer types of the first annotator: 36 extractive, 30 abstractive, 11 boolean, 10 none.
- `dev_rest`: the other 251 papers / 918 questions, `--seed 1`, excluding the pilot.
- The two subsets are disjoint and cover all 281 dev papers. A rerun reproduced byte-identical files.
- Copies are in `results/question_ids/`.

### 2026-11-09: First valid RAPTOR indexes (T04)

- F-100, seed 224, first 3 pilot papers:

  | Paper | Nodes per layer |
  |---|---|
  | 1906.07662 | 55 / 11 |
  | 2001.02284 | 99 / 19 / 5 |
  | 2002.08795 | 42 / 6 |

  All 41 summary nodes are non-empty (63–87 words). Summary nodes are retrieved now: 65 of the 330 nodes retrieved under collapsed@2000 and `raptor_original`.
- The rerun reused all 3 indexes and served all 20 answers from cache, with identical F1.
- **Open decision for T08 (fidelity vs. quality):** every summary uses exactly `summarization_length=100` output tokens and is cut mid-sentence. Upstream RAPTOR does the same, since it passes `max_tokens=summarization_length` and its default is 100. Options:
  1. Keep 100 for fidelity to upstream.
  2. Raise it to about 200 and record the change in `UPSTREAM.md`.
  3. Ask for brevity in the prompt (this changes upstream's verbatim prompt).

  Decide in the pilot. Any change alters the index key, so indexes get rebuilt automatically.
- 3-paper sanity numbers (n=10 questions; not a result):

  | Config | F1 | Mean context tokens |
  |---|---|---|
  | collapsed@2000 | 0.453 | 1,971 |
  | `raptor_original` | 0.412 | 856 |

### 2026-11-09: Labels and evaluation harness (T06, T07)

- Live check on pilot paper 1 (3 questions): run A (`qasper_default`) and run B (`qasper_paraphrase`) give 27 rows each. A rerun of A made no API calls.
- Merging the three `labels.jsonl` tables gives 81 rows → 54 unique, with **0 conflicts**: the rerun reproduced every F1.
- On these 3 questions, in-sample oracle F1 at 1,000 tokens was 0.472, against 0.324 cross-fitted on run B. This illustrates why the proposal cross-fits; it is not a result.
- The cost meter counted cache hits as spend. `summary.json` now has `cost_usd` (workload cost if uncached, the research metric) and `billed_usd` (new spend in the run).
- **Budget-matching rule as implemented:** the comparator is the **cheapest dev-frontier config whose dev mean tokens are ≥ the router's** (the best fixed config at the smallest budget not below the router's). The proposal sentence "the best fixed configuration whose mean tokens are at least the router's" could also be read as "highest F1 among all configs ≥ the router's tokens". That reading almost always picks collapsed@2000, so it would compare against more context, not matched context. Confirm before T12.

### 2026-11-09: LLM-profiler router, Plan A (optimizer track)

- **Decision (user):** build only the LLM-profiler router. The embedding classifier (Plan B) is built only if the LLM router shows no improvement on dev.
- **Implemented:**
  - `optimizer/profiler.py`, `optimizer/__init__.py`, and the `query_profiling` and `router_train` experiments;
  - router fields in traces, and offline tests in `tests/test_optimizer.py`.
  - Design and work packages are in `docs/OPTIMIZER_PLAN.md`.
- **Pilot profiling** (87 dev-pilot questions, prompt `profile_v1`, hash `db7f8941f60356e5`, inputs = question + title):
  - 0 invalid JSON and 0 retries.
  - `answer_form`: 38 short_span / 34 explanation / 15 yes_no. `granularity`: 27 specific / 50 section / 10 whole-paper. `n_facts`: 37 / 36 / 14 for 1 / 2 / 3.
  - All 11 boolean-gold questions came out `yes_no`. `unanswerable_likely` was never predicted, which is expected: the profiler can't know.
- **Profiler overhead: 332 input tokens per query on average** (plus 33 output tokens). That's 33% of a 1,000-token target, above the 200–300 estimate in the plan. The prompt's definitions and two few-shot examples make up most of it.
  - A shorter `profile_v2` prompt, or relying on DeepSeek prefix caching, is the main lever. Either needs a decision (see below).
- **Logprob confidence is unusable at temperature 0.** DeepSeek returns logprobs *after* temperature:
  - at t=0 all 2,829 returned logprobs were exactly 0.0, and `top_logprobs` alternatives were −9999;
  - at t=1 they are informative: for example, p ≈ 0.30 on the first token of `answer_form`.
  - Code changes: `use_logprobs` now defaults to false, an all-zero logprob sequence yields `confidence=None` (never a fake 1.0), and validation warns.
  - The METIS-style confidence fallback is therefore **inactive**. Only invalid profiles go to the fallback.
- **Temperature 0 isn't deterministic:** re-profiling the same 87 questions changed 14 profiles (16%).
  - Experiments stay reproducible, because each question is profiled once and the stored `profiles.jsonl` is reused for train, dev and test.
  - Canonical pilot profiles: `outputs/profile_queries/20261109T184850Z`. The earlier run `…184549Z` used logprobs and has fake confidences, so don't use it.
- **Open decisions:**
  1. **Confidence signal:**
     - (a) none: invalid profiles only fall back (the current default);
     - (b) profile at t=1 and use logprobs, which makes the profile a sample;
     - (c) self-consistency, k samples with agreement as confidence, at k× the profiler cost.
  2. **Profiler prompt length** (`profile_v1` at 332 tokens vs. a shorter v2).

### 2026-11-09: Decisions after the first profiling run (user)

- **Confidence: option (a), none.** `confidence_threshold: null` is now the default. Only invalid profiles (after one stricter retry) go to the fallback. Temperature-1 logprobs and self-consistency are not implemented.
- **Profiler prompt frozen: `profile_v3`, `prompt_hash e22d6935b289c85d`.**
  - Chosen against v1 and v2 on the 87 pilot questions. Each prompt was run twice, the second run uncached:

    | Prompt | API input tokens | Valid JSON | Boolean → `yes_no` | Non-boolean → `yes_no` | Identical profiles on an uncached rerun |
    |---|---|---|---|---|---|
    | v1 | 331.8 | 87/87 | 11/11 | 4 | 73/87 |
    | v2 | 158.8 | 87/87 (1 retry) | 11/11 | 5 | 80/87 |
    | **v3** | **139.7** | **87/87** | **11/11** | 5 | **81/87** |

  - The version is part of `prompt_hash`, which keys the response cache, the `ProfileStore` and the router checkpoint, so v1/v2 profiles can't be mixed with v3. A test pins the v3 hash.
  - Canonical v3 pilot profiles: `outputs/profile_queries/20261109T185646Z`.
  - v2 and v3 drop the few-shot examples and `unanswerable_likely` (v1 never predicted it). The schema still accepts it.
- **There is no ground truth for profile fields, so "profiling quality" is measured only by proxies:** JSON validity, `yes_no` recall on boolean-gold questions, and rerun stability.
  - Prompts disagree a lot with each other. Full-profile agreement is v1–v3 36/87 and v2–v3 56/87; `granularity` and `n_facts` move the most.
  - Under v3, `n_facts` is nearly bimodal: 51 at 1, 7 at 2, 28 at 3.
  - Whether a prompt is *useful* can only be judged by dev routing results. The prompt was chosen without looking at any F1, so dev and test are untouched by this choice.
- **Limitation, profile instability:** DeepSeek at temperature 0 isn't deterministic. On an uncached rerun, 16% of v1 profiles and 7% of v3 profiles (6/87) changed.
  - Mitigation: profile each question once and reuse the stored `profiles.jsonl` for training, calibration and evaluation. A deployed router would see this noise, so report it as a threat to validity.
- **Profile-cell sparsity (A1):** v3 on 87 questions gives 16 full cells, 5 of them singletons; (answer_form, granularity) gives 8 cells.
  - With about 1k train questions, most full cells should hold tens of queries.
  - A1 already backs off hierarchically (full cell → form|granularity → form → global; Beta prior strength `table_alpha=5`).
  - `router_train` reports cell sizes and the share of train queries in cells below 10, so sparsity is checked on real labels before any extra smoothing.
- **WP10 (H1) is a replay, not an online run.**
  - Test labeling (T12) executes all 9 configs per test question. The router's answer for a question is the recorded answer of the config it picks, which is the same retrieval and the same cached generation an online run would produce.
  - Nothing is fitted or selected on test. θ, the cost model and the matched comparator under both rules are frozen in the `router_train` checkpoint (dev only).
  - H1 refuses to run if an evaluated paper was used for training or calibration, or if any question lacks a stored profile.
  - Reported separately per system: F1, context tokens, profiler tokens, total tokens, USD.
- **Matching-rule refinement:** the conservative comparator is the best fixed config within a budget of at least the router's dev total tokens.
  - If more context stops helping, a cheaper dominating config is the comparator. Picking a dominated config just because it uses more tokens would weaken the baseline, the opposite of conservative.
  - The rule is flagged unsatisfied only when no fixed config reaches the router's tokens. The mock smoke run surfaced this case.

### 2026-11-09: T08 pilot with the agreed parameters

Parameters (user): budgets 400/1000/2000 (caps on retrieved context; actual tokens recorded), ε = 0.02, summary cap 200 tokens. Runs: `outputs/pilot_catalog/20261109T193103Z` (run A, `qasper_default`) and `…194522Z` (run B, `qasper_paraphrase`). 30 dev-pilot papers, 87 questions × 9 configs. 30 F-100 indexes, 310 summary nodes.

**Configs (run A):**

| Config | F1 | Mean context tokens | Context p10 | Nodes | Gen in / out |
|---|---|---|---|---|---|
| leaf_only@400 | 0.354 | 350 | 325 | 4.2 | 414 / 7.1 |
| leaf_only@1000 | 0.350 | 961 | 929 | 11.5 | 1029 / 8.0 |
| leaf_only@2000 | **0.416** | 1933 | 1912 | 23.2 | 2012 / 9.5 |
| summary_only@400 | 0.300 | 263 | 200 | 1.3 | 324 / 6.9 |
| summary_only@1000 | 0.327 | 865 | 802 | 4.3 | 922 / 8.2 |
| summary_only@2000 | 0.322 | 1538 | 808 | 7.7 | 1593 / 9.7 |
| collapsed@400 | 0.319 | 341 | 262 | 3.3 | 404 / 7.4 |
| collapsed@1000 | 0.361 | 938 | 874 | 8.8 | 1001 / 11.8 |
| collapsed@2000 | 0.380 | 1938 | 1857 | 18.2 | 2005 / 14.3 |

**Findings:**
- **The 400 budget is practical.**
  - No configuration retrieved an empty context.
  - leaf_only@400 (0.354) matches leaf_only@1000 (0.350) at about a third of the tokens.
  - Caveat: summary_only@400 fits only 1–2 summaries (1.3 nodes on average; p10 = one 200-token summary).
  - summary_only@2000 is not budget-bound: trees have too few summaries to fill 2,000 tokens (p10 808).
- **The 200-token summary cap still truncates (problem; parameter not changed):**
  - 292 of 310 summary calls (94%) used exactly 200 output tokens.
  - Only 10.6% of summary nodes end in sentence punctuation; median 201 cl100k tokens.
  - Truncation applies equally to every strategy, but summaries are clipped mid-sentence, which may handicap summary_only and collapsed. Awaiting a decision before collecting labels.
- **Collapsed retrieval rarely picks summaries:** 18–20% of retrieved nodes come from summary layers.
- **Monotonicity:** collapsed is monotone in budget; leaf_only and summary_only aren't. Per (question, strategy), 83.5% are monotone. This fits A1, which doesn't assume monotonicity.
- **Labels (ε = 0.02):**
  - 60 labeled and 27 all-fail (31%; abstractive 12/27, extractive 11/37, boolean 4/10, none 0/13).
  - Acceptable rates are 0.45–0.65 per config.
  - ε = 0.05 changes almost nothing, because F1 is lumpy.
  - `none`-type questions accept 8.2 of 9 configs: "Unanswerable" is right regardless.
- **Run A vs. run B:** 19% of (question, config) pairs get a different F1 under the paraphrased prompt. That's large generator noise, so cross-fitting matters.
- **Oracle headroom (cross-fitted: chosen on A, scored on B; at each frontier config's context tokens):**

  | Context tokens | Best fixed | Cross-fit oracle (gap) | In-sample oracle |
  |---|---|---|---|
  | 263 | 0.300 | 0.351 (+5.2) | 0.384 |
  | 341 | 0.319 | 0.405 (+8.7) | 0.438 |
  | 350 | 0.354 | 0.405 (+5.1) | 0.438 |
  | 938 | 0.361 | 0.447 (+8.6) | 0.491 |
  | 1933 | 0.416 | 0.447 (+3.1) | 0.491 |

  Against the go/no-go threshold of about 2× the 3-F1 target (≈ 6 F1), headroom is enough in the middle of the frontier (+8.6 at about 1,000 tokens) but not at 2,000 (+3.1). n = 87, so these are noisy.
- Not run: the 3-seed variance check (deferred; it costs about $0.35).

### 2026-11-09: Train/dev profiling (v3, hash e22d6935b289c85d)

- **Train** (`train_router`: 340 papers, 1,019 questions): `outputs/profile_train/20261109T194817Z`.
  - 0 invalid, 4 retried.
  - 141.5 input / 32.8 output tokens; $0.042.
- **Dev** (`dev_rest`: 251 papers, 918 questions): `outputs/profile_dev/20261109T195035Z`.
  - 0 invalid, 1 retried.
  - 140.2 input / 32.8 output tokens; $0.037.
- **Distribution (train):**
  - `answer_form`: 538 short_span / 291 explanation / 190 yes_no;
  - `granularity`: 567 specific / 307 section / 145 whole-paper;
  - `n_facts`: 649 / 89 / 271 / 4 / 6 for 1 / 2 / 3 / 4 / 5;
  - `cross_section`: 111 true.
- Dev has a similar mix.
- `original RAPTOR` on the pilot (`…195138Z`): F1 0.368 at 1,080 context tokens; $0.013.

## T12 dev results (2026-11-09, awaiting review; test not run)

Runs: labels `outputs/labels_{train,dev}/20261109T195548Z`; routers `outputs/router_train_llm_profile_table/20261109T230225Z` (A1) and `outputs/router_train_llm_profile_rules/20261109T230228Z` (A2). Dev = `dev_rest` (918 questions, 251 papers). Total tokens = generation in + out + profiler.

| System | Dev F1 | Total tokens/query |
|---|---|---|
| A1 profile table (θ = 0.44) | 0.336 | 1,014 |
| A2 rules | 0.337 | 1,002 |
| collapsed@1000 (A1's matched comparator) | 0.376 | 1,011 |
| leaf_only@400 (A2's matched comparator) | 0.339 | 425 |
| leaf_only@1000 | 0.400 | 1,039 |
| raptor_original | 0.387 | 1,164 |

- A1 vs matched fixed: −4.0 F1 [−5.8, −2.3], worse. A2: −0.2 [−1.7, +1.3], "equivalent", but at 2.4× the tokens of its comparator.
- The profile groups don't change the config ranking. leaf_only ≥ collapsed > summary_only, and more budget helps, in nearly every answer_form / granularity / n_facts group. So the profile carries almost no routing signal on strategy.
- The A1 selector (cheapest config with P ≥ θ) picks dominated configs: collapsed@400 and summary_only@400.

## Optimizer redesign: offline diagnostics (2026-11-09; no API calls, test untouched)

Script `scripts/offline_diagnostics.py` → `outputs/diagnostics/t12_redesign/report.json`. Dev, 918 questions.
"Matched fixed" = leaf_only budgets mixed per question by a fixed hash so the mean cost matches the candidate.

- **Strong baselines:** leaf_only@1000 is 0.400 at 1,039 tokens, +1.2 [0.0, +2.4] vs original RAPTOR. leaf_only@2000 is 0.432 at 2,043 tokens, +4.5 [+3.0, +6.0].
- **Oracle (in-sample):**
  - The 9-config oracle is 0.519 at ≤ 700 tokens, so it is not budget-bound at 1,039 or 1,164.
  - Leaf-budget-only oracle: 0.469. Leaf + collapsed: 0.495.
  - Generator noise inflates these numbers. Rerunning the same config (pilot runs A/B) gives +2–4 F1 of "oracle" gain from noise alone. The pilot cross-fit oracle at ≈1,000 tokens was +8.6 over best fixed, against an in-sample +13.
- **Retrieval features:**
  - Recomputed from the cached trees with local SBERT; they reproduce the stored leaf ranking on 40/40 sampled questions.
  - Predictability is weak. AUC on dev is 0.50–0.53 for retrieval-only features on "bigger budget helps", and ≤ 0.66 for "collapsed beats leaf".
- **Candidates:** 61 policies tried (ridge / GBDT / rules / cascades, trained on train only). **None is significantly better than matched fixed; 43 are significantly worse.** The best is answer cascade leaf@400 → leaf@2000 on abstention: +0.3 [−1.0, +1.6] at 1,527 tokens.
- **Abstention signal:**
  - leaf@400 abstains on 54% of questions; only 20% of those are gold-unanswerable.
  - Escalating abstainers to leaf@2000 gains +14.8 F1, against +2.8 for the others.
- **A1 losses vs leaf@1000 (183):**
  - By budget and strategy: 116 at budget < 1000; 135 collapsed, 43 summary_only.
  - By evidence: in 60 the chosen context misses gold evidence that leaf@1000 has; in 118 both contexts contain it.
- **Summary bottleneck:**
  - 94% of dev summaries hit the 200-token cap, and 90% don't end in sentence punctuation.
  - collapsed@1000 loses more to leaf@1000 the more summaries it retrieves: −0.1 / −1.4 / −2.5 / −5.8 F1 for 0 / ≤ 20 / ≤ 40 / > 40% summary nodes.
  - Evidence coverage falls with it, from 0.96 to 0.80.

## Prepared, not run (2026-11-09): summary-cap ablation + sufficiency cascade

- **Protocol, commands and cost estimates:** `docs/NEXT_EXPERIMENTS.md`.
- **Code (additive):**
  - prompt `qasper_sufficiency_v1` and `split_sufficiency()`;
  - traces record `sufficiency` and the provider's `prompt_cache_hit_tokens`.
- **Configs:** `summary_cap512_dev.yaml`, `cascade_stage1_{train,dev}.yaml`.
- **Scripts:** `analyze_summary_cap.py`, `analyze_cascade.py` (with a `--simulate` dry run), `probe_prompt_cache.py` (makes calls only with `--execute`), `estimate_cost.py`.
- **Official deepseek-flash prices** (checked 2026-11-09), off-peak per 1M tokens: $0.15 input on a cache miss, $0.003 on a cache hit, $0.60 output. Peak hours (01–04 and 06–10 UTC, weekdays) cost 2×.
- **Earlier spend figures** assume off-peak prices and no cache hits. Calls made during peak hours would have cost more; cache hits would have cost less.

## Results 2026-11-10: summary-cap pilot and sufficiency cascade (dev; test untouched)

Reports: `outputs/diagnostics/summary_cap_pilot/`, `outputs/diagnostics/cascade/`, `outputs/diagnostics/cache_probe.json`. Charts: `outputs/figures/next/`.

**Summary cap 512, pilot (87 questions): failed the pre-registered gate, so the full ablation was not run.**
- At the cap: 93% → 33% of summaries. Ending mid-sentence: 89% → 29%. Median length 201 → 423 tokens.
- Best evidence recall of any summary: +14.7 points [+10.4, +19.4].
- The leaf control is identical across arms (good).
- No F1 change is significant at n = 87:
  - collapsed@1000: +1.5 [−2.5, +6.2]
  - collapsed@2000: −0.8 [−4.5, +3.2]
  - original RAPTOR: +2.6 [−2.0, +7.2], but its context grows 1,080 → 1,500 tokens, because it takes top-10 nodes, not a token budget.
- Longer summaries displace leaf evidence. Leaf-evidence coverage drops for collapsed@1000 (0.74 → 0.62) and collapsed@2000 (0.94 → 0.88).

**Cache probe:**
- Without a nonce, the retry hit 1,792–1,920 tokens, because earlier runs had sent identical prompts.
- With a fresh nonce, it hit 384 tokens, about 97% of the shared prefix.
- Recorded stage-1 hits (~70%) are inflated by earlier runs; the analysis does not use them.

**Sufficiency cascade:**
- The primary, chosen on train, is: stage 1 leaf@1000 with the sufficiency prompt, escalating to leaf@2000 when the answer abstains. Dev F1 0.435 at 1,932 tokens.
- Against the matched-token leaf mix: +1.0 [−0.2, +2.1], not significant. Against the matched-USD mix: +0.8 [−0.5, +1.9]. Against leaf@2000: +0.3 [−0.9, +1.5].
- With the probe-measured prefix caching, it is +1.8 [+0.6, +3.0] against the mix at the same cached USD. This is a sensitivity result.
- The flag behaves almost like abstention. AUC for "escalation helps" is 0.51 vs 0.50. Mean gain: INSUFFICIENT +13.6, SUFFICIENT −0.6.
- **Confound:** the sufficiency prompt itself raises F1 at the same context. leaf@400: +1.8 [+0.8, +2.8]. leaf@1000: +1.1 [−0.2, +2.2].

## Results 2026-11-10 (2): summary-prompt pilot and prompt-controlled cascade (dev; test untouched)

Reports: `outputs/diagnostics/summary_arms_pilot/`, `outputs/diagnostics/cascade/` (`prompt_controlled`). Charts: `outputs/figures/next/summary_arms_*`, `cascade_prompt_control`.

**Three effects, reported separately:**

1. **Index quality (summary prompt `evidence_dense_v1`, pilot n = 87).**
   - Summaries now finish. At the cap: 93% (cap 200) → 1%. Ending mid-sentence: 89% → 1%. Median length 232 tokens, against 423 at cap 512 with the upstream prompt.
   - Evidence recall of the best summary is unchanged vs cap 200: +0.8 [−5.2, +5.9]. It is lower than at cap 512 (−13.9), which had longer summaries.
   - Evidence recall per 100 summary tokens: 0.28, vs 0.33 (cap 200) and 0.20 (cap 512).
   - F1 vs cap 200:
     - original RAPTOR +3.1 [+0.6, +6.1], but its context also grows 1,080 → 1,153 tokens (+7%);
     - collapsed@2000 +2.7 [−1.5, +7.8];
     - collapsed@1000 +2.0 [−2.4, +6.7].
   - Leaf answers are identical across arms, so the difference-in-differences equals the F1 delta.
   - Gap to leaf: collapsed@1000 is +3.0 [−0.4, +6.7] above leaf@1000 (pilot leaf@1000 is low). collapsed@2000 is −0.9 [−4.6, +3.1] below leaf@2000.
   - Index cost: ~$0.0024 per paper, vs $0.0021 (cap 200) and $0.0034 (cap 512). Inference cost per config is unchanged.
2. **Prompting.** The sufficiency answer prompt, at the same context, changes F1 by:
   - leaf@400: +1.8 [+0.8, +2.8]
   - leaf@1000: +1.1 [−0.2, +2.2]
   - leaf@2000: +1.5 [+0.1, +2.9]

   New-prompt leaf@2000 reaches 0.447.
3. **Adaptivity.**
   - The prompt-matched cascade (both stages on the new prompt) vs the matched-USD new-prompt leaf mix: **−0.3 [−1.3, +0.6]**. Against new-prompt leaf@2000: −0.7 [−1.8, +0.3].
   - Only the cache-cost sensitivity is positive: +1.5 [+0.6, +2.5].
   - **Decision (user rule): stop the cascade design.**

## Spend log

Prices: user-reported off-peak rates, $0.15 per 1M input and $0.60 per 1M output tokens.

| Date | Ticket | What | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|---|
| 2026-11-06 | T03/T04 | First baseline, 2 dev papers (thinking on; summaries empty, unusable) | n/a | n/a | n/a | ≈ 0.005 |
| 2026-11-09 | T03 | Thinking / JSON / logprobs probe | 3 | 145 | 87 | < 0.0001 |
| 2026-11-09 | T03 | Integration tests (1 answer + 1 summary) | 2 | ≈ 150 | ≈ 120 | < 0.0001 |
| 2026-11-09 | T04 | Baseline on 3 pilot papers: summaries | 41 | 19,305 | 4,100 | 0.0053 |
| 2026-11-09 | T04 | Baseline on 3 pilot papers: answers (2 configs × 10 questions) | 20 | 29,574 | 413 | 0.0047 |
| 2026-11-09 | T06 | Live label check: 1 pilot paper × 9 configs × runs A/B (44 new calls; 10 served from cache), plus a rerun of A (0 new calls) | 44 | ≈ 42,000 | ≈ 640 | ≈ 0.007 |
| 2026-11-09 | Optimizer | Pilot profiling × 2 (87 questions each; second run without logprobs) + 1 integration call + 2 logprob probes | 177 | ≈ 58,600 | ≈ 5,900 | ≈ 0.013 |
| 2026-11-09 | Optimizer | Prompt validation: v2 × 2 and v3 × 2 runs on the 87 pilot questions | 348 | ≈ 52,000 | ≈ 11,400 | ≈ 0.015 |
| 2026-11-09 | T08 | Pilot: 30 indexes (310 summaries) + 9 configs × 87 questions × runs A/B | 1,876 | ≈ 2.06 M | ≈ 74,000 | ≈ 0.32 |
| 2026-11-09 | Optimizer | Profiling train (1,019) + dev_rest (918) with v3; `raptor_original` on the pilot (87) | 2,029 | ≈ 0.38 M | ≈ 64,000 | ≈ 0.09 |
| 2026-11-09 | T12 | Train labels: 340 indexes (4,106 summaries) + 9 configs × 1,019 questions, run A | 13,277 | ≈ 12.20 M | ≈ 0.90 M | ≈ 2.37 |
| 2026-11-09 | T12 | Dev labels: 251 indexes (2,810 summaries) + 9 configs + `raptor_original` × 918 questions, run A | 11,990 | ≈ 11.64 M | ≈ 0.64 M | ≈ 2.13 |
| 2026-11-10 | Optimizer | Cascade stage 1 (sufficiency prompt), leaf@400/1000 on train (1,019) + dev (918); nominal miss-price cost, ~70% of input was actually a provider cache hit | 3,874 | ≈ 2.95 M | ≈ 64,000 | ≤ 0.49 |
| 2026-11-10 | Optimizer | Prompt-cache probe, 2 runs × 10 calls (second with nonce) | 20 | ≈ 25,000 | ≈ 500 | ≈ 0.005 |
| 2026-11-10 | Optimizer | Summary-cap 512 pilot: 30 indexes (313 summaries) + 5 non-leaf configs × 87 (leaf answers from cache) | 748 | ≈ 0.80 M | ≈ 0.13 M | ≈ 0.20 |
| 2026-11-10 | Optimizer | Prompt control: fixed leaf@2000 with the sufficiency prompt, dev (918) | 918 | ≈ 1.91 M | ≈ 18,000 | ≤ 0.30 |
| 2026-11-10 | Optimizer | Summary-prompt pilot (evidence_dense_v1, cap 512): 30 indexes (309 summaries) + 5 non-leaf configs × 87 | ≈ 744 | ≈ 0.78 M | ≈ 0.08 M | ≈ 0.17 |
| 2026-11-10 | Optimizer | Summary-prompt full dev (evidence_dense_v1, cap 512): 249 indexes (2,806 summaries) + 5 non-leaf configs × 911 questions (leaf answers from cache) | 7,368 | ≈ 8.00 M | ≈ 0.77 M | ≈ 1.66 |
| | | **Total to date** | | | | **≈ 7.78** |

Index build cost ≈ **$0.0018 per paper** (F-100, about 14 summaries per paper). Projection, before PIC and seeds: 600 papers × 1 variant ≈ $1.1.

## Pre-registration: summary_dense_dev (written 2026-11-10 12:50 UTC, before the run)

Approved by Jida. Config: `configs/experiments/summary_dense_dev.yaml`.
Treatment: evidence_dense_v1 summary prompt, cap 512 (safety limit).
Control: `outputs/labels_dev/20261109T195548Z`, which has cap 200 and the upstream prompt. Both arms use the same 918 dev_rest questions.
Budgets are fixed at 1000/2000. Nothing is tuned. Test is untouched.

All endpoints use a paired document bootstrap with 2,000 resamples.

**Primary endpoints.** Bonferroni over the two endpoints; "supported" only if the 97.5% CI excludes 0:
1. Difference-in-differences for collapsed@2000: (collapsed − leaf_only)@2000, treatment minus control.
2. Original RAPTOR (treatment) vs a leaf_only mix matched to its mean USD.

**Secondary endpoints** (95%, descriptive):
- difference-in-differences for collapsed@1000 and summary_only@1000/2000;
- treatment − control F1 per config;
- summary completeness: share at the cap and share unfinished;
- best-summary evidence recall and density;
- leaf displacement;
- index USD per paper.

**Control check.** leaf_only answers should be identical across the two arms (response cache).

**Decision rule.**
- If neither primary endpoint is supported, report "index-quality gain not confirmed at n=918" and do not claim it.
- If at least one primary endpoint is supported, the dense prompt becomes the default index for the joint experiments with PIC chunking.

## Results: summary_dense_dev (2026-11-10, pre-registered above)

Run `outputs/summary_dense_dev/20261110T124849Z`. Analysis: `outputs/diagnostics/summary_dense_dev/summary_cap_report.json`. Charts: `outputs/figures/summary_dense_dev/`.

**Exclusions.** 2 of the 251 papers (1703.07090 and 1811.08603) failed to build an index. Upstream RAPTOR's GMM raised "ill-defined empirical covariance" when clustering the new summaries. The build is seeded and deterministic, so a rerun would fail again, and fixing it would mean editing the vendored RAPTOR, so it was left alone. Their 7 questions are excluded from both arms, leaving n = 911 paired questions over 249 papers. The run status is "failed" only because of these 7 errors. The leaf control was 99.8–100% identical across arms.

**Primary endpoints (pre-registered; Bonferroni, 97.5% CI).**

| Endpoint | Estimate | 97.5% CI | Supported? |
|---|---|---|---|
| Difference-in-differences, collapsed@2000 | +1.0 F1 | [−0.7, +2.5] | **no** |
| Original RAPTOR (dense) vs leaf mix at matched cost ($0.200 vs $0.201 per 1k questions) | −0.6 F1 | 95% CI [−2.2, +0.9]; the 97.5% CI is wider | **no** |

**Decision (pre-registered rule): the index-quality gain is not confirmed at n=918.** Hierarchical retrieval with the new summaries still does not beat leaf_only at the same budget or at matched cost.

**Secondary (95%, descriptive):**
- Summaries are complete: 3.2% at the cap, 3.2% unfinished (control: 94.5% / 90.4%). Median length 238 tokens, p90 362.
- Best-summary evidence recall rose from 0.604 to 0.665, a gain of +6.2 [+4.9, +7.5] points (n=814).
- Treatment − control F1:
  - collapsed@1000: +1.5 [−0.1, +3.1]
  - collapsed@2000: +1.0 [−0.5, +2.3]
  - original RAPTOR: +1.9 [+0.5, +3.3] (context 1,086 → 1,220 tokens)
  - summary_only@1000: +7.2 [+5.1, +9.4]
  - summary_only@2000: +8.3 [+6.2, +10.2]
- Gap to leaf_only under the treatment:
  - collapsed@1000: −0.8 [−2.3, +0.7]
  - collapsed@2000: −0.9 [−2.4, +0.5]
  - summary_only@1000: −2.9 [−4.9, −0.9]
  - summary_only@2000: −3.0 [−5.2, −0.9]
  - Under the control, collapsed was −2.3 / −1.9 and summary_only −10.1 / −11.3.
- Leaf displacement is small. collapsed@1000 leaf-evidence coverage went from 0.828 to 0.792, and collapsed@2000 from 0.958 to 0.947.
- Index cost is $0.0027 per paper, up from $0.0023. Inference cost per config is unchanged.

**Interpretation.** The prompt clearly makes the summaries better: they are complete, carry more evidence, and summary-only retrieval gains 7–8 F1. Hierarchical retrieval moves from significantly worse than leaf_only to roughly level with it (CIs include 0). But it does not overtake leaf_only, and neither primary endpoint passes. The pilot's +3.1 for original RAPTOR shrank to +1.9 against its own control and −0.6 against the matched leaf mix, which is regression to the mean from the n=87 pilot. Defensible claim: "complete summaries remove most of RAPTOR's hierarchical-retrieval deficit on QASPER, but do not make it better than flat leaf retrieval".

## Sentence-level extractive compression, Phase 1: offline feasibility (2026-11-10, $0)

The run used no API calls and rebuilt no trees. The test split was not read.

- Code: `src/adaptive_raptor/compression.py` (tests: `tests/test_compression.py`).
- Driver: `scripts/compression_feasibility.py`.
- Config and pre-registered rule: `configs/analysis/compression_feasibility.yaml`.
- Output: `outputs/diagnostics/compression_feasibility/20261110T174954Z/` (report.json, tables, per-question rows, 4 plots).

**Setup**
- Source: the stored leaf_only@2000 retrievals. Leaf text comes from the cached trees.
- Units: the sentence units that upstream `split_text` built the chunks from (boundaries are runs of 2+ spaces; decimal and abbreviation fragments are merged back).
- Pool: about 61 units per question.
- Scorers:
  - BM25, with IDF over the paper's units;
  - SBERT cosine, using the retrieval embedder run locally;
  - RRF hybrid of the two;
  - random, as a floor.
- Selection: greedy under 600/1000/1400 tokens, Jaccard dedup at 0.8, optional predecessor rule. Output is in document order.
- Chunk baselines: leaf_only@2000 truncated by rank. This reproduces the stored leaf_only@400/@1000 for 100% of questions.

**Dev results.** n=918; 820 with highlighted evidence, 598 with an extractive answer.

| Context | Tokens | Highlighted recall | Answer in context |
|---|---|---|---|
| leaf_only@1000 | 968 | 0.870 | 0.488 |
| leaf_only@1400 (truncated) | 1369 | 0.924 | 0.552 |
| leaf_only@2000 | 1968 | 0.958 | 0.600 |
| sbert+nb@1000 | 952 | 0.873 | 0.515 |
| sbert+nb@1400 | 1330 | 0.921 | 0.570 |

The variant sbert+nb was selected on train.

**Sentence selection vs chunks at the same budget** (paired bootstrap, 95%):

| Budget | Highlighted recall | Answer in context | Paragraph recall |
|---|---|---|---|
| 600 | +0.8 [−0.4, +2.2] | — | — |
| 1000 | +0.3 [−0.5, +1.2] | +2.7 [−0.2, +5.4] | −2.5 [−3.2, −1.7] |
| 1400 | −0.3 [−0.9, +0.3] | +1.8 [−0.3, +3.9] | −1.9 [−2.4, −1.3] |

On train, sbert+nb@1000 was below chunks: 0.834 vs 0.846.

- BM25 is significantly worse than SBERT: −5.3 / −5.1 / −2.6 highlighted recall at the three budgets. Hybrid falls in between. Random is clearly the worst, so the scorers do carry signal.
- The predecessor rule has no measurable effect.

**Pre-registered decision**
- (a) is not met: no gain at 1000.
- (b) is technically met: 96.1% of leaf@2000's highlighted recall with 32.4% fewer tokens.
- **But (b) was mis-specified.** It does not compare against chunks at the same budget, and plain leaf_only@1400 also satisfies it (96.5% kept, 30.4% fewer tokens). The token saving comes from the smaller budget, not from sentence selection.

**Reading.** At matched tokens, sentence selection gives the same evidence coverage as chunk-level retrieval, not more. The f100 chunks are already about 4 sentences long and are ranked by the same SBERT model, so there is little granularity left to exploit. Sentence selection also loses some paragraph context. The only hint of a benefit is answer-in-context, +2–3 points, with CIs touching 0.

**Recommendation:** do not run Phase 2 as a "compression" claim. If F1 is still wanted, the fair test is compressed@1000/1400 against leaf_only@1000/1400 under the same prompt, at about $0.55 on dev. Expect a null result.

## Section 5 diagnostics: oracle headroom, profiler predictive power, accounting (2026-11-10, $0)

`scripts/section5_diagnostics.py` → `outputs/diagnostics/section5/20261110T181451Z/` (report.json, tables.md, headroom_curve, profile_interactions, best_strategy_by_profile). No API calls; test untouched. Catalog: the 9 configs, qasper_default prompt. Cost = generation in+out tokens (+ profiler tokens wherever a profile is used).

**A. Oracle headroom (dev, n=918).** Fixed = the best fixed policy (mixes of single configs). Oracle = per-question choice by true F1 under a mean-cost constraint (Lagrangian).

| B (tokens/q) | Fixed | Oracle | Headroom | Profile-cell oracle − fixed (profiler charged) | … profiler free | Pilot: share replicating on rerun |
|---|---|---|---|---|---|---|
| 700 | 0.366 | 0.519 | +15.5 [13.7, 17.7] | +0.4 [−0.9, +1.8] | +2.4 [+1.3, +3.6] | 0.81 |
| 1000 | 0.396 | 0.519 | +12.1 [10.4, 13.8] | +0.5 [−0.4, +1.4] | +1.5 [+0.4, +2.7] | 0.74 |
| 1500 | 0.415 | 0.519 | +10.5 [8.8, 12.1] | +1.4 [+0.6, +2.4] | +1.5 [+0.5, +2.5] | 0.75 |
| 2000 | 0.431 | 0.519 | +9.2 [7.7, 10.7] | +0.9 [+0.1, +1.8] | +1.1 [+0.4, +1.8] | 0.57 |

- **The per-question headroom is large and mostly real.** On the 87 pilot questions answered twice (runs A/B), an oracle chosen on run A and scored on run B keeps 57–81% of its in-sample headroom, e.g. +11.4 [6.5, 16.3] at 1000. The oracle reaches its maximum of 0.519 at only about 700 tokens/q.
- **The profile can exploit almost none of it.** Even in hindsight, the best per-cell choice over 27 full profile cells gains only +1.1 to +2.4 F1 when the profiler is free. That is about 10–15% of the per-question headroom. With its about 173 tokens/q charged, the gain falls to +0.4 to +1.4.
- **An honest profile policy loses.** Per-cell choices fitted on train with λ calibrated on train score −2.3 [−3.3, −1.3] vs fixed at about 1,030 tokens on dev, and −0.5 to −2.6 at the other budgets.
- Each single profile field used alone as the cell is ≤ 0 vs fixed at most budgets once the profiler is charged.
- A1 and A2 (F1 0.336 / 0.337 at about 1,010 tokens) sit far below even the train-fit cell policy.

**B. Profiler predictive power** (train n=1019, dev n=918). Only about 30% of questions are decisive (one strategy strictly best): 284 on dev. 43% are ties and 26% score 0 under every config.

- **Contingency of profile field vs winning config / strategy / budget** (decisive questions only): 24 tests over train and dev. One is p<0.05: train granularity × strategy, p=0.03, V=0.14. It does not replicate on dev (p=0.69). Every Cramér's V is ≤ 0.21.
- **Best config by mean F1:** leaf_only@2000 in every dev category. On train, collapsed@2000 is best for multi-fact, cross-section and whole-paper questions, by small margins.
- **RULES_V1 assumptions, tested as interactions** (difference in mean gain between categories, document bootstrap):
  - H1, "multi-fact / cross-section questions need more leaf context": **the opposite holds.** The gain from leaf@400→2000 is *smaller* for n_facts 3+ than for 1: −4.6 [−8.3, −1.0] on train, −5.0 [−9.1, −0.6] on dev. Cross-section: −6.7 [−11.4, −2.4] on train, −4.1 [−10.7, +2.4] on dev. Multi-fact questions are hard at every budget.
  - H2, "whole-paper questions benefit from summaries": the direction is right but not significant. summary_only − leaf: +3.7 (train), +5.9 [−1.1, +12.1] (dev) relative to specific_detail. Even for whole_paper, summaries remain worse than leaf in absolute terms (−5.4 on dev).
  - H3, "single-fact yes/no or short-span questions need little context": **the opposite holds.** They gain *more* from leaf@400→1000: +2.4 [−0.5, +5.4] on train, +5.6 [+2.5, +8.9] on dev. RULES_V1 sends them to the smallest budget, which explains much of A2's loss.
- **Profile accuracy:** answer_form agrees with the gold answer type on 55% (train) and 61% (dev) of questions. short_span and explanation are heavily confused. "unanswerable_likely" is never predicted, although about 9–10% of questions are gold-unanswerable.
- Caveat: an earlier version broke ties toward the cheapest config. That made budget look strongly associated (p≈0), but the effect came from tie handling, so the tie-aware version is reported.

**C. Unified accounting** (dev, per question; off-peak $; `tables.md`):

| System | F1 | Gen in | Gen out | Profiler in/out | Total | $/1k q | Index tokens/q | $/1k q incl. index |
|---|---|---|---|---|---|---|---|---|
| leaf_only@1000 | 0.400 | 1029 | 10 | 0/0 | 1039 | 0.160 | 0 | 0.160 |
| leaf_only@2000 | 0.432 | 2032 | 11 | 0/0 | 2043 | 0.311 | 0 | 0.311 |
| raptor_original | 0.387 | 1153 | 11 | 0/0 | 1164 | 0.180 | 2272 | 0.794 |
| collapsed@1000 | 0.376 | 1001 | 10 | 0/0 | 1011 | 0.156 | 2272 | 0.770 |
| A1 router | 0.336 | 830 | 10 | 140/33 | 1014 | 0.171 | 2272 | 0.786 |
| A2 router | 0.337 | 819 | 10 | 140/33 | 1002 | 0.170 | 2272 | 0.784 |
| sufficiency cascade* | 0.440 | 1930 | 24 | 0/0 | 1955 | 0.304 | 0 | 0.304 |

\* The cascade uses the new prompt, so compare it only with new-prompt baselines (NOTES "Results 2026-11-10 (2)").

- Profiler overhead is 17% of the routers' total tokens.
- Index summarization costs about 2,272 tokens per question, amortized over about 3.7 questions per paper. Every non-leaf strategy needs it, so per question it costs more than the generation itself.
- Leaf-only systems need no summaries.

## Pre-registration: retrieval-aware utility predictor (E2/E3), written 2026-11-10 18:40 UTC, before any E2 code ran

**Scope.**
- Offline replay only, $0. No LLM profiler, no new generation. Test split untouched.
- Overlap with earlier work: the 2026-11-09 offline_diagnostics step 2 already tried ridge/GBDT per-config F1 predictors with question, retrieval and embedding features (48 candidates; best −0.3 [−1.3, +0.7] vs the matched leaf mix on dev). This is a bounded follow-up. What is new:
  1. targets are gains relative to leaf@1000;
  2. marginal-context retrieval features (what the extra 1000→2000 / 400→1000 chunks add);
  3. selection on train only, a frozen model, and one dev look.

**Action space.** leaf_only@{400, 1000, 2000}. Base a0 = leaf@1000.

**Targets.** Δ400 = F1(leaf@400) − F1(leaf@1000) and Δ2000 = F1(leaf@2000) − F1(leaf@1000). They are regressed separately.

**Policy.** a*(q) = argmax_a [Δ̂(q,a) − λ·(C(q,a) − C(q,a0))]. C is known at retrieval time: the prompt input tokens of that context plus the config's mean train output tokens.

**Features** (no gold data):
- Q: SBERT query embedding plus surface question features.
- R: leaf similarity statistics (top-k, gaps, std, entropy); marginal-chunk similarity, novelty (1 − max cosine to the already-included chunks) and query-term coverage gain; positional diversity of the retrieved leaves; paper length.
- Q+R: both.

**Models.** Ridge (alpha chosen by an inner CV on the train fold) and a small GBDT with fixed hyperparameters. 3 feature sets × 2 models = 6 candidates. No other variants.

**Selection, on train only.**
- 5-fold paper-grouped cross-fitting.
- Metric: cross-fitted train F1 minus the train leaf mix at matched mean tokens, at target 1039 tokens/q (the cost of leaf@1000).
- The best candidate is refit on all of train. λ is calibrated on the cross-fitted train predictions so the mean cost is ≤ 1039. Both are frozen, with a hash.

**Dev, evaluated once.**
- Primary: the frozen policy vs the leaf mix matched to its realized dev mean tokens.
- **Gate:** Δ ≥ +2.0 F1 **and** the 95% paired document-bootstrap CI lower bound > 0.
- Reported, not gating:
  - the 1500-token operating point;
  - the λ-sweep frontier;
  - regret vs the leaf-only oracle and the all-9 oracle;
  - the share of leaf-only oracle headroom captured;
  - the USD cost;
  - comparison with A1/A2.

**Decision.** If the gate fails, stop router work and report the oracle opportunity and the predictability gap as the Section 5 result. If it passes, the frozen model is evaluated once on test. That needs paid test generation and separate approval.

## Results: E1 headroom decomposition + E2/E3 utility predictor (2026-11-10, $0)

**E1** (`scripts/headroom_decomposition.py` → `outputs/diagnostics/headroom_decomposition/20261110T183717Z/`). Dev, n=918.

Baseline = the best fixed policy with mixing, i.e. the upper hull of single configs (not a single fixed config).

| Action space | Oracle max F1 (saturates at tokens/q) | Headroom at 1000 | Headroom at 1500 | Pilot: headroom at 1000 scored on rerun B |
|---|---|---|---|---|
| leaf budgets only | 0.469 (730) | +7.1 | +5.5 | +9.2 [5.1, 13.3] |
| hierarchical only (6) | 0.479 (677) | +10.2 | +8.0 | +5.6 [1.5, 9.9] |
| strategy@1000 (3) | 0.455 (965) | +9.6 | — | +3.5 [−0.6, +8.0] |
| all 9 | 0.519 (707) | +12.1 | +10.5 | +11.4 [6.5, 16.3] |

- **Leaf-budget choice alone gives about 60% of the all-9 headroom at 1000.** Its headroom also replicates best on the rerun.
- Strategy choice at a fixed budget replicates worst (its CI includes 0).
- **Charging index costs.** At 2,272 tokens/q for every non-leaf choice, the hierarchical-only space cannot reach 2,043 tokens/q. The all-9 oracle still reaches 0.515 at 1000, because it uses the tree only where it pays. This per-question charge is a lower bound on index cost, since a paper's tree is paid in full once any question uses it.

**Oracle sanity checks.** These explain the flat 0.519 at 700–2000.
- With ties broken toward the cheapest config, the unconstrained oracle needs only 707 tokens/q on average. With ties broken toward the most expensive, it needs 1,858.
- Above about 707 the budget constraint is slack, so the oracle F1 is flat by construction.
- Tie structure: 26% of questions score 0 under every config; 50% reach their max under several configs (13% under all 9); only 24% have a unique best config.
- So much of the cheap headroom comes from "an equally good, cheaper config exists". That is a cost-saving opportunity, not an accuracy one.

**Stable gains (pilot A/B).** leaf@2000 beats leaf@1000 on 18.4% of questions in run A, and on 10.3% in both runs. Sign agreement across reruns, given a nonzero change in A, is only 0.48–0.77 per config.

**E2** (`scripts/utility_router.py --stage train` → frozen `outputs/diagnostics/utility_router/20261110T183924Z/`, sha256 5f6afb7b…). Train, 5-fold paper-grouped cross-fitting.
- Cross-fitted predictions are essentially uncorrelated with the realized gains, for every candidate: corr(D̂, D) between −0.10 and +0.05; AUC for D2000 > 0 between 0.45 and 0.52.
- Gains are zero for 79% of questions (D2000: 14% positive, 8% negative), so the regressions shrink almost everything to 0.
- A sanity check rules out a feature bug. The same features in a plain classifier reach CV AUC 0.54–0.60, consistent with the 2026-11-09 AUCs. They are not used, because they were not pre-registered.
- Train-CV vs the matched mix: every candidate is ≤ 0. The best is ridge|R at −0.13 [−0.39, +0.02], so that one is the selected candidate.

**E3: dev, the single pre-registered look** (`dev_report.json`, `utility_router_dev.png`):

| Operating point | Dev F1 | Tokens/q | Choice mix 400/1000/2000 | vs matched leaf mix | Regret vs leaf oracle | Leaf headroom captured |
|---|---|---|---|---|---|---|
| λ@1039 (primary) | 0.3998 | 1,042 | 0 / 909 / 9 | +0.01 [0.00, +0.04] | 6.9 [5.6, 8.3] | ≈ 0% |
| λ@1500 | 0.4130 | 1,402 | 0 / 578 / 340 | −0.05 [−1.1, +1.0] | 5.6 [4.5, 6.8] | 2.8% |

- **Gate (≥ +2.0 F1 and CI lower bound > 0): FAILED.** At the primary point the policy collapsed to leaf@1000.
- The λ sweep, which is for reporting only and was not used for selection, sits on the fixed-mix line, apart from one point near 575 tokens/q (+0.7) that was not pre-registered.
- **Decision (pre-registered): stop router work.** The Section 5 result is:
  1. the opportunity is large: per-question headroom +12 F1 at 1000 tokens/q, about 75% of it replicating on a rerun;
  2. it is not predictable at query time: LLM profiles, question features, embeddings and retrieval/marginal-context features all give ≈ 0 incremental gain over a matched fixed mixture;
  3. a large part of the headroom is "a cheaper config ties", plus generation-level variation that no pre-generation signal can see.


## 2026-11-11: P0b tree-navigation diagnostic (offline, stopped)

Design: `docs/OPTIMIZER_TREE_AWARE_MODIFICATION.md`; script: `scripts/check_tree_navigation.py`.
Report: `outputs/diagnostics/tree_navigation/20261111T151651Z/report.md`.

- Protocol frozen before dev: top-down only; beam on summary layers, all reached leaves eligible for budget packing. Candidate beams {1,2,4}; train selects separately at caps 1000/2000 by matched-cost highlighted-evidence recall delta. Both selected beam 4.
- Train 340 documents / 1,019 questions; dev 251 / 918. All 5,811 baseline node-sequence comparisons (@400/@1000/@2000 across both splits) reproduce stored traces exactly.
- Reuses compression_feasibility gold metrics verbatim: lexical highlighted-evidence coverage, not exact evidence-span matching. Dev recall denominator 820 questions; 98 without highlights excluded, all 918 included in token accounting.
- Actual-cost fixed randomized leaf-mix upper hull comparator, re-estimated within paired document bootstrap; 4000 replicates, Bonferroni-adjusted 97.5% CI for two budget points. Dev is an exploratory, previously analyzed split.

| Cap | Beam | Actual tokens | HL recall | Matched mix | Delta pp [97.5% CI] |
|---|---|---|---|---|---|
| 1000 (primary) | 4 | 956.7 | 0.8417 | 0.8682 | -2.65 [-3.89, -1.47] |
| 2000 (secondary) | 4 | 1801.2 | 0.9103 | 0.9491 | -3.88 [-5.09, -2.74] |

Primary gate FAILED; do not start P1/P2 integration. This is not a universal impossibility claim or answer-F1 measurement. Dev max-level histogram: L0=1, L1=185, L2=65; median multi-parent leaf fraction 0. Reused cached SBERT query embeddings; no model calls, API, downloads, or test reads. Five targeted tests and lint pass. No pipeline/vendor changes or commit. Frozen SHA256: 43ca9d059e9a6ac695c3b82c6858a5fee47793787ff0ae2aa65615641c52c0ac.


## 2026-11-11: NarrativeQA pilot, pre-registration (written before any NarrativeQA API call)

Purpose: does RAPTOR's hierarchy help on long stories (where the paper reports its larger gain, Table 1: ROUGE-L 30.87 vs 29.26 with UnifiedQA), and is the per-question headroom there replicable and larger than leaf-budget choice alone? Fidelity audit first: `docs/RAPTOR_FIDELITY.md` (algorithm code identical to upstream 7da1d48; 6 documented deviations; `collapsed@2000` = paper main setting).

- Data: HF `deepmind/narrativeqa` @ 2e643e73 (pinned sha256). Our `dev` = HF validation (115 stories, 3,461 q). Pilot = 20 stories, stratified 10 books / 10 movie scripts, seed 0 (`data/subsets/narrativeqa/2e643e73/dev_pilot.txt`): 594 questions, 1.74M cl100k tokens (median 50k, max 488k). The other 95 dev stories and the test split stay unused. 4 pilot stories fall back on a story marker (IMSDb page header kept).
- Configs: the same 9 as QASPER (leaf/summary/collapsed × 400/1000/2000), f100 chunks, upstream summary prompt, summary cap 200. Run A prompt `narrativeqa_default`; run B prompt `narrativeqa_paraphrase` on the same indexes (as the QASPER pilot).
- Metric: primary ROUGE-L F1 (max over the 2 references; rouge-score tokenization, no stemming; not the paper's modified script, so only config-vs-config comparisons are meaningful). Secondary: token F1.
- Estimated cost: indexes $0.8–1.4, answers ≈ $2.6 (A+B), total ≈ $3.5–4, off-peak.
- Analyses (paired document bootstrap; 20 stories, so this is a pilot: decisions use point estimates in BOTH runs plus CI direction, no confirmatory claims):
  1. Tree structure: layers per story, nodes per layer.
  2. RAPTOR claim: `collapsed@B` and `summary_only@B` vs the leaf-only fixed mixture at matched actual tokens (B = 1000, 2000), each run.
  3. Headroom decomposition as in E1: per-question oracle chosen on run A, scored on run B, at ≈1000 and ≈2000 tokens/q, for leaf budgets only vs all 9 configs.
- Go/no-go for a full-dev run (remaining 95 stories, ≈ $15, needs separate approval): proceed only if hierarchy shows value beyond leaf budgets, i.e. (a) collapsed or summary_only beats the matched leaf mix by ≥ +1.0 ROUGE-L in both runs, or (b) the all-9 replicated headroom (A-chosen, B-scored) exceeds the leaf-only replicated headroom by ≥ +2.0 ROUGE-L at ≈1000 tokens/q. Otherwise stop and report.


## 2026-11-11: Adaptive-k offline test, pre-registration (written before running)

Rule: official Adaptive-k (Taguchi et al. 2025; megagonlabs/adaptive-k-retrieval `largest_gap`, `--ignore_extreme_tail 0.1 --retrieve_more 5`): keep top (largest-drop index + 1 + 5) leaf chunks. Untuned, so dev is usable despite earlier analyses. Script `scripts/adaptive_k_offline.py`; RAPTOR SBERT embeddings (paper used bge-large), cached query embeddings, stored QASPER labels; no API.
- Kept chunk count → budget: primary "ceil" (smallest of leaf@400/1000/2000 containing all kept chunks; 2000 if none, counted as capped); secondary "nearest" (by tokens).
- Primary: dev, ceil, F1 vs leaf fixed mixture at the same mean generation tokens, paired document bootstrap 95%. Train reported as replication. Success = CI lower bound > 0. Expected (from E2): ≈ 0.

**Adaptive-k result** (`outputs/diagnostics/adaptive_k/20261111T163535Z/`). Ranking reproduced exactly (0 mismatches vs stored leaf@2000 on 918 + 1,019 questions).

| Split | Mapping | F1 | Tokens/q | Matched leaf mix | Delta [95% CI] |
|---|---|---|---|---|---|
| dev (primary) | ceil | 0.4061 | 1,237 | 0.4096 | −0.35 [−1.15, +0.44] |
| dev | nearest | 0.3680 | 767 | 0.3711 | −0.31 [−1.74, +1.04] |
| train | ceil | 0.3279 | 1,204 | 0.3338 | −0.59 [−1.27, +0.09] |
| train | nearest | 0.2951 | 744 | 0.3063 | −1.12 [−2.21, −0.04] |

- Not supported: no gain over the matched fixed mixture on either split; "nearest" on train is significantly worse.
- The rule mostly degenerates to the buffer: on dev 39% of questions keep exactly 6 chunks (largest drop right after the first chunk, + buffer 5) and the median is 7 (about 590 tokens, 15% of a paper's leaves); p10–p90 6–23. So per-question variation is small and mostly noise.
- Limitation: answers exist only at 400/1000/2000, so the exact k-chunk context (≈ 590 tokens median) is approximated by a budget. This favours neither direction by design (ceil over-includes, nearest rounds), and both mappings agree.
- Consistent with E2 (similarity-gap features had no predictive value) and with the paper's own factoid tables, where Adaptive-k trails the best fixed budget.

**NarrativeQA pilot result** (runs A `outputs/narrativeqa_pilot/20261111T153658Z`, B `outputs/narrativeqa_pilot/20261111T175627Z`; analysis `outputs/diagnostics/narrativeqa_pilot/20261111T180900Z/report.json`). 594 q × 9 configs × 2 runs, 0 errors. Spend ≈ $2.6 (summaries 2.34M in / 0.60M out; answers A 6.04M in / 47k out, B 6.08M in / 45k out; off-peak 15:37–18:08 UTC).
- Trees: summary layers per story 2/3/4 = 6/13/1 (deeper than QASPER, as in the paper's 0–3/0–4).
- Fixed ROUGE-L (A / B): leaf@1000 0.311 / 0.290, leaf@2000 0.346 / 0.333; collapsed@1000 0.300 / 0.299, collapsed@2000 0.325 / 0.324; summary@2000 0.304 / 0.309.
- (a) RAPTOR claim not reproduced with this reader: vs matched leaf mix, collapsed@2000 −2.07 [−3.91, −0.07] (A) / −1.07 [−3.33, +1.19] (B); collapsed@1000 −0.99 (A) / +0.82 (B), CIs span 0; summary_only@2000 −4.10 [−7.15, −1.06] (A). No config ≥ +1.0 in both runs. Token F1 agrees.
- (b) Replicated headroom at ≈1000 tokens/q (A-chosen, B-scored): leaf budgets +5.26 [3.26, 7.29], all 9 +12.11 [9.17, 15.05]; difference +6.85 ≥ 2.0 → **criterion (b) met, pre-registered decision = go** (full dev still needs separate approval).
- Exploratory (not pre-registered; scratchpad `nqa_armcheck.py`), replicated headroom at 1000 by option set: leaf3 +5.3, coll3 +2.9, summ3 +4.0, leaf+coll6 +8.6, leaf+summ6 +11.8, all9 +12.1. The gain comes from offering summary-only contexts as an alternative to leaves (complementarity), not from collapsed. Caveat: A and B share the same retrieved context, so per-cell A–B correlation is 0.85; "replicated" removes prompt/decoding noise but not context-specific luck, and headroom is an oracle upper bound, not something a router has shown it can predict (cf. QASPER, where E2 found no predictive features).

## 2026-11-12: Failure casebook (offline, $0)

`scripts/failure_casebook.py` → `outputs/diagnostics/failure_casebook/20261113T014222Z/` (`casebook.md` readable sample, 12 per category, seed fixed; `cases.jsonl` all 921 failing questions with every system's answer, chosen config, retrieved node texts + layers). Fail = score < 0.3. Routers A1/A2 replayed from frozen checkpoints; replay reproduces the stored dev F1.
- QASPER dev: router A2 fails on 96 questions that raptor_original answers; 80 of them are routed to collapsed@400 and 69 answer "Unanswerable" — the router's main failure is under-budgeting, not strategy choice. 355 questions fail for every one of the 9 configs.
- NarrativeQA pilot: ~60% of questions score < 0.3 for every system; in about 70% of those failures a reference answer is ≥ 50% recalled in the retrieved context and the model rarely abstains (≤ 4%), so many are reader/metric (paraphrase vs ROUGE-L) failures rather than retrieval failures. Heuristic tags only.

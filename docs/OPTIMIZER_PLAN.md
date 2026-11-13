# Optimizer implementation plan: LLM profiler (Plan A) with a trained-router fallback (Plan B)

Status (2026-11-09): **Plan A implemented and tested offline.** WP0–WP6 (A1, A2), WP8, WP9 and WP10 are done. WP7 (direct selection) is not started.

**Frozen settings (user decisions, 2026-11-09):**
- Profiler prompt `profile_v3` (`prompt_hash e22d6935b289c85d`, about 140 API input tokens).
- No confidence fallback (`confidence_threshold: null`); invalid profiles fall back.
- A1 is primary and A2 is the ablation.

**Limitations to report:**
- On an uncached rerun at t=0, 7% of v3 profiles (16% for v1) changed. Stored profiles are reused for reproducibility.
- Profile fields have no ground truth.
- H1 is evaluated by replay over T12 labels. The design and its literature are in `../Optimizer_LLM_Plan.md`. `Adaptive_RAPTOR_Proposal.md` still describes the embedding classifier.

**Team decision (2026-11-09): Plan B is conditional.** The embedding classifier (B) and the hybrid (H, which uses B's features) are **not built** unless the LLM router shows **no improvement on dev**. That means the `router_train` dev replay against the matched best fixed config gives a CI that is not above 0, or a mean ΔF1 ≤ 0. This replaces the A-vs-B switch rule in WP9. `FeatureExtractor` and `AcceptabilityModel` stay stubs.

**Findings that change the design** (details and numbers in `NOTES.md`):
- DeepSeek logprobs are post-temperature: at t=0 they're all 0.0, so the METIS-style confidence fallback is inactive (`use_logprobs: false`). Only invalid profiles fall back. Options are listed in `NOTES.md`.
- `profile_v1` costs 332 input tokens per query (the plan assumed 200–300).
- Profiles changed for 16% of questions on a rerun at t=0. Profile each question once and reuse the stored file.

| WP | Status | Where |
|---|---|---|
| WP0 | Done | `models/chat.py` (thinking disabled); `NOTES.md` |
| WP1 | Done (profiler only; the generator keeps its own client) | `models/chat.py::ChatClient` |
| WP2 | Done | `optimizer/profiler.py` |
| WP3 | Done; pilot profiled | `query_profiling`, `configs/experiments/profile_queries.yaml` |
| WP4 | Done (T06) | `profiling.py` |
| WP5 | Done | `optimizer/__init__.py`: `CostModel`, `PlanSelector`, `calibrate_theta`, `theta_sweep`, checkpoints |
| WP6 | A1 and A2 done. B and H deferred (conditional) | `ProfileTableModel`, `RuleMapModel` (`RULES_V1`) |
| WP7 | Not started (optional) | |
| WP8 | Done | `policy.py`, `pipeline.py`, router fields in `ExecutionTrace` |
| WP9 | Done. Needs real train/dev labels to run | `router_train`, `configs/experiments/router_train.yaml` |
| WP10 | Done (replay over T12 labels; mock smoke test) | `h1_optimizer`, `configs/experiments/h1.yaml` |
| WP11 | Not started (optional) | |

## 0. Invariants (these hold for every work package)

1. **One selection core.** Every router variant only produces `P(acceptable | q, c)` for the 9 catalog configs. A single shared `PlanSelector` then picks the cheapest config with P ≥ θ, else the fallback. H1's θ calibration, budget matching and freeze apply unchanged to all variants.
2. **The router pays for itself.** Cost = context tokens + profiler input tokens. θ calibration, the dev frontiers and H1's primary analysis all use this total (`optimizer.cost_includes_profiler: true`, the default).
3. **No test leakage.**
   - Profiles are computed from the question and title only, never from answers or evidence.
   - Every fit and calibrate call goes through `guard_split`.
   - Test profiles are produced with the frozen prompt hash recorded in the checkpoint.
4. **No paid calls outside an explicit experiment command.** Tests and `smoke_mock.yaml` use `MockProfiler`. Real profiling happens only inside `run.py` experiments that pass preflight.
5. **Everything is cached and logged.** Each profile is cached by (provider, model, settings, prompt hash, question id). Every routed trace records the profile, the confidence, profiler tokens and latency, and the fallback reason.

## 1. Data flow

```text
                 ┌──────────── offline (train / dev) ──────────────────────────────┐
 questions ──► [query_profiling experiment] ──► profiles.jsonl ─┐
 (train,dev,     LLMProfiler (cached)                            │
  test)                                                          ▼
 catalog runs ─► traces.jsonl ─► LabelStore ─► acceptability_labels(ε) ─► AcceptabilityModel.fit
 (pilot_catalog,                                                 │     A1 ProfileTable | A2 RuleMap
  T08/T11)                                                       │     B  LogReg(emb+lex) | H Hybrid
                                                                 ▼
                              dev probs ─► CostModel ─► calibrate_theta / theta_sweep
                                                                 │
                                         switch rule (A vs B) ──►│──► RouterCheckpoint (frozen, T12)
                 └───────────────────────────────────────────────┼─────────────────────────────────┘
                                                                 ▼
 online (H1 test):  question+title ─► Router.route ─► profile? ─► P(c) ─► PlanSelector ─► 1 config
                                                     ─► retrieve ─► generate ─► trace (+router cost)
```

Profiling is done **once per question** as its own experiment, and its output is reused by training, dev calibration and the test run. As a result:

- the H1 test run never makes a cold profiler call it hasn't logged;
- the profiles stay identical across runs A and B and across indexes (PIC router, S1).

## 2. Work packages

Each WP lists files, the interface, tests (all offline unless marked), and "done when". **Bold** WPs are on the critical path.

### **WP0: Unblock DeepSeek (prerequisite; backlog T03)**
- `generation/__init__.py` and `models/summarization.py`:
  - Send the documented DeepSeek parameter that disables thinking. Today `thinking` is only recorded (see the warning in `OpenAICompatibleGenerator.__init__`).
  - Raise on an empty `content`.
  - Record `completion_tokens` and reasoning tokens separately when the API reports them.
- **Probe once (paid, about 5 calls, inside an integration test):**
  - Does `deepseek-flash` in non-thinking mode support `response_format={"type": "json_object"}` and `logprobs` / `top_logprobs`?
  - Record the answer in `NOTES.md`. WP2's confidence method depends on it.
- Done when: the summarizer and generator return non-empty text and output tokens ≈ visible tokens.

### **WP1: Shared chat client** (`models/chat.py`, new)
- Move the client construction, `JsonCache`, tenacity retry and usage parsing out of `OpenAICompatibleGenerator` into:
  ```python
  @dataclass
  class ChatResult:
      text: str
      input_tokens: int
      output_tokens: int
      latency_ms: float
      model_version: str
      logprobs: list[tuple[str, float]] | None
      cache_hit: bool

  class ChatClient:
      def __init__(self, provider, model, *, temperature, max_tokens, thinking=False,
                   api_key_env=None, base_url_env=None, timeout_s=60, cache_dir=None,
                   use_cache=True, max_retries=3): ...
      def complete(self, messages, *, json_mode=False, logprobs=False,
                   cache_namespace: str) -> ChatResult: ...
  ```
- `OpenAICompatibleGenerator` becomes a thin wrapper around it, so generator behavior and cache keys are unchanged. Keep the old key format, or bump the cache namespace and note it.
- Tests: the existing generation tests pass unchanged. A fake OpenAI client (monkeypatched) checks the retry behavior, empty-content errors and cache hits.

### **WP2: Profile schema and profilers** (`optimizer/profiler.py`, new)
```python
class QueryProfile(StrictModel):            # pydantic, extra="forbid"
    answer_form: Literal["yes_no", "short_span", "explanation", "unanswerable_likely"]
    granularity: Literal["specific_detail", "section_level", "whole_paper"]
    n_facts: conint(ge=1, le=5)
    cross_section: bool

@dataclass
class ProfileResult:
    question_id: str
    profile: QueryProfile | None            # None => invalid / failed
    confidence: float | None                # None => not available
    input_tokens: int; output_tokens: int; latency_ms: float
    prompt_hash: str; model_version: str; cache_hit: bool
    error: str | None = None

class QueryProfiler(ABC):
    prompt_hash: str
    def profile(self, question_id: str, question: str, title: str) -> ProfileResult: ...

class LLMProfiler(QueryProfiler):            # ChatClient, json_mode, temperature 0
class MockProfiler(QueryProfiler):           # deterministic from question text (wh-word, length)
def build_profiler(cfg: ProfilerSection) -> QueryProfiler
```
- **Prompt** (`optimizer/prompts/profile_v1.txt`, whose hash is part of the cache key and the checkpoint). The draft below defines each field in one line and gives two few-shot examples taken **from train only**:
  ```text
  You classify a question about a scientific paper. Do not answer it.
  Paper title: {title}
  Question: {question}
  Return JSON with exactly these keys:
  "answer_form": "yes_no" | "short_span" | "explanation" | "unanswerable_likely"
  "granularity": "specific_detail" (one number/name/setting) | "section_level" (one method or
                 result) | "whole_paper" (contributions, overall findings)
  "n_facts": integer 1-5, how many separate facts the answer needs
  "cross_section": true if the facts are likely in different sections, else false
  ```
- **Confidence:**
  - *If logprobs are available (WP0):* confidence = product, over the 4 fields, of the probability of the first token of the chosen value. Locate those tokens with an incremental JSON scan over the returned tokens.
  - *Otherwise:* `confidence=None`, and the fallback rule uses only invalid-JSON failures.
  - Self-consistency (3 samples) is a **dev-only diagnostic** script, never part of routing.
- **Failure handling:** invalid JSON or a schema violation → one retry with the same prompt plus "Return only valid JSON". If that also fails, return `profile=None` with the error.
- Tests: schema rejects extra or out-of-range fields; MockProfiler is deterministic; the cache key changes when the prompt or model changes; confidence extraction is checked on a hand-made logprob sequence; invalid JSON → retry → `None`.

### **WP3: `query_profiling` experiment** (`experiments/__init__.py`, `config/schema.py`, `configs/experiments/profile_queries.yaml`)
- New `ExperimentType.QUERY_PROFILING`.
  - Loads the dataset split and subset, joins each question to its document title, and profiles questions concurrently (`runtime.concurrency`).
  - Writes `<run_dir>/profiles.jsonl`.
  - `status.json` metrics: counts, invalid rate, confidence histogram, token totals, spend estimate.
- `ProfileStore` (JSONL, keyed by question_id + prompt_hash) is loaded by training and by the router.
- Preflight requires credentials only when `optimizer.profiler.provider != "mock"`.
- Tests: running on `qasper_tiny` with the mock profiler gives one profile per question, identical across reruns, and the second run is all cache hits.

### **WP4: Labels** (`profiling.py: acceptability_labels`)
- Group by (question_id, index_variant, seed, run).
  - `best = max F1`. If `best == 0`, the query is all-fail and maps to `None`, which keeps it out of training.
  - Otherwise each config is acceptable if `F1 ≥ best − ε`.
- Also return per-answer-type label rates, so the proposal §5.2 label-distribution report can be written.
- Tests: hand-built records cover ties, ε edges, all-fail queries and missing configs. A question that lacks a config is excluded and **counted**, never silently filled.

### **WP5: Shared selection core** (`optimizer/__init__.py`)
- `CostModel.fit(dev_records, profiler_overhead_tokens)`: `cost(c)` = mean dev `tokens_used(c)`. `route_overhead` is the mean profiler input tokens measured in WP3 (0 for Plan B). `total(c) = cost(c) + route_overhead`.
- `PlanSelector.select`: the cheapest config by `cost(c)` with `p[c] ≥ θ`, ties broken by catalog order. If none qualifies, or the profile is invalid or low-confidence, return the fallback with `used_fallback=True` and a reason.
- `calibrate_theta`: grid over θ ∈ {0.05, …, 0.95}. Choose the θ whose dev mean **total** cost is closest to `target_mean_tokens` **without exceeding it**. `theta_sweep` returns (θ, mean total tokens, mean F1) using the dev F1 lookup table, so no new calls are needed.
- `save_checkpoint` / `load_checkpoint`: a directory holding `checkpoint.json` (config hash, catalog, θ, fallback, model_type, prompt hash, profiler model, feature spec, label ε, train/dev doc-id hashes) and `model.joblib`.
- Tests: the selector is cheapest-first; θ=1 → fallback; a monotonic toy sweep is checked; calibration never exceeds the target; a checkpoint round trip gives identical decisions; `guard_split("test", "calibrate")` raises.

### **WP6: Acceptability models** (`optimizer/__init__.py`, one `AcceptabilityModel` protocol: `fit(X, labels, groups)` / `predict_proba(X)`)

| `model_type` | Plan | Input | Model |
|---|---|---|---|
| `llm_profile_table` | **A1 (primary)** | profile | Cell acceptance rate with Beta(α·p̄_c, α·(1−p̄_c)) smoothing. Back-off when a cell has fewer than `min_cell` = 10 examples: full cell → (form, granularity) → form → global. α is chosen by GroupKFold on documents. |
| `llm_profile_rules` | A2 (ablation) | profile | A hand-written mapping committed in `optimizer/rules_v1.yaml` **before** dev labels exist: the profile gives allowed strategies and a minimum budget, and P = 1 for allowed configs, else 0. Analogue of METIS Alg. 1. |
| `logistic_regression` | **B** | PCA32(emb) + lexical | The existing proposal §5.3 design: per-config L2 logistic regression, `class_weight="balanced"`, C chosen by GroupKFold. |
| `hybrid_logreg` | H | B's features + one-hot profile | Same as B. |
| `ordinal_min_budget` | B′ | as B | Kept from the schema; built only if T08 shows monotone quality in budget. |

- `FeatureExtractor` (B and H): PCA is fitted on train embeddings only. Lexical features: has-digit, has-year, capitalized-token count, wh-word one-hot, yes/no-question flag, token length. Profile one-hots are added when `"llm_profile"` is in `features`.
- Tests: the table recovers known rates on synthetic data; back-off triggers at `min_cell`; LR is separable on synthetic data; PCA is never fitted on dev (spy); GroupKFold never splits a document.

### WP7: Direct LLM selection (A3, reference only) (`optimizer/profiler.py`)
- `DirectSelector.select(question, title) -> ConfigRef`: the prompt lists the 9 configs in plain words.
- It is not θ-calibratable, so it appears as one point on the dev and test frontier. Built **only if** WP0–WP9 are on schedule.

### **WP8: Online routing and pipeline plumbing**
- `QueryContext` gains `title: str` and `profile: ProfileResult | None`.
- `ConfigPolicy.configs_for(question, query_embedding, document=None)`. `QAPipeline._answer_doc` passes `doc`, which it already has.
  - This is backward-compatible: `CatalogPolicy` ignores `document`.
- `Router.route(ctx)`:
  1. Look up the profile in `ProfileStore`. On a miss, call the profiler. Misses are logged and counted, and in a test run a nonzero miss count is reported.
  2. Features → `predict_proba`.
  3. `PlanSelector`.
- `ExecutionTrace` new fields:
  - `router_model`, `profile` (dict), `profile_confidence`;
  - `router_input_tokens`, `router_output_tokens`, `router_ms`;
  - `route_fallback_reason`, `route_probabilities`.
  - `summarize()` adds `total_tokens = context_tokens + router_input_tokens` and the fallback rate.
- Tests:
  - `smoke_mock` with `optimizer.enabled: true` and the mock profiler runs end to end, with exactly 1 config per question.
  - The trace fields are populated, and the fallback reason is set when the mock returns an invalid profile.
  - `CatalogPolicy` traces are byte-identical to before (regression).

### **WP9: Training, dev selection, freeze** (`train_router`; `configs/experiments/router_train.yaml`)
- `train_router(config, label_store_path, split="train")`:
  - Inputs: labels + profiles (+ embeddings).
  - Fit the model selected by `model_type`, then the cost model and θ on dev.
  - Output: the checkpoint.
- `scripts/select_router.py` (offline; uses dev label tables only):
  - Train A1, A2, B and H.
  - Write `results/dev/router_frontiers.{csv,png}` (total-cost basis) plus each variant's fallback rate.
  - **Apply the pre-registered switch rule:**
    - Choose A1 or H if it beats B by ≥ 1.0 F1 at the target total budget **and** its fallback rate is ≤ 20%.
    - Otherwise choose B.
  - Write the decision and the numbers to `NOTES.md`, then freeze the chosen checkpoint (T12).
- Done when:
  - one checkpoint is frozen with its config hash recorded;
  - routing takes < 1 ms excluding the embedding and the profile lookup;
  - a profile cache miss is measured separately.

### WP10: H1 runner hook (backlog T07 / T13; only the router-specific part)
- The `h1_optimizer` experiment loads the frozen checkpoint and runs `QAPipeline` with `RouterPolicy` on test.
- Matching against the best fixed config uses **total tokens** (primary) and context tokens (secondary), through `evaluation/research.py`.
- The non-chosen router is evaluated as a secondary run. A2, A3 and the B-vs-A comparison go in the results.

### WP11: Bandit replay (Plan C, optional, no API calls)
- `scripts/bandit_replay.py`: LinUCB and Thompson sampling over the 9 arms, with reward F1 − λ·tokens/1000. It runs over train labels in 5 random orders, using B's features.
- Report the cumulative regret against the full-information router B. This needs no new labels.

## 3. Schema changes (`config/schema.py`)

```yaml
optimizer:
  enabled: true
  model_type: llm_profile_table   # | llm_profile_rules | logistic_regression | hybrid_logreg | ordinal_min_budget
  features: [llm_profile]          # B: [query_embedding, lexical]; H: all three
  pca_dim: null                    # required only with query_embedding
  profiler:                        # new ProfilerSection; required iff llm_profile in features
    provider: deepseek             # or mock
    model: deepseek-flash
    prompt: profile_v1
    inputs: [question, title]      # [question, title, abstract] = ablation
    max_output_tokens: 60
    json_mode: true
    use_logprobs: true             # false if WP0 probe says unsupported
    confidence_threshold: 0.9      # tuned on dev; null disables
    profiles_path: null            # ProfileStore from the query_profiling run
  table: {alpha: auto, min_cell: 10}
  cost_includes_profiler: true
  label_epsilon: 0.05
  target_mean_tokens: 1000
  fallback: {strategy: collapsed, budget: 2000}
  checkpoint: null
```

New validation rules (`config/validation.py`):
- An `llm_profile*` or `hybrid` model needs a `profiler`.
- `pca_dim` requires `query_embedding`. This rule exists already.
- `h1_optimizer` on test requires `checkpoint` and `profiles_path`, and the checkpoint's prompt hash must equal the profiler's prompt hash.
- `required_credentials` includes the profiler provider.

## 4. Order and parallelism (three people)

```text
WP0 ──► WP1 ──► WP2 ──► WP3 ──► (profile pilot+train+dev, ~$0.3) ─┐
WP4 (labels)  ─────────────────────────────────────────────────────┤
WP5 (selection core) ──► WP6 (A1, A2, B, H) ───────────────────────┼─► WP9 select+freeze ─► WP10 H1
WP8 (plumbing; needs WP2 interface + WP5) ─────────────────────────┘
WP7, WP11: optional, after WP9
```

| Track | Owner (suggested) | WPs |
|---|---|---|
| LLM side | person 1 | WP0, WP1, WP2, WP3, WP7 |
| Learning side | person 2 | WP4, WP5, WP6 (B, H), WP11 |
| Integration | person 3 | WP6 (A1, A2), WP8, WP9 script, WP10 |

Mapping onto the proposal's 4-week schedule:
- **Week 2:** WP0–WP6 and WP8, profiling of pilot, train and dev questions, and the train labels.
- **Start of week 3:** WP9 selection and freeze.
- **Week 3:** test profiling (with the frozen prompt) and WP10.

WP4–WP6 can start right away on the mock profiler and fixture labels, without waiting for WP0.

## 5. API calls and spend (estimate; the pilot measures the real values)

| Item | Calls | Tokens | Cost at $0.15 / $0.60 per 1M |
|---|---|---|---|
| WP0 probe | ~5 | negligible | ≈ $0 |
| Profile pilot (200 q) | 200 | ~300 in / ~40 out each | < $0.02 |
| Profile train (~1k) + dev (1,005) + test (1,451) | ~3.5k | ~1.05M in / ~0.14M out | ≈ $0.25 |
| Retries + self-consistency diagnostic on dev (3×) | ~3.5k | same order | ≈ $0.25 |
| A3 direct selection on dev + test (optional) | ~2.5k | ~500 in each | ≈ $0.25 |
| **Total profiler-side** | | | **< $1** |

Router training and calibration themselves make no API calls. They reuse the catalog labels already budgeted in proposal §6.4.

## 6. Test checklist (offline, default `pytest`)

| Area | Tests |
|---|---|
| Profiler | Schema strictness; mock determinism; cache-key sensitivity; logprob confidence; invalid JSON → retry → None |
| Experiment | `query_profiling` on `qasper_tiny` (mock): counts, rerun = all cache hits, `status.json` metrics |
| Labels | ε edges, ties, all-fail → None, missing config counted |
| Models | Table back-off; Beta smoothing; LR on synthetic data; PCA fitted on train only; GroupKFold by document |
| Selection | Cheapest-first; θ edges; calibration ≤ target on the total-cost basis; checkpoint round trip |
| Leakage | `guard_split` on fit / calibrate / select with test; checkpoint-vs-profiler prompt-hash mismatch rejected |
| Pipeline | Router smoke run: 1 config per question, router trace fields present, CatalogPolicy regression unchanged |
| Integration (opt-in, paid) | One real profile call; WP0 capability probe |

## 7. Risks and decision points

| Risk | Detection | Action |
|---|---|---|
| Profiler overhead cancels the gains (the main risk to H1) | Dev frontier on the total-cost basis (WP9) | Switch rule → B. Report A as a secondary result. |
| No logprobs from `deepseek-flash` | WP0 probe | Set `use_logprobs: false`. The fallback then covers invalid JSON only. Report the dev self-consistency rate as a stability diagnostic. |
| Profile cells too coarse (most queries share 2–3 cells) | Cell histogram after WP3 | Use H (the hybrid) as the Plan A candidate instead of A1. |
| The `deepseek-flash` alias changes mid-project | `model_version` in profiles and traces | Profile all splits within the same week. Record the version in the checkpoint. |
| Profiler and generator are the same model (shared biases) | — | State it as a threat to validity. Optional robustness check: profile dev with a second model and compare only the agreement rate. |
| Test profiles missing at H1 time | Router miss counter | H1 preflight requires `profiles_path` to cover all test question ids. |

## 8. Backlog changes (proposed, not applied)

- **T11 split:**
  - **T11a** LLM profiler (WP1–WP3);
  - **T11b** selection core and labels (WP4, WP5);
  - **T11c** acceptability models and selection (WP6, WP9);
  - **T11d** pipeline plumbing (WP8).
- **T03** gains the WP0 probe.
- **T12** freezes the profiler's prompt hash and model version along with θ.
- **T13** reports both routers, plus the total-cost and context-cost matching.
- **New T18** (optional): direct selection (WP7) and bandit replay (WP11).

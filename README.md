# Adaptive-RAPTOR

Dynamic chunking and query-aware context optimization on top of
[RAPTOR](https://github.com/parthsarthi03/raptor) (CMSC 724 course project).

- **H1 (query time):** a lightweight router that picks a (retrieval strategy, context budget)
  per query beats the best fixed configuration at matched average context tokens.
- **H2 (index time):** PIC-capped leaf chunking beats the strongest non-PIC chunking control
  at a matched budget.

Design: [`../Adaptive_RAPTOR_Proposal.md`](../Adaptive_RAPTOR_Proposal.md) and
[`../Implementation_Evaluation_Draft.md`](../Implementation_Evaluation_Draft.md).
Work tracking: [`backlog/`](backlog/README.md).

**Status:** the infrastructure is complete. The original RAPTOR baseline runs end to end from
YAML, with dataset download, index caching, traces and results. The research algorithms
(PIC, router, statistics) are clearly marked placeholders; see the
[checklist](#implementation-checklist).

---

## 1. Architecture

```text
QASPER (download.py) ─► load_dataset ─► Chunker ─► IndexBuilder ─► ConfigPolicy ─► RetrievalExecutor ─► Generator ─► answer F1 ─► traces
                         (subsets)     (fixed │     (upstream RAPTOR   (catalog │      (upstream           (DeepSeek │                  summary, labels,
                                        PIC*)       tree + cache)      router*)        TreeRetriever)      mock)                         predictions
* = placeholder until implemented
```

```text
src/raptor/                 upstream RAPTOR, verbatim (UPSTREAM.md); not linted, never edited
src/adaptive_raptor/
├── run.py                  CLI: load YAML -> validate -> preflight -> dispatch
├── config/                 schema.py (typed), loader.py (YAML + --set), validation.py (per type)
├── experiments/            package API (base/registry/runners), runner.py, outputs.py
├── pipeline.py             the shared QA loop (all runners use it)
├── policy.py               ConfigPolicy: which configs to run per question (CatalogPolicy)
├── data/                   records.py (shared schema), qasper.py, download.py, subsets.py,
│                           fixtures/qasper_tiny.json (offline smoke data)
├── chunking/               one API: contracts, upstream fixed baseline, adaptive stubs*
├── models/                 credentials.py, embedding.py (SBERT/OpenAI/mock), summarization.py
├── indexing/               builder.py (RAPTOR tree from any chunks, seeded), store.py (cache)
├── retrieval.py            upstream TreeRetriever strategy/budget adapters
├── generation/             one API: prompts, mock, DeepSeek/OpenAI-compatible adapter
├── evaluation/             metrics.py, traces.py, research.py* (frontier/matching/bootstrap)
├── profiling.py            LabelRecord, LabelStore, acceptability-label stub*
├── optimizer/              one API: typed optimizer contracts* + RouterPolicy adapter
├── stub.py                 the placeholder marker (see below)
└── paths.py, cache.py
```

The concise file-by-file map, implementation contracts, order, and commands are in
[`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).

**Design rules:**

- **Config, orchestration and research code are separate.** `config/` has no heavy imports, so validation never loads torch or RAPTOR. `experiments/` owns run directories, metadata and status. Research modules receive typed config sections.
- **Upstream RAPTOR is reused, not rewritten.** These come from `src/raptor` unchanged:
  - leaf embedding and clustering (UMAP + GMM);
  - recursive summarization;
  - `Tree`;
  - `TreeRetriever`.

  `IndexBuilder.build_from_chunks` is upstream `build_from_text` without its internal `split_text` call, so any chunker can supply the leaves. Each strategy runs upstream collapsed retrieval on a view of the tree restricted to that strategy's levels.
- **Placeholders fail fast.** Unimplemented research classes inherit from `adaptive_raptor.stub.Stub`. The pipeline rejects them **before** loading data, building indexes or calling APIs, and the run is reported as `not_implemented` (exit code 3), never as success.

## 2. Prerequisites

- **OS:** macOS (arm64 is the primary target) or Linux. On Linux, the CPU-only build of torch is used.
- **Tools:** `curl`, `tar`, `git`.
- **Disk:** about 1 GB for `.venv`. On top of that:
  - about 165 MB for QASPER (raw and processed);
  - about 420 MB for the SBERT model, downloaded on the first real index build.
- **API key:** `DEEPSEEK_API_KEY`, only for real runs. Setup and tests never call paid APIs.

## 3. One-command setup

```bash
./scripts/setup.sh                 # add --with-tests to also run the test suite
```

The script is idempotent. It runs these steps:

1. Checks the OS and the required tools.
2. Finds `uv`, or installs it (Homebrew on macOS, otherwise the official installer; `--no-install-uv` disables this).
3. Installs Python 3.11 and `uv sync --frozen` from `uv.lock`.
4. Creates `data/`, `indexes/`, `outputs/`, `.cache/`, and `.env` from `.env.example`, without overwriting an existing `.env`.
5. Runs `scripts/check_environment.py`: imports, an upstream-copy check, tokenizer, every example config, directories, credentials (reports only which are set, never the values), dataset and SBERT cache.
6. Runs the **offline smoke run**: the whole baseline pipeline on a bundled fixture with mock models.

Run the environment check alone with `uv run python scripts/check_environment.py`.

## 4. Datasets

```bash
./scripts/download_datasets.sh --dataset qasper            # download + preprocess (~4 s)
./scripts/download_datasets.sh --dataset qasper --verify   # offline integrity check
./scripts/download_datasets.sh --dataset qasper --make-subset pilot --split dev --n-docs 30 --seed 0
```

The official QASPER v0.3 release comes from AllenAI's S3, with pinned SHA-256 checksums. Splits are kept as released: 888 / 281 / 416 papers and 2,593 / 1,005 / 1,451 questions (train / dev / test). Re-running the script never re-downloads or re-processes verified files.

See [`docs/DATA.md`](docs/DATA.md) for the layout, schema, manifest, subsets and the data-root setting. In short, the data root is `--data-dir`, else `$ADAPTIVE_RAPTOR_DATA_DIR`, else `./data`.

## 5. YAML configuration

Every experiment is a YAML file in `configs/experiments/`:

| File | Type | Runs today? |
|---|---|---|
| `smoke_mock.yaml` | `raptor_baseline` | **yes**: offline, mock models, bundled fixture |
| `baseline.yaml` | `raptor_baseline` | **yes**: real SBERT + DeepSeek, 2 dev papers (needs key) |
| `pilot_catalog.yaml` | `raptor_baseline` | **yes**: full 3×3 catalog on the pilot subset (E0, labels) |
| `h1.yaml` | `h1_optimizer` | validates; runner `not_implemented` |
| `h2.yaml` | `h2_chunking` | validates; runner `not_implemented` |
| `ablation.yaml` | `strategy_budget_ablation` | validates; runner `not_implemented` |
| `oracle_interaction.yaml` | `oracle_interaction` | validates; runner `not_implemented` |

```bash
uv run python -m adaptive_raptor.run --config configs/experiments/h1.yaml --validate-only
uv run python -m adaptive_raptor.run -c configs/experiments/h2.yaml --validate-only --print-config
uv run python -m adaptive_raptor.run -c configs/experiments/baseline.yaml \
    --set dataset.max_documents=5 --set runtime.concurrency=8      # override without editing
```

**Sections:**

| Section | What it controls |
|---|---|
| `experiment` | name, type, seed, output_dir, run_label |
| `dataset` | name, version, path, split, max_documents/questions, sample_seed, doc_ids_file, exclude_doc_ids_file |
| `chunking.variants[]` | `fixed` / `pic` / `permuted_boundaries` / `paragraph` with method-specific fields |
| `raptor` | embedding, summarization, summarization_length, num_layers, clustering, seeds, index_dir, cache_indexes, deterministic_umap |
| `retrieval` | strategies, budgets, include_raptor_original, raptor_original |
| `optimizer` | enabled, model_type, features, pca_dim, label_epsilon, theta / target_mean_tokens, fallback, checkpoint |
| `generation` | provider, model, temperature, max_output_tokens, prompt_template, prices (cost meter) |
| `evaluation` | metrics, baselines, matched_budget, oracle, ablation, bootstrap, equivalence_margin, save_contexts, output_formats |
| `runtime` | concurrency, batch_size, max_retries, use_cache, cache_dir, log_level |

**Validation happens before anything runs:**

1. Schema: types and ranges; unknown keys are rejected.
2. Experiment-type rules. For example, H2 needs a PIC variant and a `matched_budget`, and on the test split the router must be frozen.
3. Preflight, only for runners that do real work: API keys present and dataset prepared.

`--validate-only` never builds indexes, calls APIs or writes files.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Success, or config valid |
| 1 | Run failed or had errors |
| 2 | Invalid config, or environment not ready |
| 3 | Not implemented |

## 6. Running the RAPTOR baseline

```bash
uv run python -m adaptive_raptor.run --config configs/experiments/smoke_mock.yaml   # offline check
# add DEEPSEEK_API_KEY to .env, then:
uv run python -m adaptive_raptor.run --config configs/experiments/baseline.yaml
```

Each question runs every configured retrieval configuration, plus `raptor_original` when `include_raptor_original` is set. `raptor_original` is upstream `TreeRetriever.retrieve` unchanged: collapsed retrieval, `top_k=10`, 2,000 tokens.

**Each run writes to `outputs/<name>/<UTC timestamp>/`, never overwriting an earlier run:**

| File | Contents |
|---|---|
| `resolved_config.yaml` | Full config including defaults |
| `run_metadata.json` | Seed(s), git commit/dirty, dataset/version/split, RAPTOR + generator settings, upstream commit, package versions, platform |
| `traces.jsonl` | One row per question × config: retrieved node ids, levels and texts, context tokens, retrieval/generation latency, answer, input/output tokens, model version, F1, answer type, error |
| `indexes.jsonl` | Per document: cache hit, key hash, path, build stats (nodes per layer, summary tokens, build time) |
| `summary.{csv,json,md}` | Per-config mean F1 (overall and by answer type), context tokens, latencies, tokens, cost |
| `labels.jsonl` | Router training records (`LabelRecord`) |
| `predictions/*.jsonl` | Official-evaluator input format |
| `status.json` | `succeeded` / `failed` / `not_implemented`, message, metrics, duration |

**Index cache.** Location: `indexes/<dataset>/<version>/<variant>/seed<seed>/<doc_id>__<hash>.pkl` (+ `.json` with the full key).

- The key covers everything that changes a tree:
  - dataset and version, document, chunker config, and a hash of the actual chunk texts;
  - embedding and summarization models;
  - summarization length, layer count, clustering, tokenizer, `deterministic_umap`;
  - seed, upstream commit, format version.
- Changing any of these yields a new file.
- A file whose stored key doesn't match is **rejected** with `IndexCacheError`, never silently reused.
- LLM summaries and answers are also cached under `.cache/`, so reruns are free.

**Reproducibility.** Upstream RAPTOR leaves UMAP unseeded, so trees differed between processes even with fixed seeds (we measured it). With `raptor.deterministic_umap: true` (the default), the builder passes `random_state=seed` to upstream's UMAP calls during the build, and a test checks that separate processes build identical trees. `UPSTREAM.md` lists every behavioral difference from upstream.

## 7. Adding a chunker (e.g. PIC)

1. Implement `PICChunker.chunk(document) -> list[Chunk]` in `chunking/__init__.py` and remove `Stub` from the class bases.
2. Return text plus `start`/`end` offsets into `Document.text` and `meta` (e.g. `{"offsets": "exact", "pic_side": "rel"}`). The contract is at the top of that file.
3. Run it: set the variant in any `raptor_baseline` config. Nothing else changes, because the cache key already includes the chunker config and chunk texts.

   ```yaml
   chunking:
     variants: [{name: pic_capped, method: pic, max_tokens: 300}]
   ```

4. A new method also needs a model in `config/schema.py` (add it to `ChunkingVariant`) and a branch in `build_chunker`.

## 8. Implementing the optimizer

`optimizer/__init__.py` is the single typed implementation surface:

| Component | Interface |
|---|---|
| `features` | `FeatureExtractor.fit/transform(QueryContext)` |
| `acceptability` | `AcceptabilityModel.fit/predict_proba` |
| `cost` | `CostModel.fit(LabelRecord)/cost(ConfigRef)` |
| `selection` | `PlanSelector.select(probs, theta) -> RoutingDecision` |
| `calibration` | `calibrate_theta`, `theta_sweep` |
| `training` | `train_router`, `save_checkpoint`, `load_checkpoint` |
| `router` | `Router.route(QueryContext) -> RoutingDecision` |

How to work on it:

1. **Get training data.** Run `pilot_catalog.yaml`, or any `raptor_baseline` config with the full catalog. Its `labels.jsonl` is router training data. For cross-fitting run B, add `--set experiment.run_label=B --set generation.prompt_template=qasper_paraphrase`.
2. **Implement the contracts.** Fill them in and remove `Stub` only from completed classes. Calibration and training call `data.guard_split`, which refuses the test split.
3. **Plug it in.** `RouterPolicy(router)` is a `ConfigPolicy`. Pass it to `QAPipeline(config, run_dir, policy=...)`, and retrieval, generation and evaluation stay unchanged. Replace `H1OptimizerExperiment` in `experiments/__init__.py` when the optimizer is ready.

## 9. Adding and running experiments

1. **Write the config.** Copy a YAML file from `configs/experiments/` and validate it with `--validate-only`.
2. **For a new experiment type:**
   - add it to `ExperimentType` and add its rules in `config/validation.py`;
   - subclass `experiments.Experiment` (set `planned_steps()`, `run()`, and `requires_credentials` if it calls APIs);
   - register it with `experiments.register`.
3. **Build on the shared pieces.** Use `QAPipeline` for the QA loop, `evaluation.traces` for summaries and export, and `profiling` for label records.

## 10. Testing and troubleshooting

```bash
uv run pytest                    # offline suite: no network, no model downloads, no API calls
uv run pytest -m integration     # opt-in: SBERT download, one DeepSeek call, official evaluator
uv run ruff check . && uv run ruff format --check .
```

The offline suite covers:

- imports, including a check that validation never loads torch or RAPTOR;
- config schema and rules, and CLI exit codes;
- QASPER parsing and offsets, and download / verify / idempotency / checksum failure;
- subsets, and the test-split guard;
- upstream chunk equality and building with real UMAP + GMM;
- save/load/reuse, cache rejection, and same-seed reproducibility within and across processes;
- every strategy × budget, and equality with upstream collapsed retrieval;
- end-to-end CLI runs (artifacts, trace schema, index reuse), placeholder fail-fast, and preflight without a key.

| Problem | Fix |
|---|---|
| `environment is not ready` / exit code 2 | Add the named key to `.env`, or run `./scripts/download_datasets.sh --dataset qasper`. |
| `IndexCacheError` | The cached index doesn't match its key. Delete the named file; it will be rebuilt. |
| `not_implemented` / exit code 3 | The config uses a placeholder (PIC, router, research runner). |
| First real run is slow | SBERT (~420 MB) downloads once; numba compiles UMAP once per process. |
| `uv sync --frozen` fails after editing `pyproject.toml` | Run `uv lock` and commit `uv.lock`. |
| Linux needs CUDA torch | Remove the `[tool.uv.sources]` block in `pyproject.toml` and `uv lock`. |

---

## Implementation checklist

**Infrastructure: complete**

- [x] Upstream RAPTOR copied verbatim; provenance and behavioral differences in `UPSTREAM.md`
- [x] `uv` project, lockfile (macOS arm64 + Linux CPU), `setup.sh`, `check_environment.py`
- [x] QASPER download / verify / preprocess / manifest / subsets; offline fixture
- [x] Typed YAML config, per-type validation, preflight, CLI, run directories, metadata
- [x] Fixed-size chunking (upstream `split_text`) with approximate offsets
- [x] RAPTOR index building from any chunks; seeded and reproducible; cache with strict keys
- [x] Retrieval: leaf_only / summary_only / collapsed × budget, and upstream `raptor_original`
- [x] DeepSeek/OpenAI generator (credentials, retries, disk cache, usage, model version); mock
- [x] Official QASPER answer F1; traces; per-config summaries; label records; official-format predictions
- [x] Placeholder fail-fast; test-split guard; offline tests and opt-in integration tests

**Research: to implement** (backlog ticket in parentheses)

- [ ] PIC-capped, permuted-boundary and paragraph chunkers (T09)
- [ ] Acceptability labels, features, acceptability models, cost model, selection, θ calibration, training and checkpoints, Router (T11)
- [ ] Frontiers and cross-fitted oracle, budget matching, paired document bootstrap, evidence recall (T07, T14)
- [ ] Runners for H1, H2, the ablation, and the oracle interaction (T13)

**Not yet validated**

- No real (SBERT + DeepSeek) run has been executed, because there is no API key and paid calls were not authorized.
- The cross-check against the official evaluator is an opt-in integration test that has not been run.

## License

`src/raptor/` is © the RAPTOR authors, MIT license (`LICENSE-RAPTOR.txt`). See the QASPER
homepage for dataset terms.

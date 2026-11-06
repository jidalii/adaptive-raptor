# Adaptive-RAPTOR implementation guide

This repository keeps the working RAPTOR baseline and experiment infrastructure intact,
while concentrating the unfinished research work in three places marked **IMPLEMENT**.
The vendored `src/raptor/` package is byte-for-byte identical to the neighboring reference
checkout at `../raptor/raptor/`; reuse it instead of copying its algorithms.

## Final directory tree

```text
adaptive-raptor/
├── configs/experiments/          YAML experiment definitions
├── scripts/
│   ├── setup.sh                  reproducible Python/dependency setup + smoke run
│   ├── check_environment.py      offline environment and provenance checks
│   └── download_datasets.sh      QASPER download/verify/subset entry point
├── src/
│   ├── raptor/                   vendored upstream RAPTOR; do not duplicate or edit
│   └── adaptive_raptor/
│       ├── run.py                CLI: load YAML, validate, dispatch, map exit codes
│       ├── pipeline.py           shared dataset → index → retrieve → generate → score loop
│       ├── policy.py             fixed catalog policy and policy interface
│       ├── chunking/__init__.py  baseline chunker + adaptive contracts       [IMPLEMENT]
│       ├── optimizer/__init__.py router types, training/routing contracts     [IMPLEMENT]
│       ├── evaluation/
│       │   ├── metrics.py        official-compatible QASPER answer F1
│       │   ├── traces.py         trace schema, aggregation, result writers
│       │   └── research.py       frontiers/matching/bootstrap contracts       [IMPLEMENT]
│       ├── experiments/
│       │   ├── __init__.py       base class, registry, baseline, research runners
│       │   ├── runner.py         validation/preflight/seeding/status lifecycle
│       │   └── outputs.py        run directories and reproducibility metadata
│       ├── config/
│       │   ├── schema.py         typed YAML schema
│       │   ├── loader.py         YAML loading, overrides, resolved output
│       │   └── validation.py     experiment-specific semantic rules
│       ├── data/
│       │   ├── records.py        Document, Question, answer, and split records
│       │   ├── qasper.py         processed QASPER reader
│       │   ├── download.py       pinned download, verification, preprocessing
│       │   ├── subsets.py        deterministic document subsets
│       │   └── fixtures/         tiny offline smoke dataset
│       ├── indexing/
│       │   ├── builder.py        builds upstream RAPTOR trees from supplied chunks
│       │   └── store.py          cache keys, persistence, provenance constants
│       ├── retrieval.py          strategy/budget adapters over upstream TreeRetriever
│       ├── generation/__init__.py prompts, mock and OpenAI-compatible adapters
│       ├── models/               credentials, embedding and summarization adapters
│       ├── profiling.py          label records/store + label contract             [IMPLEMENT]
│       ├── cache.py              atomic JSON response cache
│       ├── paths.py              repository/data path resolution
│       └── stub.py               fail-fast marker for unfinished research code
└── tests/                        imports, config, data, RAPTOR, pipeline, CLI tests
```

`__init__.py` files that only export a package API and the YAML/fixture files are omitted
from the tree above. There are 35 project Python files under `src/adaptive_raptor`, down
from 63; removed files were thin one-class modules whose behavior now lives in the package
entry point shown above.

## What to implement

Do not change the infrastructure files unless an interface must evolve.

1. `chunking/__init__.py`
   - Implement `PICChunker.chunk`, `PermutedBoundaryChunker.chunk`, and
     `ParagraphChunker.chunk`.
   - Remove `Stub` from a class only after its implementation and tests are complete.
   - `FixedSizeChunker` is the existing upstream-compatible control; keep it reusable.
2. `profiling.py` and `optimizer/__init__.py`
   - Implement `acceptability_labels`, then feature extraction, acceptability and cost
     models, plan selection, calibration, checkpoint I/O, training, and `Router.route`.
   - `RouterPolicy` is already the adapter into `QAPipeline`.
3. `evaluation/research.py`
   - Implement fixed/oracle frontiers, budget matching, and paired document bootstrap.
4. `experiments/__init__.py`
   - Replace each `PlaceholderExperiment` research runner with orchestration that composes
     the implemented components. The working `RaptorBaselineExperiment` is the example.

## Interface contracts

### Adaptive chunking

```python
Chunker.chunk(document: Document) -> list[Chunk]
```

- Non-empty input returns non-empty chunk texts in source order and drops no sentence.
- `n_tokens` uses the configured RAPTOR tokenizer.
- `start` and `end` are offsets into `Document.text` when available.
- Set `meta["offsets"]` to `exact`, `approximate`, or `none`.
- Chunk configuration and resulting text are already included in the index cache key.

### Retrieval optimizer

```python
FeatureExtractor.fit(train).transform(queries)
AcceptabilityModel.fit(features, labels, groups).predict_proba(features)
CostModel.fit(dev_records).cost(config)
PlanSelector.select(probabilities, theta) -> RoutingDecision
Router.route(QueryContext) -> RoutingDecision
```

- Use train/dev data for fitting and calibration; `guard_split` rejects test leakage.
- Probability dictionaries are keyed by `ConfigRef.id` (for example
  `collapsed@1000`).
- The selected `RoutingDecision.config` must be a member of the frozen catalog.
- Persist every fitted artifact and the calibrated threshold in `RouterCheckpoint`.
- Keep routing deterministic for a fixed checkpoint and query embedding.

### Experiment runners

```python
Experiment.planned_steps() -> list[str]
Experiment.run() -> ExperimentResult
```

- Reuse `QAPipeline`; inject a `RouterPolicy` or restricted variants/catalog as needed.
- Return `NOT_IMPLEMENTED` while required research logic is still a stub.
- Never report success after silently skipping an algorithm.
- Keep YAML validation and preflight before data/model/API work.

## Suggested implementation order

1. Run the offline baseline and preserve its passing tests.
2. Implement chunkers and unit-test offsets, coverage, determinism, and token caps.
3. Run `pilot_catalog.yaml` to produce `labels.jsonl`.
4. Implement acceptability labels, cost and selection before model training.
5. Implement features, acceptability training, calibration, and checkpoint round trips.
6. Implement `Router.route`, connect H1, then implement statistical evaluation.
7. Connect H2, ablation, and oracle-interaction runners last.

## Commands

```bash
# Reproducible setup; add --with-tests for the full offline suite
./scripts/setup.sh --with-tests

# Download/verify QASPER and optionally create a deterministic pilot subset
./scripts/download_datasets.sh --dataset qasper
./scripts/download_datasets.sh --dataset qasper --verify
./scripts/download_datasets.sh --dataset qasper --make-subset pilot --split dev --n-docs 30 --seed 0

# Validate YAML without model loading, output creation, or API calls
uv run python -m adaptive_raptor.run -c configs/experiments/h1.yaml --validate-only

# Offline end-to-end smoke run and a real baseline run
uv run python -m adaptive_raptor.run -c configs/experiments/smoke_mock.yaml
uv run python -m adaptive_raptor.run -c configs/experiments/baseline.yaml

# Verification
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Real generation needs the provider key named by the YAML (the supplied baseline uses
`DEEPSEEK_API_KEY`). Integration tests remain opt-in with `uv run pytest -m integration`.

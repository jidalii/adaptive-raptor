# Datasets

## QASPER v0.3 (primary)

**Source.** The official AllenAI release ([homepage](https://allenai.org/data/qasper)):

| Archive | Contents | SHA-256 (pinned in `data/download.py`) |
|---|---|---|
| `qasper-train-dev-v0.3.tgz` (10.8 MB) | train + dev JSON | `a28fdf96…91bdb5a` |
| `qasper-test-and-evaluator-v0.3.tgz` (3.9 MB) | test JSON + `qasper_evaluator.py` | `72a52a41…171bdd72` |

Both come from `https://qasper-dataset.s3.us-west-2.amazonaws.com/`. Their MD5 hashes matched the S3 ETags when the checksums were pinned (2026-11-06). The archive READMEs don't state a license; see the homepage for dataset terms.

### Download

```bash
./scripts/download_datasets.sh --dataset qasper              # download, extract, preprocess, verify
./scripts/download_datasets.sh --dataset qasper --verify     # offline check only
./scripts/download_datasets.sh --dataset qasper --force      # redo everything
./scripts/download_datasets.sh --dataset qasper --data-dir /big/disk/data
```

Re-running is free:

- An archive is downloaded only if it's missing or its checksum is wrong.
- Preprocessing re-runs only if the archive checksums, the preprocessing version, or the processed files' checksums changed.

`--verify` works offline and checks all of the following:

- archive checksums, and processed-file checksums against the manifest;
- the exact paper and question counts per split;
- question → paper references and unique question ids;
- every paragraph offset;
- that no paper appears in two splits.

### Data root

The data root is chosen in this order:

1. `--data-dir`
2. `$ADAPTIVE_RAPTOR_DATA_DIR`
3. `./data`

In experiment YAML, `dataset.path` sets the same root. Nothing depends on a particular machine's paths.

```text
<data root>/
├── raw/qasper/v0.3/
│   ├── qasper-train-dev-v0.3.tgz, qasper-test-and-evaluator-v0.3.tgz
│   └── extracted/  qasper-{train,dev,test}-v0.3.json, qasper_evaluator.py, README*.md
├── processed/qasper/v0.3/
│   ├── manifest.json
│   └── {train,dev,test}/documents.jsonl, questions.jsonl
└── subsets/qasper/v0.3/<split>_<name>.txt
```

### Splits (official, unchanged)

| Split | Papers | Questions | Use |
|---|---|---|---|
| train | 888 | 2,593 | router training labels |
| dev | 281 | 1,005 | pilot, θ calibration, comparator selection |
| test | 416 | 1,451 | final evaluation only |

Test data must never be used for fitting. `adaptive_raptor.data.guard_split(split, purpose)` raises `TestSplitLeakageError` for `train` / `calibrate` / `select` on `test`; router training and calibration call it.

### Internal schema (`data/records.py`)

- **`Document`:** `doc_id` (the paper id), `title`, `text`, and `paragraphs` (each with `section`, `text`, `start`, `end`).
  - `text` is the title, then the abstract, then every non-empty section paragraph, joined with blank lines.
  - Each paragraph's offsets are exact: `text[start:end] == paragraph.text`.
- **`Question`:** `question_id`, `doc_id`, `text`, and `answers` (one per annotator).
- **`GoldAnswer`:** `text`, `answer_type`, `evidence`, `highlighted_evidence`, `annotation_id`.
  - `answer_type` and `text` follow the official evaluator exactly:

    | Type | Answer text |
    |---|---|
    | `extractive` | spans joined by ", " |
    | `abstractive` | the free-form answer |
    | `boolean` | Yes / No |
    | `none` | "Unanswerable" |

### Manifest

`processed/qasper/v0.3/manifest.json` records:

- source URLs and archive checksums;
- the preprocessing version and its rules;
- per-split counts and processed-file checksums;
- the creation time.

Every experiment copies the dataset name, version, split and file checksums into `status.json` (under `metrics.dataset`). Its `run_metadata.json` also records the dataset name, version and split.

### Paper-level subsets

Subsets are always chosen by **paper**, never by question, so a paper's questions stay together.

```bash
# 30 pilot papers from dev (seeded, sorted ids, same on every machine)
./scripts/download_datasets.sh --dataset qasper --make-subset pilot --split dev --n-docs 30 --seed 0
# the rest of dev, excluding the pilot
./scripts/download_datasets.sh --dataset qasper --make-subset calib --split dev --n-docs 200 \
    --seed 1 --exclude data/subsets/qasper/v0.3/dev_pilot.txt
```

Use them in YAML through `dataset.doc_ids_file` and `dataset.exclude_doc_ids_file`. `max_documents` with `sample_seed` gives seeded subsets without a file. Commit the id files you report results on (they are small), e.g. under `results/question_ids/`.

## Offline fixture (`qasper_tiny`)

`src/adaptive_raptor/data/fixtures/qasper_tiny.json` holds 3 synthetic papers in the official raw format, covering every answer type. Use it with `dataset.name: qasper_tiny` for smoke runs and tests. It is **not** QASPER, and results on it mean nothing.

## QuALITY (planned)

Not supported yet (backlog T16). `--dataset quality` exits with a message.

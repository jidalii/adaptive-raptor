# T02 · QASPER download, splits and pilot set

Tier: 1 · Owner: _ · Depends: T01

## What

- `./scripts/download_datasets.sh --dataset qasper` (→ `adaptive_raptor/data/download.py`): official QASPER v0.3 from AllenAI's S3 with pinned SHA-256, extracted to `data/raw/`, preprocessed to `data/processed/qasper/v0.3/{train,dev,test}/` with a `manifest.json`. Idempotent; `--verify` checks everything offline. This is the **only** code path that downloads data.
- `data/qasper.py` + `data/records.py`: `Document` (full text with exact paragraph offsets), `Question`, and `GoldAnswer` per annotator, using the official evaluator's answer types (extractive / abstractive / boolean / none). Splits: `train`, `dev` (official dev file), `test`.
- `--make-subset NAME --split dev --n-docs N --seed S [--exclude FILE]`: seeded, paper-level id lists under `data/subsets/`.
- **Pilot set:** a seeded sample of dev papers (about 100 questions) for T08, plus a disjoint dev remainder for calibration and selection.
- Commit the id lists you report on to `results/question_ids/` (`dev_pilot.txt`, …).
- `docs/DATA.md`: the dataset tutorial.

## Done when

- A fresh clone can follow `docs/DATA.md` and end with verified QASPER data.
- The sampler gives identical id lists on reruns; pilot and the rest of dev share no papers.
- The pilot and remainder id lists are committed under `results/question_ids/`.

## Description

Splitting by **paper** keeps the router from memorizing a paper's topic (proposal §7). Exact offsets let gold evidence map identically onto every chunking variant.

## Notes

- Nov 6: download, verify, subsets, loader, `docs/DATA.md` and tests done. Counts match the release: 888 / 281 / 416 papers and 2,593 / 1,005 / 1,451 questions; no annotation lacks an answer.
- **Left:** create the pilot and remainder subsets (`pilot_catalog.yaml` expects `data/subsets/qasper/v0.3/dev_pilot.txt`) and commit copies to `results/question_ids/`. Git is now initialized.

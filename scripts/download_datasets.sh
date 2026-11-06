#!/usr/bin/env bash
# Download, verify and preprocess datasets (idempotent).
#
#   ./scripts/download_datasets.sh --dataset qasper
#   ./scripts/download_datasets.sh --dataset qasper --verify
#   ./scripts/download_datasets.sh --dataset qasper --data-dir /path/to/data
#   ./scripts/download_datasets.sh --dataset qasper --make-subset pilot --split dev --n-docs 30 --seed 0
#
# Data root: --data-dir > $ADAPTIVE_RAPTOR_DATA_DIR > ./data
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m adaptive_raptor.data.download "$@"

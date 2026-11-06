"""Filesystem locations. Nothing here is tied to a particular machine.

Data root resolution order: explicit argument > ``$ADAPTIVE_RAPTOR_DATA_DIR`` > ``./data``.
Relative paths are resolved against the current working directory (run from the repo root).
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "ADAPTIVE_RAPTOR_DATA_DIR"
REPO_ROOT = Path(__file__).resolve().parents[2]  # valid for the editable/dev install


def data_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get(DATA_DIR_ENV)
    return Path(env).expanduser() if env else Path("data")


def raw_dir(dataset: str, version: str, root: str | Path | None = None) -> Path:
    return data_root(root) / "raw" / dataset / version


def processed_dir(dataset: str, version: str, root: str | Path | None = None) -> Path:
    return data_root(root) / "processed" / dataset / version

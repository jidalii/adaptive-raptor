"""Datasets: shared records, QASPER loading, deterministic subsets."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_raptor.data.records import (
    DatasetSplit,
    Document,
    GoldAnswer,
    Paragraph,
    Question,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "qasper_tiny.json"

__all__ = [
    "DatasetSplit",
    "Document",
    "GoldAnswer",
    "Paragraph",
    "Question",
    "TestSplitLeakageError",
    "guard_split",
    "load_dataset",
]


class TestSplitLeakageError(RuntimeError):
    """Raised when test data would be used for training or calibration."""

    __test__ = False  # not a pytest test class


def guard_split(split: str, purpose: str) -> None:
    """Call before any fitting step. Test data may only be used for final evaluation."""
    if split == "test" and purpose in ("train", "calibrate", "select"):
        raise TestSplitLeakageError(f"refusing to use the test split for '{purpose}'")


def load_dataset(section) -> DatasetSplit:
    """Load a split as configured by a ``DatasetSection``. Never downloads."""
    from adaptive_raptor.data import qasper
    from adaptive_raptor.data.subsets import read_id_file, select
    from adaptive_raptor.paths import processed_dir

    manifest: dict = {}
    if section.name == "qasper":
        pdir = processed_dir("qasper", section.version, section.path)
        docs, qs = qasper.load_processed(pdir, section.split)
        mpath = pdir / "manifest.json"
        if mpath.is_file():
            m = json.loads(mpath.read_text())
            manifest = {k: m.get(k) for k in ("dataset", "version", "raw_sha256",
                                              "preprocess_version", "sources")}
            manifest["split_files"] = m.get("splits", {}).get(section.split, {}).get("files")
    elif section.name == "narrativeqa":
        pdir = processed_dir("narrativeqa", section.version, section.path)
        docs, qs = qasper.load_processed(pdir, section.split)  # same JSONL record format
        mpath = pdir / "manifest.json"
        if mpath.is_file():
            m = json.loads(mpath.read_text())
            manifest = {k: m.get(k) for k in ("dataset", "version", "revision", "raw_sha256",
                                              "preprocess_version")}
            manifest["split_files"] = m.get("splits", {}).get(section.split, {}).get("files")
    elif section.name == "qasper_tiny":
        raw = json.loads(FIXTURE_PATH.read_text())[section.split]
        docs, qs, _ = qasper.parse_split(raw)
        manifest = {"dataset": "qasper_tiny", "version": "fixture",
                    "preprocess_version": qasper.PREPROCESS_VERSION}
    else:
        raise NotImplementedError(f"dataset '{section.name}' is not supported yet")

    include = read_id_file(section.doc_ids_file) if section.doc_ids_file else None
    exclude = read_id_file(section.exclude_doc_ids_file) if section.exclude_doc_ids_file else None
    docs, qs = select(
        docs, qs, include=include, exclude=exclude, max_documents=section.max_documents,
        max_questions=section.max_questions, seed=section.sample_seed,
    )
    return DatasetSplit(section.name, section.version, section.split, docs, qs, manifest)

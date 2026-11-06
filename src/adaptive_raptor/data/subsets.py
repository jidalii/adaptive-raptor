"""Deterministic, paper-level subsets. Questions always follow their papers."""

from __future__ import annotations

import random
from pathlib import Path

from adaptive_raptor.data.records import Document, Question


def read_id_file(path: Path) -> list[str]:
    ids = [ln.strip() for ln in Path(path).read_text().splitlines()]
    return [i for i in ids if i and not i.startswith("#")]


def write_id_file(path: Path, ids: list[str], header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {h}" for h in header.splitlines()] + list(ids)
    path.write_text("\n".join(lines) + "\n")


def sample_doc_ids(doc_ids: list[str], n: int | None, seed: int) -> list[str]:
    """Same (ids, n, seed) -> same sample, on every machine. Returned in sorted order."""
    ids = sorted(doc_ids)
    if n is None or n >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, n))


def select(
    documents: list[Document],
    questions: list[Question],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_documents: int | None = None,
    max_questions: int | None = None,
    seed: int = 0,
) -> tuple[list[Document], list[Question]]:
    ids = [d.doc_id for d in documents]
    if include is not None:
        keep = set(include)
        unknown = keep - set(ids)
        if unknown:
            raise ValueError(f"{len(unknown)} requested doc ids are not in this split")
        ids = [i for i in ids if i in keep]
    if exclude is not None:
        drop = set(exclude)
        ids = [i for i in ids if i not in drop]
    chosen = set(sample_doc_ids(ids, max_documents, seed))
    docs = [d for d in documents if d.doc_id in chosen]
    qs = [q for q in questions if q.doc_id in chosen]
    if max_questions is not None:
        qs = qs[:max_questions]
        used = {q.doc_id for q in qs}
        docs = [d for d in docs if d.doc_id in used]
    return docs, qs

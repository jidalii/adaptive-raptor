"""NarrativeQA (Kočiský et al., 2018): full stories, free-form answers (2 references).

Source: Hugging Face ``deepmind/narrativeqa`` at a pinned revision (see
``adaptive_raptor.data.download``). The parquet shards are first converted to raw JSONL by
``scripts/narrativeqa_parquet_to_jsonl.py``, which runs in an isolated environment because
of pyarrow. This module turns that raw JSONL into the shared records.

Document text (``PREPROCESS_VERSION`` 1):
1. Movie scripts are IMSDb HTML: ``<script>``/``<style>`` blocks are dropped, the other tags
   are replaced by a space, and HTML entities are unescaped. Books are plain Gutenberg text.
2. The story is cut to the dataset's ``start``/``end`` markers. The markers were tokenized
   by the dataset authors (for example "WHO S YOUR", "new eBooks ."), so they are matched
   as sequences of alphanumeric tokens, case-insensitively: start = first occurrence, end =
   last occurrence after the start. If a marker is not found, that side is not cut and the
   document is counted under ``marker_fallback`` (no silent guessing).
3. A byte-order mark is removed, trailing spaces are stripped, and runs of 3+ newlines
   become 2. Line structure is kept (upstream ``split_text`` splits on newlines).

Questions: one per parquet row, id ``<doc_id>:<index_in_doc>``. Both reference answers are
kept as ``abstractive`` GoldAnswers. The dataset has no evidence annotations.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

from adaptive_raptor.data.records import Document, GoldAnswer, Question

PREPROCESS_VERSION = 1
SPLITS = {"dev": "validation", "train": "train", "test": "test"}  # ours -> HF split name

_SCRIPT = re.compile(r"(?is)<(script|style)\b.*?</\1\s*>")
_TAG = re.compile(r"(?s)<[^>]*>")
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def strip_html(text: str) -> str:
    return html.unescape(_TAG.sub(" ", _SCRIPT.sub(" ", text)))


def _find_tokens(
    tokens: list[tuple[str, int, int]], marker: str, last: bool
) -> tuple[int, int] | None:
    """Char span of the first (or last) occurrence of ``marker``'s token sequence."""
    want = [t.lower() for t in _TOKEN.findall(marker)]
    if not want:
        return None
    n = len(want)
    rng = range(len(tokens) - n, -1, -1) if last else range(len(tokens) - n + 1)
    for i in rng:
        if tokens[i][0] == want[0] and all(tokens[i + k][0] == want[k] for k in range(1, n)):
            return tokens[i][1], tokens[i + n - 1][2]
    return None


def cut_story(text: str, start: str, end: str) -> tuple[str, dict[str, bool]]:
    tokens = [(m.group().lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
    s = _find_tokens(tokens, start, last=False)
    lo = s[0] if s else 0
    rest = [t for t in tokens if t[1] >= lo]
    e = _find_tokens(rest, end, last=True)
    hi = e[1] if e else len(text)
    if hi <= lo:  # inconsistent markers: keep everything
        lo, hi, s, e = 0, len(text), None, None
    return text[lo:hi], {"start_found": s is not None, "end_found": e is not None}


def clean_text(text: str) -> str:
    text = text.replace("﻿", "").replace("ï»¿", "")
    text = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_document(raw: dict) -> tuple[Document, dict]:
    text = strip_html(raw["text"]) if raw["kind"] == "movie" else raw["text"]
    story, found = cut_story(text, raw.get("start") or "", raw.get("end") or "")
    story = clean_text(story)
    meta = {
        "doc_id": raw["id"],
        "kind": raw["kind"],
        "title": raw.get("title", ""),
        "word_count_reported": raw.get("word_count"),
        "words": len(story.split()),
        **found,
    }
    return Document(doc_id=raw["id"], text=story, title=raw.get("title", "")), meta


def parse_split(raw_docs: list[dict], raw_questions: list[dict]):
    docs, metas = [], []
    for r in raw_docs:
        d, m = parse_document(r)
        docs.append(d)
        metas.append(m)
    known = {d.doc_id for d in docs}
    qs = []
    for r in raw_questions:
        if r["doc_id"] not in known:
            raise ValueError(f"question references unknown document {r['doc_id']}")
        answers = tuple(
            GoldAnswer(text=a, answer_type="abstractive", annotation_id=str(i))
            for i, a in enumerate(r["answers"])
            if a and a.strip()
        )
        qs.append(
            Question(f"{r['doc_id']}:{r['index_in_doc']}", r["doc_id"], r["question"], answers)
        )
    stats = {
        "marker_fallback_start": sum(not m["start_found"] for m in metas),
        "marker_fallback_end": sum(not m["end_found"] for m in metas),
        "kinds": dict(Counter(m["kind"] for m in metas)),
        "questions_without_answers": sum(not q.answers for q in qs),
    }
    return docs, qs, metas, stats


def read_raw(raw_dir: Path, hf_split: str) -> tuple[list[dict], list[dict]]:
    def rows(name):
        path = raw_dir / f"{hf_split}_{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    return rows("documents"), rows("questions")


def load_meta(processed: Path, split: str) -> list[dict]:
    path = processed / split / "meta.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

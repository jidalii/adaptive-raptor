"""Sentence-level extractive compression of retrieved leaf chunks (offline, no LLM calls).

Phase 1 (feasibility) is used only by ``scripts/compression_feasibility.py``. The pipeline
does not call this module yet.

Pipeline for one question:

1. ``sentence_units``: split each retrieved leaf chunk back into the sentence units that
   built it. Upstream ``raptor.utils.split_text`` splits text on ``. ! ? \\n``, drops the
   delimiters, and joins the pieces of a chunk with spaces, so unit boundaries show up
   as runs of two or more spaces. Pieces that the delimiter split wrongly (decimals such
   as "3 0", abbreviations such as "e g") are merged back into the previous unit. Every
   unit keeps its source leaf index, its position in the leaf and its character span.
2. Score units against the question: ``BM25`` (IDF over all units of the paper),
   cosine similarity of sentence embeddings (the retrieval SBERT model, run locally),
   their reciprocal-rank fusion, or a seeded random order as a floor.
3. ``select``: greedy by score under a token budget. Near-duplicate units are skipped
   (content-token Jaccard). A unit that depends on its predecessor (starts with an
   anaphor or connective, or is very short) can bring that predecessor along. The
   predecessor must also come from the retrieved pool.
4. ``assemble``: the chosen units in document order. Consecutive units are joined by a
   space and gaps by a blank line. The token count of the assembled string never exceeds
   the budget.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

WORD = re.compile(r"[a-z0-9]+")
_BOUNDARY = re.compile(r"\s{2,}|\n+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "for",
        "on",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "by",
        "with",
        "as",
        "at",
        "from",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "their",
        "they",
        "we",
        "our",
        "which",
        "what",
        "how",
        "do",
        "does",
        "did",
        "can",
        "not",
        "no",
        "yes",
        "than",
        "into",
        "using",
        "use",
        "used",
        "also",
        "such",
        "each",
        "other",
        "more",
        "most",
        "any",
        "all",
        "some",
        "there",
        "here",
        "have",
        "has",
        "had",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "about",
        "between",
        "both",
        "over",
        "under",
        "via",
        "per",
        "i",
        "you",
    ]
)

# A unit that starts with one of these usually needs the preceding sentence.
DEPENDENT_STARTS = frozenset(
    [
        "this",
        "these",
        "that",
        "those",
        "it",
        "its",
        "they",
        "their",
        "them",
        "such",
        "however",
        "thus",
        "therefore",
        "hence",
        "moreover",
        "furthermore",
        "additionally",
        "also",
        "besides",
        "here",
        "there",
        "which",
        "both",
        "similarly",
        "consequently",
        "instead",
        "otherwise",
        "then",
        "finally",
        "second",
        "third",
        "last",
    ]
)


def content_tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


# ---------------------------------------------------------------------------- settings


class CompressionSettings(BaseModel):
    """``configs/analysis/compression_feasibility.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str = "compression_feasibility"
    source_config: str = "leaf_only@2000"
    runs: dict[str, str] = Field(description="split -> label run directory (traces.jsonl)")
    budgets: list[int] = [600, 1000, 1400]
    chunk_baselines: list[int] = [400, 600, 1000, 1400, 2000]
    scorers: list[Literal["bm25", "sbert", "hybrid", "random"]] = [
        "bm25",
        "sbert",
        "hybrid",
        "random",
    ]
    neighbors: list[bool] = [False, True]
    redundancy_jaccard: float = Field(0.8, gt=0, le=1)
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    dependent_max_words: int = 5
    embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    tokenizer: str = "cl100k_base"
    seed: int = 224
    out_dir: str = "outputs/diagnostics/compression_feasibility"


# ---------------------------------------------------------------------------- units


@dataclass(frozen=True)
class Unit:
    leaf: int  # leaf node index (= chunk position in the paper)
    pos: int  # unit position within the leaf
    start: int  # character span in the leaf text
    end: int
    text: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.leaf, self.pos)


def _is_continuation(prev: str, cur: str) -> bool:
    """True if ``cur`` is a piece wrongly split off ``prev`` by the '.' delimiter."""
    if not prev:
        return False
    if prev[-1].isdigit() and cur[:1].isdigit():  # decimal: "3 0", "92 4%"
        return True
    if cur[:1].islower():  # abbreviation: "e g", "et al", "i e", "vs"
        return True
    return len(prev.split()) <= 1  # dangling one-word piece: "Fig", "Eq", "Sec"


def sentence_units(leaf_index: int, text: str) -> list[Unit]:
    """Split one leaf chunk into the sentence units it was built from."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BOUNDARY.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    merged: list[list[int]] = []
    for s, e in spans:
        piece = text[s:e].strip()
        if not piece:
            continue
        s = s + (len(text[s:e]) - len(text[s:e].lstrip()))
        e = s + len(piece)
        if merged and _is_continuation(text[merged[-1][0] : merged[-1][1]], piece):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [
        Unit(leaf_index, i, s, e, " ".join(text[s:e].split())) for i, (s, e) in enumerate(merged)
    ]


def pool_units(leaves: Sequence[tuple[int, str]]) -> list[Unit]:
    """Units of the retrieved leaves ``(index, text)``, in document order."""
    out: list[Unit] = []
    for idx, text in sorted(leaves):
        out.extend(sentence_units(idx, text))
    return out


# ---------------------------------------------------------------------------- scoring


class BM25:
    """Okapi BM25 over a fixed corpus of unit texts (for example, all units of a paper)."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        docs = [content_tokens(t) for t in corpus]
        self.n = len(docs)
        self.avgdl = float(np.mean([len(d) for d in docs])) if docs else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        q = set(content_tokens(query))
        out = np.zeros(len(texts))
        for i, t in enumerate(texts):
            toks = content_tokens(t)
            tf = Counter(toks)
            norm = self.k1 * (1 - self.b + self.b * len(toks) / max(self.avgdl, 1e-9))
            out[i] = sum(
                self.idf.get(w, 0.0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf
            )
        return out


def rrf(*scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion. Ties keep input order through the stable sort."""
    total = np.zeros(len(scores[0]))
    for s in scores:
        ranks = np.empty(len(s))
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        total += 1.0 / (k + ranks)
    return total


# ---------------------------------------------------------------------------- selection


def is_dependent(text: str, max_words: int = 5) -> bool:
    words = text.split()
    if not words:
        return False
    first = re.sub(r"[^a-z]", "", words[0].lower())
    return first in DEPENDENT_STARTS or len(words) <= max_words


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def assemble(units: Sequence[Unit]) -> str:
    """Units in document order: a space inside a contiguous run, a blank line at gaps."""
    parts: list[str] = []
    prev: Unit | None = None
    for u in sorted(units, key=lambda x: x.key):
        if prev is None:
            parts.append(u.text)
        else:
            contiguous = (u.leaf == prev.leaf and u.pos == prev.pos + 1) or (
                u.leaf == prev.leaf + 1 and u.pos == 0
            )
            parts.append((" " if contiguous else "\n\n") + u.text)
        prev = u
    return "".join(parts)


def select(
    units: Sequence[Unit],
    scores: np.ndarray,
    budget: int,
    encode,
    *,
    redundancy_jaccard: float = 0.8,
    neighbors: bool = False,
    dependent_max_words: int = 5,
) -> list[Unit]:
    """Greedy top-score selection under ``budget`` tokens of the assembled context.

    ``encode`` is a tokenizer's ``encode``. Units that don't fit are skipped, and smaller
    units further down the ranking may still fill the budget. If ``neighbors`` is set, a
    dependent unit brings its predecessor in the pool (same leaf, or the last unit of the
    previous leaf when that leaf was retrieved) if the pair fits. Otherwise it is taken alone.
    """
    tok = [len(encode(u.text)) for u in units]
    toks = [set(content_tokens(u.text)) for u in units]
    index = {u.key: i for i, u in enumerate(units)}
    last_pos: dict[int, int] = {}
    for u in units:
        last_pos[u.leaf] = max(last_pos.get(u.leaf, -1), u.pos)

    def predecessor(i: int) -> int | None:
        u = units[i]
        if u.pos > 0:
            return index.get((u.leaf, u.pos - 1))
        if u.leaf - 1 in last_pos:
            return index.get((u.leaf - 1, last_pos[u.leaf - 1]))
        return None

    chosen: set[int] = set()
    used = 0
    sep = 2  # upper bound on the separator tokens one unit adds
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        i = int(i)
        if i in chosen or any(_jaccard(toks[i], toks[j]) >= redundancy_jaccard for j in chosen):
            continue
        group = [i]
        if neighbors and is_dependent(units[i].text, dependent_max_words):
            p = predecessor(i)
            if p is not None and p not in chosen:
                group = [p, i]
        cost = sum(tok[g] + sep for g in group)
        if used + cost > budget and len(group) == 2:
            group, cost = [i], tok[i] + sep
        if used + cost > budget:
            continue
        chosen.update(group)
        used += cost
    picked = [units[i] for i in sorted(chosen, key=lambda j: units[j].key)]
    # exact check on the assembled string; drop the lowest-scored unit until it fits
    by_score = sorted(chosen, key=lambda j: scores[j])
    while picked and len(encode(assemble(picked))) > budget:
        drop = by_score.pop(0)
        picked = [u for u in picked if u.key != units[drop].key]
    return picked

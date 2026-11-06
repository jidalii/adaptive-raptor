"""Leaf chunking interfaces, the RAPTOR baseline, and adaptive extension points.

The baseline delegates text splitting to upstream RAPTOR. ``PICChunker`` and its
controls deliberately remain explicit stubs: this is the one file to edit when
implementing the adaptive chunking study.

Contract for every chunker:

* a non-empty document produces non-empty chunks in document order;
* no source sentence is dropped;
* ``start``/``end`` are character offsets into ``Document.text`` when known;
* ``meta["offsets"]`` is ``exact``, ``approximate``, or ``none``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from adaptive_raptor.config.schema import (
    ChunkingVariant,
    FixedChunking,
    ParagraphChunking,
    PermutedBoundaryChunking,
    PICChunking,
)
from adaptive_raptor.data.records import Document
from adaptive_raptor.stub import Stub


@dataclass(frozen=True)
class Chunk:
    text: str
    n_tokens: int
    start: int | None = None
    end: int | None = None
    sentence_ids: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into leaf chunks."""


def _approx_start(text: str, chunk: str, cursor: int) -> int | None:
    words = re.findall(r"\S+", chunk)[:6]
    if not words:
        return None
    match = re.compile(r"\s*".join(re.escape(word) for word in words)).search(text, cursor)
    return match.start() if match else None


class FixedSizeChunker(Chunker):
    """Original RAPTOR splitting with approximate source offsets added."""

    def __init__(self, cfg: FixedChunking, tokenizer=None) -> None:
        self.cfg = cfg
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import tiktoken

            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def chunk(self, document: Document) -> list[Chunk]:
        from raptor.utils import split_text

        texts = split_text(document.text, self.tokenizer, self.cfg.chunk_size, self.cfg.overlap)
        starts: list[int | None] = []
        cursor = 0
        for text in texts:
            start = _approx_start(document.text, text, cursor)
            starts.append(start)
            if start is not None and self.cfg.overlap == 0:
                cursor = start + 1
        chunks = []
        for i, text in enumerate(texts):
            next_start = next(
                (start for start in starts[i + 1 :] if start is not None), len(document.text)
            )
            start = starts[i]
            chunks.append(
                Chunk(
                    text=text,
                    n_tokens=len(self.tokenizer.encode(text)),
                    start=start,
                    end=next_start if start is not None else None,
                    meta={"offsets": "approximate" if start is not None else "none"},
                )
            )
        return chunks


# Research extension points. Keep their signatures stable; replace Stub and the method
# bodies when implementing the algorithms.


class PICChunker(Stub, Chunker):
    def __init__(self, cfg: PICChunking, summarizer=None, embedder=None) -> None:
        self.cfg = cfg
        self.summarizer = summarizer
        self.embedder = embedder

    def chunk(self, document: Document) -> list[Chunk]:
        """Summary -> sentence cosine -> threshold -> capped same-side runs."""
        raise NotImplementedError("PIC-capped chunking is not implemented yet")


class PermutedBoundaryChunker(Stub, Chunker):
    def __init__(self, cfg: PermutedBoundaryChunking, source: Chunker | None = None) -> None:
        self.cfg = cfg
        self.source = source

    def chunk(self, document: Document) -> list[Chunk]:
        """Reuse PIC chunk lengths at randomly permuted sentence boundaries."""
        raise NotImplementedError("permuted-boundary control is not implemented yet")


class ParagraphChunker(Stub, Chunker):
    def __init__(self, cfg: ParagraphChunking) -> None:
        self.cfg = cfg

    def chunk(self, document: Document) -> list[Chunk]:
        """Use author paragraphs, splitting long paragraphs at sentence boundaries."""
        raise NotImplementedError("paragraph control is not implemented yet")


def build_chunker(cfg: ChunkingVariant) -> Chunker:
    """Build a chunker from one typed YAML chunking variant."""
    if isinstance(cfg, FixedChunking):
        return FixedSizeChunker(cfg)
    if isinstance(cfg, PICChunking):
        return PICChunker(cfg)
    if isinstance(cfg, PermutedBoundaryChunking):
        return PermutedBoundaryChunker(cfg)
    if isinstance(cfg, ParagraphChunking):
        return ParagraphChunker(cfg)
    raise TypeError(f"unknown chunking config: {type(cfg).__name__}")


__all__ = [
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "PICChunker",
    "PermutedBoundaryChunker",
    "ParagraphChunker",
    "build_chunker",
]

"""Embedding models for RAPTOR, as upstream ``BaseEmbeddingModel`` instances.

- ``sbert``: upstream ``raptor.SBertEmbeddingModel`` (downloads the model on first use).
- ``openai``: upstream ``raptor.OpenAIEmbeddingModel``.
- ``mock``: deterministic hashed bag-of-words vectors for offline tests. Not for results.

All are wrapped in :class:`CachedEmbeddingModel`, so a query embedded once is reused by
every retrieval configuration (upstream retrieval re-embeds the query on each call).
"""

from __future__ import annotations

import hashlib
import re
import threading

import numpy as np

MOCK_DIM = 256


def _base():
    from raptor.EmbeddingModels import BaseEmbeddingModel

    return BaseEmbeddingModel


def make_mock_embedding_model():
    class MockHashEmbeddingModel(_base()):
        """Hashed bag-of-words; similar texts get similar vectors. Offline and deterministic."""

        def create_embedding(self, text):
            vec = np.zeros(MOCK_DIM, dtype=np.float32)
            vec[0] = 1.0  # never all-zero (cosine distance stays defined)
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[1 + h % (MOCK_DIM - 1)] += 1.0
            return vec / np.linalg.norm(vec)

    return MockHashEmbeddingModel()


def make_cached(model):
    class CachedEmbeddingModel(_base()):
        def __init__(self, inner) -> None:
            self.inner = inner
            self._cache: dict[str, object] = {}
            self._lock = threading.Lock()

        def create_embedding(self, text):
            with self._lock:
                hit = self._cache.get(text)
            if hit is not None:
                return hit
            emb = self.inner.create_embedding(text)
            with self._lock:
                self._cache[text] = emb
            return emb

    return CachedEmbeddingModel(model)


def make_embedding_model(ref):
    """Build an embedding model from a ``ModelRef`` (lazy upstream imports)."""
    if ref.provider == "mock":
        model = make_mock_embedding_model()
    elif ref.provider == "sbert":
        from raptor.EmbeddingModels import SBertEmbeddingModel

        model = SBertEmbeddingModel(model_name=ref.model)
    elif ref.provider == "openai":
        from raptor.EmbeddingModels import OpenAIEmbeddingModel

        model = OpenAIEmbeddingModel(model=ref.model)
    else:
        raise NotImplementedError(f"embedding provider '{ref.provider}' is not supported")
    return make_cached(model)

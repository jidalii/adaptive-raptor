"""The vendored upstream RAPTOR package imports and exposes its original API."""

import pytest

pytestmark = pytest.mark.heavy


def test_upstream_raptor_imports():
    raptor = pytest.importorskip("raptor")
    for name in [
        "RetrievalAugmentation",
        "RetrievalAugmentationConfig",
        "ClusterTreeBuilder",
        "ClusterTreeConfig",
        "TreeRetriever",
        "TreeRetrieverConfig",
        "SBertEmbeddingModel",
        "BaseSummarizationModel",
        "Node",
        "Tree",
    ]:
        assert hasattr(raptor, name), name


def test_upstream_split_text_runs_offline():
    import tiktoken

    from raptor.utils import split_text

    text = "RAPTOR builds a tree. " * 60
    chunks = split_text(text, tiktoken.get_encoding("cl100k_base"), max_tokens=50)
    assert len(chunks) > 1


def test_fixed_chunker_wraps_upstream():
    from adaptive_raptor.chunking import build_chunker
    from adaptive_raptor.config.schema import FixedChunking
    from adaptive_raptor.data import Document

    chunker = build_chunker(FixedChunking(name="f100", chunk_size=100))
    chunks = chunker.chunk(Document(doc_id="d", text="A sentence here. " * 100))
    assert chunks and all(c.n_tokens <= 100 for c in chunks)

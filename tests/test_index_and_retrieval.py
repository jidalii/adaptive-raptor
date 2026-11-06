"""RAPTOR integration: index build/save/load/reuse and the retrieval interface contract.

Uses mock embeddings and summaries (offline), but the real upstream clustering
(UMAP + GMM), tree construction and TreeRetriever.
"""

import json

import pytest

from adaptive_raptor.chunking import build_chunker
from adaptive_raptor.config.schema import (
    ConfigRef,
    FixedChunking,
    ModelRef,
    RaptorSection,
    Strategy,
)
from adaptive_raptor.data import load_dataset
from adaptive_raptor.indexing import IndexBuilder, IndexCacheError
from adaptive_raptor.models.embedding import make_embedding_model
from adaptive_raptor.models.summarization import make_mock_summarizer

pytestmark = pytest.mark.heavy

VARIANT = FixedChunking(name="f100", chunk_size=100)


def _raptor_cfg(tmp_path, **kw):
    return RaptorSection(
        embedding=ModelRef(provider="mock", model="hash"),
        summarization=ModelRef(provider="mock", model="first-words"),
        index_dir=tmp_path / "idx",
        **kw,
    )


@pytest.fixture(scope="module")
def doc():
    from adaptive_raptor.config.schema import DatasetSection

    return load_dataset(DatasetSection(name="qasper_tiny", split="dev")).documents[0]


@pytest.fixture(scope="module")
def chunks(doc):
    return build_chunker(VARIANT).chunk(doc)


def _builder(cfg, calls=None):
    def factory():
        if calls is not None:
            calls.append(1)
        return make_mock_summarizer()

    return IndexBuilder(
        cfg,
        dataset="qasper_tiny",
        dataset_version="fixture",
        embedding_model=make_embedding_model(cfg.embedding),
        summarizer_factory=factory,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory, chunks):
    tmp = tmp_path_factory.mktemp("index")
    cfg = _raptor_cfg(tmp)
    handle = _builder(cfg).load_or_build("fixture.0001", VARIANT, chunks, seed=224)
    return cfg, handle


def test_fixed_chunks_match_upstream_split_text(doc, chunks):
    import tiktoken

    from raptor.utils import split_text

    upstream = split_text(doc.text, tiktoken.get_encoding("cl100k_base"), 100)
    assert [c.text for c in chunks] == upstream
    assert all(c.n_tokens <= 100 for c in chunks)
    starts = [c.start for c in chunks]
    assert all(s is not None for s in starts) and starts == sorted(starts)


def test_build_creates_summary_layer(built, chunks):
    _, h = built
    assert not h.cache_hit
    assert h.build_stats["nodes_per_layer"][0] == len(chunks)
    assert h.build_stats["num_layers"] >= 1 and len(h.tree.layer_to_nodes[1]) >= 1
    assert h.path.is_file() and h.path.name.endswith(f"__{h.key_hash[:16]}.pkl")
    assert h.path.with_name(h.path.name[:-4] + ".json").is_file()


def test_reuse_from_cache_without_rebuilding(built, chunks):
    cfg, h = built
    calls = []
    again = _builder(cfg, calls).load_or_build("fixture.0001", VARIANT, chunks, seed=224)
    assert again.cache_hit and again.key_hash == h.key_hash and calls == []
    assert {i: n.text for i, n in again.tree.all_nodes.items()} == {
        i: n.text for i, n in h.tree.all_nodes.items()
    }


def test_changed_parameters_never_reuse(built, chunks, tmp_path):
    cfg, h = built
    other = cfg.model_copy(update={"summarization_length": 50})
    h2 = _builder(other).load_or_build("fixture.0001", VARIANT, chunks, seed=224)
    assert not h2.cache_hit and h2.key_hash != h.key_hash and h2.path != h.path
    h3 = _builder(cfg).load_or_build("fixture.0001", VARIANT, chunks[:-1], seed=224)
    assert not h3.cache_hit and h3.key_hash != h.key_hash  # different leaves


def test_tampered_cache_is_rejected(built, chunks):
    cfg, h = built
    meta_path = h.path.with_name(h.path.name[:-4] + ".json")
    meta = json.loads(meta_path.read_text())
    meta["key"]["seed"] = 999
    meta_path.write_text(json.dumps(meta))
    try:
        with pytest.raises(IndexCacheError, match="mismatch"):
            _builder(cfg).load_or_build("fixture.0001", VARIANT, chunks, seed=224)
    finally:
        meta["key"]["seed"] = 224
        meta_path.write_text(json.dumps(meta))


def test_same_seed_is_reproducible(chunks, tmp_path):
    cfg = _raptor_cfg(tmp_path, cache_indexes=False)
    t1 = _builder(cfg).build("fixture.0001", VARIANT, chunks, 7)[0]
    t2 = _builder(cfg).build("fixture.0001", VARIANT, chunks, 7)[0]

    def shape(t):
        return {k: sorted(sorted(n.children) for n in v) for k, v in t.layer_to_nodes.items()}

    assert shape(t1) == shape(t2)
    assert [n.text for n in t1.leaf_nodes.values()] == [n.text for n in t2.leaf_nodes.values()]


@pytest.fixture(scope="module")
def executor(built):
    from adaptive_raptor.config.schema import RetrievalSection
    from adaptive_raptor.retrieval import RetrievalExecutor

    cfg, _ = built
    return RetrievalExecutor(
        RetrievalSection(include_raptor_original=True), make_embedding_model(cfg.embedding)
    )


@pytest.mark.parametrize("strategy", list(Strategy))
@pytest.mark.parametrize("budget", [150, 400, 2000])
def test_strategy_levels_and_budget(built, executor, strategy, budget):
    _, h = built
    r = executor.retrieve(
        "How many sentences are in the corpus?", h, ConfigRef(strategy=strategy, budget=budget)
    )
    assert r.config_id == f"{strategy.value}@{budget}"
    assert r.tokens_used <= budget
    assert len(r.node_ids) == len(r.levels) == len(r.texts)
    if strategy is Strategy.LEAF_ONLY:
        assert set(r.levels) <= {0} and r.node_ids
    if strategy is Strategy.SUMMARY_ONLY:
        assert r.levels and min(r.levels) >= 1


def test_collapsed_matches_upstream(built, executor):
    from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig

    _, h = built
    q = "What learning rate is used with Adam?"
    ours = executor.retrieve(q, h, ConfigRef(strategy=Strategy.COLLAPSED, budget=1000))
    up = TreeRetriever(
        TreeRetrieverConfig(
            context_embedding_model="EMB", embedding_model=executor.embedding_model
        ),
        h.tree,
    )
    nodes, _ = up.retrieve_information_collapse_tree(
        q, top_k=len(h.tree.all_nodes), max_tokens=1000
    )
    assert ours.node_ids == [n.index for n in nodes]


def test_raptor_original_retrieval(built, executor):
    _, h = built
    r = executor.retrieve_original("Do they use crowdworkers?", h)
    assert r.config_id == "raptor_original" and r.node_ids and r.tokens_used <= 2000


_FINGERPRINT = """
import hashlib
from adaptive_raptor.chunking import build_chunker
from adaptive_raptor.config.schema import DatasetSection, FixedChunking, ModelRef, RaptorSection
from adaptive_raptor.data import load_dataset
from adaptive_raptor.indexing import IndexBuilder
from adaptive_raptor.models.embedding import make_embedding_model
from adaptive_raptor.models.summarization import make_mock_summarizer
doc = load_dataset(DatasetSection(name="qasper_tiny", split="dev")).documents[0]
v = FixedChunking(name="f100")
cfg = RaptorSection(embedding=ModelRef(provider="mock", model="h"),
                    summarization=ModelRef(provider="mock", model="m"), cache_indexes=False)
b = IndexBuilder(cfg, dataset="t", dataset_version="t",
                 embedding_model=make_embedding_model(cfg.embedding),
                 summarizer_factory=make_mock_summarizer)
tree, _ = b.build(doc.doc_id, v, build_chunker(v).chunk(doc), 224)
nodes = sorted((k, sorted(n.children), n.text) for k, n in tree.all_nodes.items())
print(hashlib.sha256(repr(nodes).encode()).hexdigest())
"""


def test_builds_are_reproducible_across_processes():
    """Upstream UMAP is unseeded; deterministic_umap must make separate processes agree."""
    import subprocess
    import sys

    from tests.conftest import ROOT

    outs = {
        subprocess.run(
            [sys.executable, "-c", _FINGERPRINT],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(outs) == 1

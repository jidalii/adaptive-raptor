"""Build RAPTOR trees from pre-computed leaf chunks, with caching and seed control.

Upstream reuse (``src/raptor`` is unchanged):
- ``ClusterTreeBuilder.construct_tree``: UMAP + GMM clustering and recursive summarization.
- ``TreeBuilder.create_node`` / ``multithreaded_create_leaf_nodes``: leaf embedding.
``build_from_chunks`` is upstream ``build_from_text`` without its internal ``split_text``
call, so any chunker can supply the leaves. Deliberate differences, all for
reproducibility (the upstream source files are not edited):

1. leaf nodes are re-sorted by index after multithreaded creation (upstream keeps thread
   completion order);
2. numpy/random are seeded before each build;
3. with ``raptor.deterministic_umap`` (default), upstream's ``umap.UMAP(...)`` calls receive
   ``random_state=seed`` for the duration of the build. Upstream leaves it unset, so UMAP
   runs its parallel unseeded path and trees differ between processes even with seeds.

Upstream ``construct_tree`` mutates ``builder.num_layers``, so a fresh builder is created
for every document.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import logging
import random
import threading
import time
import types
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_raptor.chunking import Chunk
from adaptive_raptor.config.schema import ChunkingVariant, RaptorSection
from adaptive_raptor.indexing.store import (
    EMB_KEY,
    index_key,
    index_paths,
    load_index,
    save_index,
)

log = logging.getLogger(__name__)
_UMAP_PATCH_LOCK = threading.Lock()


@contextlib.contextmanager
def seeded_umap(seed: int, enabled: bool):
    """Make upstream ``raptor.cluster_utils`` call ``umap.UMAP(..., random_state=seed)``."""
    if not enabled:
        yield
        return
    import umap as umap_module

    from raptor import cluster_utils

    with _UMAP_PATCH_LOCK:
        original = cluster_utils.umap
        cluster_utils.umap = types.SimpleNamespace(
            UMAP=functools.partial(umap_module.UMAP, random_state=seed)
        )
        try:
            yield
        finally:
            cluster_utils.umap = original


@dataclass
class IndexHandle:
    doc_id: str
    variant: str
    seed: int
    tree: Any  # raptor.tree_structures.Tree
    key_hash: str
    path: Path | None = None
    cache_hit: bool = False
    build_stats: dict[str, Any] = field(default_factory=dict)


def raptor_index_params(cfg: RaptorSection) -> dict[str, Any]:
    """The RAPTOR parameters that affect a built tree (part of the cache key)."""
    return {
        "embedding": cfg.embedding.model_dump(mode="json"),
        "summarization": cfg.summarization.model_dump(mode="json"),
        "summarization_length": cfg.summarization_length,
        "num_layers": cfg.num_layers,
        "clustering": cfg.clustering.model_dump(mode="json"),
        "tokenizer": cfg.tokenizer,
        "deterministic_umap": cfg.deterministic_umap,
    }


def make_tree_builder(cfg: RaptorSection, embedding_model, summarization_model):
    import tiktoken

    from raptor import ClusterTreeBuilder, ClusterTreeConfig
    from raptor.tree_structures import Tree

    class ChunkTreeBuilder(ClusterTreeBuilder):
        def build_from_chunks(self, chunks: list[str], use_multithreading: bool = True):
            if use_multithreading:
                leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
                leaf_nodes = dict(sorted(leaf_nodes.items()))  # deterministic order
            else:
                leaf_nodes = {}
                for index, text in enumerate(chunks):
                    __, node = self.create_node(index, text)
                    leaf_nodes[index] = node
            layer_to_nodes = {0: list(leaf_nodes.values())}
            all_nodes = copy.deepcopy(leaf_nodes)
            root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)
            return Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

    config = ClusterTreeConfig(
        reduction_dimension=cfg.clustering.reduction_dimension,
        clustering_params={
            "threshold": cfg.clustering.threshold,
            "max_length_in_cluster": cfg.clustering.max_length_in_cluster,
        },
        tokenizer=tiktoken.get_encoding(cfg.tokenizer),
        num_layers=cfg.num_layers,
        summarization_length=cfg.summarization_length,
        summarization_model=summarization_model,
        embedding_models={EMB_KEY: embedding_model},
        cluster_embedding_model=EMB_KEY,
    )
    return ChunkTreeBuilder(config)


def tree_stats(tree, tokenizer) -> dict[str, Any]:
    layers = {int(k): len(v) for k, v in tree.layer_to_nodes.items()}
    summary_nodes = [n for k, v in tree.layer_to_nodes.items() if int(k) > 0 for n in v]
    return {
        "num_layers": tree.num_layers,
        "nodes_per_layer": layers,
        "num_nodes": len(tree.all_nodes),
        "leaf_tokens": sum(len(tokenizer.encode(n.text)) for n in tree.layer_to_nodes[0]),
        "summary_tokens": sum(len(tokenizer.encode(n.text)) for n in summary_nodes),
    }


class IndexBuilder:
    """Builds, saves and reuses RAPTOR indexes. One instance per experiment run."""

    def __init__(
        self,
        cfg: RaptorSection,
        *,
        dataset: str,
        dataset_version: str,
        embedding_model,
        summarizer_factory: Callable[[], Any],
        use_cache: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.dataset_version = dataset_version
        self.embedding_model = embedding_model
        self._summarizer_factory = summarizer_factory
        self._summarizer = None
        self.use_cache = cfg.cache_indexes if use_cache is None else use_cache

    @property
    def summarizer(self):
        if self._summarizer is None:  # created only when a build is actually needed
            self._summarizer = self._summarizer_factory()
        return self._summarizer

    def build(self, doc_id: str, variant: ChunkingVariant, chunks: list[Chunk], seed: int):
        import tiktoken

        if not chunks:
            raise ValueError(f"no chunks for document {doc_id}")
        random.seed(seed)
        np.random.seed(seed)
        builder = make_tree_builder(self.cfg, self.embedding_model, self.summarizer)
        t0 = time.perf_counter()
        with seeded_umap(seed, self.cfg.deterministic_umap), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs.*overridden.*")
            tree = builder.build_from_chunks([c.text for c in chunks])
        stats = tree_stats(tree, tiktoken.get_encoding(self.cfg.tokenizer))
        stats["build_time_s"] = round(time.perf_counter() - t0, 3)
        stats["num_chunks"] = len(chunks)
        return tree, stats

    def load_or_build(
        self, doc_id: str, variant: ChunkingVariant, chunks: list[Chunk], seed: int
    ) -> IndexHandle:
        from adaptive_raptor.cache import stable_hash

        key = index_key(
            dataset=self.dataset,
            dataset_version=self.dataset_version,
            doc_id=doc_id,
            chunking=variant.model_dump(mode="json", exclude={"name"}),
            chunk_texts=[c.text for c in chunks],
            raptor=raptor_index_params(self.cfg),
            seed=seed,
        )
        pkl, meta = index_paths(
            self.cfg.index_dir, self.dataset, self.dataset_version, variant.name, seed, doc_id, key
        )
        key_hash = stable_hash(key)
        if self.use_cache:
            cached = load_index(key, pkl, meta)
            if cached is not None:
                tree, m = cached
                return IndexHandle(
                    doc_id, variant.name, seed, tree, key_hash, pkl, True, m.get("build_stats", {})
                )
        tree, stats = self.build(doc_id, variant, chunks, seed)
        save_index(tree, key, stats, pkl, meta)
        log.info(
            "built index %s (%s, seed %d): %s", doc_id, variant.name, seed, stats["nodes_per_layer"]
        )
        return IndexHandle(doc_id, variant.name, seed, tree, key_hash, pkl, False, stats)

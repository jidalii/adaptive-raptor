"""On-disk RAPTOR index cache.

An index file is reused only if its stored key equals the requested key exactly. The key
covers everything that changes the tree: dataset/version, document, chunker config and the
actual chunk texts, embedding/summarization models, clustering parameters, tokenizer,
seed, the upstream RAPTOR commit, and this module's format version.

Layout: <index_dir>/<dataset>/<version>/<variant>/seed<seed>/<doc_id>__<keyhash>.{pkl,json}
"""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_raptor.cache import stable_hash

INDEX_FORMAT_VERSION = 2  # 2: summaries made with thinking disabled (T03 fix)
UPSTREAM_RAPTOR_COMMIT = "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767"
EMB_KEY = "EMB"  # embedding key used for every node and for retrieval


class IndexCacheError(RuntimeError):
    """A cached index exists but cannot be trusted for this request."""


def index_key(
    *,
    dataset: str,
    dataset_version: str,
    doc_id: str,
    chunking: dict[str, Any],
    chunk_texts: list[str],
    raptor: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "format": INDEX_FORMAT_VERSION,
        "upstream_raptor": UPSTREAM_RAPTOR_COMMIT,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "doc_id": doc_id,
        "chunking": chunking,
        "chunks_sha256": stable_hash(chunk_texts),
        "raptor": raptor,
        "seed": seed,
    }


def index_paths(
    index_dir: Path, dataset: str, version: str, variant: str, seed: int, doc_id: str, key: dict
) -> tuple[Path, Path]:
    h = stable_hash(key)[:16]
    safe_doc = doc_id.replace("/", "_")
    folder = Path(index_dir) / dataset / version / variant / f"seed{seed}"
    # Plain string names: doc ids contain dots (e.g. "1911.10742"), so Path.with_suffix
    # would silently drop the key hash.
    return folder / f"{safe_doc}__{h}.pkl", folder / f"{safe_doc}__{h}.json"


def save_index(tree, key: dict, stats: dict, pkl_path: Path, meta_path: Path) -> None:
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pkl_path.parent / (pkl_path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(tree, f)  # same format as upstream RetrievalAugmentation.save
    tmp.replace(pkl_path)
    meta = {
        "key": key,
        "key_hash": stable_hash(key),
        "build_stats": stats,
        "created_at": datetime.now(UTC).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")


def load_index(key: dict, pkl_path: Path, meta_path: Path):
    """Return (tree, meta) or None if not cached. Raises IndexCacheError if untrustworthy."""
    if not pkl_path.exists() and not meta_path.exists():
        return None
    if not (pkl_path.is_file() and meta_path.is_file()):
        raise IndexCacheError(f"incomplete index cache entry {pkl_path.name}; delete it to rebuild")
    meta = json.loads(meta_path.read_text())
    if meta.get("key") != json.loads(json.dumps(key, default=str)):
        raise IndexCacheError(f"index key mismatch at {meta_path}; refusing to reuse")
    from raptor.tree_structures import Tree

    with pkl_path.open("rb") as f:
        tree = pickle.load(f)
    if not isinstance(tree, Tree):
        raise IndexCacheError(f"{pkl_path} does not contain a raptor Tree")
    if any(EMB_KEY not in n.embeddings for n in tree.all_nodes.values()):
        raise IndexCacheError(f"{pkl_path} has nodes without '{EMB_KEY}' embeddings")
    return tree, meta

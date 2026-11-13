"""Adaptive-k (Taguchi, Maekawa, Bhutani 2025) as a per-question budget rule, offline.

    uv run python scripts/adaptive_k_offline.py [--out DIR]

No API calls, no tuning. Pre-registered in NOTES.md (2026-11-11, Adaptive-k).

Rule, as in the official code (megagonlabs/adaptive-k-retrieval, retriever.py
``find_threshold_largest_gap`` + ``adaptive_retrieve``; run_qa.sh flags
``--retrieval_strategy largest_gap --ignore_extreme_tail 0.1 --retrieve_more 5``):
  s = cosine similarities of the question to all N leaf chunks, sorted descending
  g = s[i+1] - s[i] (i = 0..N-2);  tail = int((N-1) * 0.1)
  i* = argmin(g[: len(g) - tail])  (the largest drop; whole g if tail == 0)
  keep the top i* + 1 + 5 chunks.
Our embeddings are the RAPTOR SBERT ones (multi-qa-mpnet-base-cos-v1), not the paper's
bge-large; the ranking is the one leaf_only already uses (reproduced and checked against
the stored leaf@2000 node lists).

Answers exist only for leaf_only@400/1000/2000 (prefixes of the same ranking), so the kept
chunk count is mapped to a budget:
  primary "ceil": the smallest budget whose stored prefix contains all kept chunks
                  (2000 if none; such questions are counted as capped);
  secondary "nearest": the budget nearest to the kept chunks' token count.
Score: QASPER F1 and generation tokens of the mapped config; compared with the leaf fixed
mixture at the same mean cost (hash mix of neighbouring budgets) and with fixed leaf@1000;
paired document bootstrap (2000x, 95%). Reported on dev (primary) and train.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import LEAF, Split, compare, fixed, load_tree, system, vs_matched  # noqa: E402

BUDGETS = [400, 1000, 2000]
TAIL, MORE = 0.1, 5


def adaptive_k(sorted_sims: np.ndarray) -> int:
    """Number of chunks kept by the official largest-gap rule (ties: first index)."""
    n = len(sorted_sims)
    if n <= 1:
        return n
    gaps = np.diff(sorted_sims)  # <= 0
    tail = int((n - 1) * TAIL)
    seg = gaps[: len(gaps) - tail] if tail > 0 else gaps
    i = int(np.argmin(seg))
    return min(n, i + 1 + MORE)


def run_split(split: str) -> dict:
    from raptor.utils import distances_from_embeddings, indices_of_nearest_neighbors_from_distances

    run = Path(f"outputs/labels_{split}/20261109T195548Z")
    qfile = Path(f"data/processed/qasper/v0.3/{split}/questions.jsonl")
    ecache = Path(f"outputs/diagnostics/utility_router/cache/utility_features_{split}.npz")
    for p in (run / "labels.jsonl", run / "traces.jsonl", qfile, ecache):
        if not p.is_file():
            raise FileNotFoundError(p)
    s = Split(run, None, qfile, configs=LEAF)
    qemb = np.load(ecache, allow_pickle=True)["feats"].item()["emb"]
    enc = tiktoken.get_encoding("cl100k_base").encode
    bydoc = defaultdict(list)
    for q in s.qs:
        bydoc[s.doc[q]].append(q)
    ceil, nearest, rows = {}, {}, []
    mismatches = 0
    for doc, qs in sorted(bydoc.items()):
        tree = load_tree(doc, s.index_hash[doc])
        leaves = tree.layer_to_nodes[0]
        ids = [n.index for n in leaves]
        tok = {n.index: len(enc(n.text)) for n in leaves}
        embs = [n.embeddings["EMB"] for n in leaves]
        for q in qs:
            dist = np.asarray(distances_from_embeddings(qemb[q], embs))
            order = [int(i) for i in indices_of_nearest_neighbors_from_distances(dist)]
            ranked = [ids[i] for i in order]
            stored = s.nodes[q]["leaf_only@2000"]
            if ranked[: len(stored)] != stored:
                mismatches += 1
            sims = 1.0 - dist[order]
            k = adaptive_k(sims)
            ktok = sum(tok[i] for i in ranked[:k])
            lens = {b: len(s.nodes[q][f"leaf_only@{b}"]) for b in BUDGETS}
            fit = [b for b in BUDGETS if lens[b] >= k]
            b_ceil = fit[0] if fit else 2000
            b_near = min(BUDGETS, key=lambda b: (abs(b - ktok), b))
            ceil[q], nearest[q] = f"leaf_only@{b_ceil}", f"leaf_only@{b_near}"
            rows.append(
                {
                    "question_id": q,
                    "doc_id": doc,
                    "n_leaves": len(ids),
                    "k": k,
                    "k_tokens": ktok,
                    "ceil": b_ceil,
                    "nearest": b_near,
                    "capped": not fit,
                }
            )
    out = {
        "n": len(s.qs),
        "ranking_mismatches_vs_stored_leaf2000": mismatches,
        "k": {
            "median": float(np.median([r["k"] for r in rows])),
            "p10_p90": [float(np.percentile([r["k"] for r in rows], p)) for p in (10, 90)],
            "share_of_leaves_median": float(np.median([r["k"] / r["n_leaves"] for r in rows])),
        },
        "k_tokens_median": float(np.median([r["k_tokens"] for r in rows])),
        "capped_share": round(float(np.mean([r["capped"] for r in rows])), 4),
        "fixed": {c: system(s, fixed(s, c), name=c) for c in LEAF},
        "mappings": {},
        "rows": rows,
    }
    for name, choice in (("ceil", ceil), ("nearest", nearest)):
        row = system(s, choice, name=f"adaptive_k_{name}")
        row["choice_mix"] = dict(Counter(choice.values()))
        vs_matched(s, row, choice)
        row["vs_leaf1000"] = compare(s, choice, fixed(s, "leaf_only@1000"))
        out["mappings"][name] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or Path("outputs/diagnostics/adaptive_k") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    rep = {
        "note": "offline, no API; official Adaptive-k largest_gap rule (tail 0.1, +5); "
        "QASPER leaf ranking with RAPTOR SBERT; F1 from stored labels; doc bootstrap 95%",
        "splits": {},
    }
    for split in ("dev", "train"):
        rep["splits"][split] = run_split(split)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    for split, r in rep["splits"].items():
        print(
            f"== {split}: n={r['n']} "
            f"ranking mismatches={r['ranking_mismatches_vs_stored_leaf2000']}"
        )
        print(
            f"   k median {r['k']['median']} (p10-p90 {r['k']['p10_p90']}), share of leaves "
            f"{r['k']['share_of_leaves_median']:.2f}, tokens median {r['k_tokens_median']}, "
            f"capped {r['capped_share']}"
        )
        for c, v in r["fixed"].items():
            print(f"   fixed {c}: F1 {v['f1']} tokens {v['total_tokens']}")
        for name, v in r["mappings"].items():
            print(
                f"   adaptive-k/{name}: F1 {v['f1']} tokens {v['total_tokens']} "
                f"mix {v['choice_mix']}"
            )
            print(
                f"      matched leaf mix F1 {v['matched_fixed']['f1']}; "
                f"delta {v['vs_matched_fixed']}; "
                f"vs leaf@1000 {v['vs_leaf1000']}"
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

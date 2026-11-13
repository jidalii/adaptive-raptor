"""P0b: offline, train-selected tree navigation. Run --stage train, then --stage dev.
No model/API calls, downloads, index writes, test data, or pipeline changes.
Gold metrics are imported verbatim from compression_feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compression_feasibility import chunk_truncate, ctoks, evaluate, load_gold
from offline_diagnostics import LEAF, Split, load_tree

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "version": 1,
    "budgets": [1000, 2000],
    "beams": [1, 2, 4],
    "leaf_budgets": [0, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400],
    "bootstrap": 4000,
    "confidence": 0.975,
    "seed": 224,
    "selection": "per budget, max train matched-cost hl_recall delta; ties smaller beam",
    "gate": "primary 1000 only: delta > 0 and 97.5% CI lower > 0; secondary cannot rescue",
    "beam_scope": "summary layers only; rank reached leaves then pack whole nodes",
    "packing": "normalized text joined by two newlines; first nonfitting node stops",
    "comparator": "split-specific best fixed randomized mix; refit in each bootstrap",
    "metrics": (
        "compression_feasibility.evaluate; hl recall is lexical coverage, not exact evidence"
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def no_network():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")

    def denied(*args, **kwargs):
        raise RuntimeError("P0b forbids network connections")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied


def context(ids, texts):
    return "\n\n".join(" ".join(texts[i].splitlines()) for i in ids)


def pack(ranked, texts, budget, encode):
    chosen = []
    for i in ranked:
        if len(encode(context([*chosen, i], texts))) > budget:
            break
        chosen.append(i)
    return chosen


def navigate(tree, scores, beam):
    """Expand selected summary nodes; do not limit the final leaf frontier by beam."""
    levels = {n.index: int(k) for k, ns in tree.layer_to_nodes.items() for n in ns}
    highest = max(levels.values())
    frontier = [n.index for n in tree.layer_to_nodes[highest]]
    path = []
    for level in range(highest, 0, -1):
        if any(levels[i] != level for i in frontier):
            raise ValueError("non-adjacent or invalid tree edge")
        selected = sorted(set(frontier), key=lambda i: (scores[i], i))[:beam]
        children = sorted({c for i in selected for c in tree.all_nodes[i].children})
        if any(c not in levels or levels[c] != level - 1 for c in children):
            raise ValueError("invalid child layer")
        path.append(
            {
                "level": level,
                "candidates": len(frontier),
                "selected": selected,
                "edges": [[i, c] for i in selected for c in sorted(tree.all_nodes[i].children)],
            }
        )
        frontier = children
        if not frontier:
            break
    return sorted(set(frontier), key=lambda i: (scores[i], i)), path


def upper_hull(cost, quality):
    """Nondecreasing upper concave envelope, with dominated fixed points removed."""
    hull = []
    for i in sorted(range(len(cost)), key=lambda i: (cost[i], -quality[i], i)):
        x, y = float(cost[i]), float(quality[i])
        if hull and (x == hull[-1][0] or y <= hull[-1][1]):
            continue
        while len(hull) > 1:
            x0, y0, _ = hull[-2]
            x1, y1, _ = hull[-1]
            if (y1 - y0) * (x - x1) <= (y - y1) * (x1 - x0):
                hull.pop()
            else:
                break
        hull.append((x, y, i))
    return hull


def match(cost, quality, target):
    """Expected fixed mix at budget <= target (saturated frontier may cost less)."""
    hull = upper_hull(cost, quality)
    if target < hull[0][0] - 1e-9 or target > max(cost) + 1e-9:
        raise ValueError("target outside baseline cost support")
    if target >= hull[-1][0]:
        i = hull[-1][2]
        return i, i, 0.0
    for a, b in zip(hull, hull[1:], strict=False):
        if a[0] <= target <= b[0]:
            return a[2], b[2], (target - a[0]) / (b[0] - a[0])
    raise AssertionError("unmatched cost")


def arrays(rows, method):
    return np.array(
        [
            [
                r["methods"][method]["context_tokens"],
                r["methods"][method]["hl_recall"]
                if r["methods"][method]["hl_recall"] is not None
                else np.nan,
            ]
            for r in rows
        ],
        dtype=float,
    )


def compare(rows, candidate, bootstrap=0):
    names = [f"leaf@{b}" for b in PROTOCOL["leaf_budgets"]]
    base = np.stack([arrays(rows, name) for name in names], axis=1)
    cand = arrays(rows, candidate)
    mask = np.isfinite(cand[:, 1])
    if not mask.any():
        raise ValueError("no questions with highlighted evidence")
    cost = base[:, :, 0].mean(0)
    qual = base[mask, :, 1].mean(0)
    target = cand[:, 0].mean()
    i, j, w = match(cost, qual, target)
    expected = (1 - w) * base[:, i, 1] + w * base[:, j, 1]
    d = cand[mask, 1] - expected[mask]
    out = {
        "candidate": candidate,
        "actual_tokens": float(target),
        "hl_recall": float(cand[mask, 1].mean()),
        "mix": {
            "lo": names[i],
            "hi": names[j],
            "weight_hi": w,
            "actual_tokens": float((1 - w) * cost[i] + w * cost[j]),
            "hl_recall": float((1 - w) * qual[i] + w * qual[j]),
        },
        "delta": float(d.mean()),
        "n_evidence": int(mask.sum()),
        "improved": int((d > 1e-12).sum()),
        "worsened": int((d < -1e-12).sum()),
    }
    same = arrays(rows, f"leaf@{candidate.split('@')[-1]}")
    out["same_nominal_budget_delta"] = float((cand[mask, 1] - same[mask, 1]).mean())
    if bootstrap:
        docs = sorted({r["doc_id"] for r in rows})
        groups = np.array([docs.index(r["doc_id"]) for r in rows])
        # Refit the comparator hull and the cost match in every document resample.
        sums_c = np.stack([base[groups == g, :, 0].sum(0) for g in range(len(docs))])
        sums_q = np.stack([np.nansum(base[groups == g, :, 1], axis=0) for g in range(len(docs))])
        counts = np.bincount(groups, minlength=len(docs))
        evcounts = np.bincount(groups, weights=mask.astype(int), minlength=len(docs))
        cc = np.bincount(groups, weights=cand[:, 0], minlength=len(docs))
        cq = np.bincount(groups, weights=np.nan_to_num(cand[:, 1]), minlength=len(docs))
        rng = np.random.default_rng(PROTOCOL["seed"])
        values = []
        for _ in range(bootstrap):
            ix = rng.integers(0, len(docs), len(docs))
            n, ne = counts[ix].sum(), evcounts[ix].sum()
            if not ne:
                continue
            c, q = sums_c[ix].sum(0) / n, sums_q[ix].sum(0) / ne
            ii, jj, ww = match(c, q, cc[ix].sum() / n)
            values.append(cq[ix].sum() / ne - ((1 - ww) * q[ii] + ww * q[jj]))
        out["ci_97_5"] = np.quantile(values, [0.0125, 0.9875]).tolist()
        out["bootstrap_valid"] = len(values)
    return out


def run_split(split, out, selected=None):
    import tiktoken

    from raptor.utils import distances_from_embeddings, indices_of_nearest_neighbors_from_distances

    run = Path(f"outputs/labels_{split}/20261109T195548Z")
    qfile = Path(f"data/processed/qasper/v0.3/{split}/questions.jsonl")
    ecache = Path(f"outputs/diagnostics/utility_router/cache/utility_features_{split}.npz")
    required = [run / "labels.jsonl", run / "traces.jsonl", run / "indexes.jsonl", qfile, ecache]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)
    s = Split(run, None, qfile, configs=LEAF)
    gold = load_gold(qfile, set(s.qs))
    qemb = np.load(ecache, allow_pickle=True)["feats"].item()["emb"]
    if not set(s.qs) <= qemb.keys() or not set(s.qs) <= gold.keys():
        raise ValueError("missing query embedding or gold record")
    encode = tiktoken.get_encoding("cl100k_base").encode
    bydoc = defaultdict(list)
    for q in s.qs:
        bydoc[s.doc[q]].append(q)
    audit, rows, hashes = [], [], {str(p): digest(p) for p in required}
    checks = Counter()
    for di, (doc, qs) in enumerate(sorted(bydoc.items())):
        tree = load_tree(doc, s.index_hash[doc])
        ip = Path("indexes/qasper/v0.3/f100/seed224") / f"{doc}__{s.index_hash[doc][:16]}.pkl"
        hashes[str(ip)] = digest(ip)
        leaves = tree.layer_to_nodes[0]
        leaf_ids = [n.index for n in leaves]
        texts = {i: n.text for i, n in tree.all_nodes.items()}
        tok = {i: len(encode(texts[i])) for i in leaf_ids}
        parents = Counter(c for ns in tree.layer_to_nodes.values() for n in ns for c in n.children)
        audit.append(
            {
                "doc_id": doc,
                "max_level": max(int(k) for k in tree.layer_to_nodes),
                "nodes_per_layer": {str(k): len(v) for k, v in tree.layer_to_nodes.items()},
                "multi_parent_leaf_fraction": sum(parents[i] > 1 for i in leaf_ids) / len(leaves),
            }
        )
        for q in qs:
            # Baseline exactly follows upstream ordering and its old per-node packing.
            dist = distances_from_embeddings(qemb[q], [n.embeddings["EMB"] for n in leaves])
            ranked = [leaf_ids[int(i)] for i in indices_of_nearest_neighbors_from_distances(dist)]
            for b in (400, 1000, 2000):
                actual = chunk_truncate(ranked, tok, b)
                if actual != s.nodes[q][f"leaf_only@{b}"]:
                    dump(
                        out / "baseline_mismatch.json",
                        {
                            "split": split,
                            "question": q,
                            "budget": b,
                            "actual": actual,
                            "stored": s.nodes[q][f"leaf_only@{b}"],
                        },
                    )
                    raise ValueError(f"baseline mismatch {split}/{q}/{b}")
                checks[str(b)] += 1
            if any(
                s.nodes[q][f"leaf_only@{b}"]
                != s.nodes[q]["leaf_only@2000"][: len(s.nodes[q][f"leaf_only@{b}"])]
                for b in (400, 1000)
            ):
                raise ValueError("stored leaf retrieval is not a prefix")
            nodes = list(tree.all_nodes.values())
            ds = distances_from_embeddings(qemb[q], [n.embeddings["EMB"] for n in nodes])
            scores = {n.index: float(d) for n, d in zip(nodes, ds, strict=True)}
            row = {
                "question_id": q,
                "doc_id": doc,
                "max_level": audit[-1]["max_level"],
                "methods": {},
                "navigation": {},
            }

            def measure(ids, texts=texts, q=q):
                ctx = context(ids, texts)
                metrics = evaluate(ctx, q, s, gold, encode)
                evid = {t for ann in gold[q]["hl"] for sent in ann for t in ctoks(sent)}
                words = ctoks(ctx)
                metrics["evidence_word_precision_proxy"] = (
                    sum(t in evid for t in words) / len(words) if words and evid else None
                )
                return {**metrics, "node_ids": ids}

            row["all_leaf_hl_recall"] = measure(leaf_ids)["hl_recall"]
            for b in PROTOCOL["leaf_budgets"]:
                ids = pack(ranked, texts, b, encode)
                row["methods"][f"leaf@{b}"] = measure(ids)
            beams = PROTOCOL["beams"] if selected is None else sorted(set(selected.values()))
            for beam in beams:
                reached, path = navigate(tree, scores, beam)
                fallback = not reached
                if fallback:
                    reached = ranked
                row["navigation"][str(beam)] = {
                    "path": path,
                    "fallback": fallback,
                    "reached_leaf_ids": reached,
                    "reached_fraction": len(reached) / len(leaves),
                    "reachable_tokens": len(encode(context(reached, texts))),
                }
                for b in PROTOCOL["budgets"]:
                    if selected is not None and selected[str(b)] != beam:
                        continue
                    ids = pack(reached, texts, b, encode)
                    val = measure(ids)
                    val["budget_utilization"] = val["context_tokens"] / b
                    base_ids = row["methods"][f"leaf@{b}"]["node_ids"]
                    union = set(ids) | set(base_ids)
                    val["leaf_jaccard"] = (
                        len(set(ids) & set(base_ids)) / len(union) if union else 1.0
                    )
                    row["methods"][f"beam{beam}@{b}"] = val
            rows.append(row)
        if (di + 1) % 50 == 0:
            print(f"{split}: {di + 1}/{len(bydoc)} documents; {len(rows)} questions", flush=True)
    with (out / f"per_question_{split}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    dump(out / f"audit_{split}.json", audit)
    summary = {
        "n_questions": len(rows),
        "n_documents": len(audit),
        "baseline_exact_checks": dict(checks),
        "max_level_histogram": dict(Counter(str(a["max_level"]) for a in audit)),
        "multi_parent_leaf_fraction_median": float(
            np.median([a["multi_parent_leaf_fraction"] for a in audit])
        ),
        "n_without_highlighted_evidence": sum(r["all_leaf_hl_recall"] is None for r in rows),
        "all_leaf_hl_recall_mean": float(
            np.mean([r["all_leaf_hl_recall"] for r in rows if r["all_leaf_hl_recall"] is not None])
        ),
        "methods": {},
    }
    for name in rows[0]["methods"]:
        vals = [r["methods"][name] for r in rows]
        summary["methods"][name] = {}
        for k in vals[0]:
            if k == "node_ids":
                continue
            xs = [v[k] for v in vals if v[k] is not None]
            summary["methods"][name][k] = float(np.mean(xs)) if xs else None
            summary["methods"][name][f"n_{k}"] = len(xs)
    summary["beam_coverage"] = {}
    for beam in rows[0]["navigation"]:
        ns = [r["navigation"][beam] for r in rows]
        summary["beam_coverage"][beam] = {
            "question_weighted_median_reached_fraction": float(
                np.median([x["reached_fraction"] for x in ns])
            ),
            "fraction_questions_reaching_90pct": float(
                np.mean([x["reached_fraction"] >= 0.9 for x in ns])
            ),
            "fallback_questions": sum(x["fallback"] for x in ns),
        }
    dump(out / f"summary_{split}.json", summary)
    dump(out / f"inputs_{split}.json", hashes)
    return rows, summary


def source_hashes():
    paths = [
        Path(__file__),
        Path(__file__).with_name("compression_feasibility.py"),
        Path(__file__).with_name("offline_diagnostics.py"),
        ROOT / "src/raptor/utils.py",
        ROOT / "docs/OPTIMIZER_TREE_AWARE_MODIFICATION.md",
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["train", "dev"], required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    os.chdir(ROOT)
    no_network()
    if a.stage == "train":
        out = a.out or Path("outputs/diagnostics/tree_navigation") / datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        out.mkdir(parents=True, exist_ok=False)
        dump(out / "protocol.json", PROTOCOL)
        rows, summary = run_split("train", out)
        comparisons, selected = {}, {}
        for b in PROTOCOL["budgets"]:
            cc = [compare(rows, f"beam{beam}@{b}") for beam in PROTOCOL["beams"]]
            comparisons[str(b)] = cc
            best = max(
                zip(PROTOCOL["beams"], cc, strict=True), key=lambda x: (x[1]["delta"], -x[0])
            )
            selected[str(b)] = best[0]
        dump(out / "train_selection.json", comparisons)
        frozen = {
            "protocol": PROTOCOL,
            "selected_beam": selected,
            "sources": source_hashes(),
            "train_inputs_sha256": digest(out / "inputs_train.json"),
            "train_doc_ids": sorted({r["doc_id"] for r in rows}),
        }
        dump(out / "frozen.json", frozen)
        (out / "FROZEN_SHA256").write_text(digest(out / "frozen.json") + "\n")
        print(
            json.dumps(
                {
                    "out": str(out),
                    "selected": selected,
                    "audit": {k: v for k, v in summary.items() if k != "methods"},
                },
                indent=2,
            )
        )
    else:
        if a.out is None:
            p.error("dev requires --out pointing to the frozen train directory")
        out = a.out
        if (out / "DEV_STARTED").exists():
            raise RuntimeError("dev already started; no automatic repeat")
        frozen = json.loads((out / "frozen.json").read_text())
        if digest(out / "frozen.json") != (out / "FROZEN_SHA256").read_text().strip():
            raise ValueError("frozen hash mismatch")
        if frozen["protocol"] != PROTOCOL or frozen["sources"] != source_hashes():
            raise ValueError("source/protocol changed after freeze")
        if digest(out / "inputs_train.json") != frozen["train_inputs_sha256"]:
            raise ValueError("train input manifest changed")
        (out / "DEV_STARTED").write_text(datetime.now(UTC).isoformat() + "\n")
        rows, summary = run_split("dev", out, frozen["selected_beam"])
        if {r["doc_id"] for r in rows} & set(frozen["train_doc_ids"]):
            raise ValueError("train/dev document overlap")
        report = {
            "frozen_sha256": digest(out / "frozen.json"),
            "selected_beam": frozen["selected_beam"],
            "points": {},
        }
        for b in PROTOCOL["budgets"]:
            name = f"beam{frozen['selected_beam'][str(b)]}@{b}"
            report["points"][str(b)] = compare(rows, name, PROTOCOL["bootstrap"])
            report["points"][str(b)]["by_max_level"] = {
                str(level): compare([r for r in rows if r["max_level"] == level], name)
                for level in sorted({r["max_level"] for r in rows})
                if any(r["all_leaf_hl_recall"] is not None for r in rows if r["max_level"] == level)
            }
        primary = report["points"]["1000"]
        report["gate"] = {
            "passed": primary["delta"] > 0 and primary["ci_97_5"][0] > 0,
            "rule": PROTOCOL["gate"],
        }
        dump(out / "dev_report.json", report)
        lines = [
            "# P0b: tree navigation offline diagnostic",
            "",
            "No API/model calls; test untouched. "
            "Gold metric is lexical evidence coverage, not answer F1.",
            "",
            "| Budget cap | Train-selected beam | Actual tokens | HL recall | "
            "Fixed-mix recall | Delta (pp), 97.5% CI |",
            "|---|---|---|---|---|---|",
        ]
        for b, v in report["points"].items():
            lo, hi = v["ci_97_5"]
            lines.append(
                f"| {b} | {frozen['selected_beam'][b]} | {v['actual_tokens']:.1f} | "
                f"{v['hl_recall']:.4f} | {v['mix']['hl_recall']:.4f} | "
                f"{100 * v['delta']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] |"
            )
        lines += [
            "",
            f"Primary gate passed: {report['gate']['passed']}",
            "",
            "Comparator: split-specific upper hull of fixed leaf policies, mixed in expectation "
            "and recomputed inside each document bootstrap. Costs use every question; "
            "evidence recall excludes questions without usable highlights. "
            "Dev is an exploratory diagnostic on a previously analyzed split.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

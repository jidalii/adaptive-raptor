"""Summary-cap ablation analysis: control (cap 200) vs treatment (cap 512). Offline.

    uv run python scripts/analyze_summary_cap.py \
        --control outputs/labels_dev/<run> --treat outputs/summary_cap512_dev/<run> \
        --split dev [--out outputs/diagnostics/summary_cap]

Pre-registered questions (see docs/NEXT_EXPERIMENTS.md):
  Q1 summary quality   share of summaries at the cap / ending mid-sentence; best gold-evidence
                       recall of any summary node (paired by question)
  Q2 displacement      at the same budget cap: context tokens, nodes, summary share, leaf-only
                       evidence coverage (do longer summaries push leaf evidence out?)
  Q3 relative gain     F1 per config per arm; (strategy - leaf_only)@B per arm and the
                       difference-in-differences, paired document bootstrap
  Q4 strong baselines  treatment collapsed@B vs leaf_only@B; original RAPTOR vs leaf mix at
                       matched tokens
  Q5 cost              index-build USD per paper (summarizer usage) and inference USD per arm
Controls: leaf_only answers must be identical across arms (response cache) -> their delta must
be 0; any non-zero leaf delta is reported as generator/retrieval drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from adaptive_raptor.evaluation.research import paired_document_bootstrap

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import (  # noqa: E402
    ORIG,
    PRICE_IN,
    PRICE_OUT,
    STOP,
    Split,
    boot,
    evidence_hit,
    fixed_mix,
    load_tree,
    system,
)

BUDGETS = (1000, 2000)
STRATS = ("leaf_only", "collapsed", "summary_only")


def configs() -> list[str]:
    return [f"{s}@{b}" for s in STRATS for b in BUDGETS] + [ORIG]


def summary_stats(split: Split, cap: int) -> dict:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    toks, unfinished, layers = [], [], []
    trees = {}
    for doc, h in split.index_hash.items():
        tree = load_tree(doc, h)
        trees[doc] = tree
        layers.append(len(tree.layer_to_nodes))
        for k, nodes in tree.layer_to_nodes.items():
            if int(k) >= 1:
                for n in nodes:
                    toks.append(len(enc.encode(n.text)))
                    unfinished.append(not n.text.rstrip().endswith((".", "!", "?", '"', ")")))
    t = np.array(toks)
    return trees, {
        "cap": cap,
        "papers": len(trees),
        "summaries": int(len(t)),
        "summaries_per_paper": round(len(t) / max(1, len(trees)), 2),
        "mean_layers": round(float(np.mean(layers)), 2),
        "median_tokens": float(np.median(t)) if len(t) else None,
        "p90_tokens": float(np.quantile(t, 0.9)) if len(t) else None,
        "share_at_cap": round(float((t >= cap - 10).mean()), 3) if len(t) else None,
        "share_unfinished": round(float(np.mean(unfinished)), 3) if len(t) else None,
    }


def recall(evidence: list[str], text: str) -> float:
    have = set(w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP)
    best = 0.0
    for e in evidence:
        if e.startswith("FLOAT SELECTED"):
            continue
        toks = [w for w in re.findall(r"[a-z0-9]+", e.lower()) if w not in STOP]
        if toks:
            best = max(best, sum(t in have for t in toks) / len(toks))
    return best


def composition(split: Split, trees, q: str, c: str) -> dict:
    tree = trees[split.doc[q]]
    nodes = split.nodes[q].get(c) or []
    lv = split.levels[q].get(c) or []
    leaf_text = " ".join(
        tree.all_nodes[i].text for i, lvl in zip(nodes, lv, strict=False) if lvl == 0
    )
    all_text = " ".join(tree.all_nodes[i].text for i in nodes if i in tree.all_nodes)
    ev = split.evidence.get(q, [])
    return {
        "nodes": len(nodes),
        "summary_share": float(np.mean([x >= 1 for x in lv])) if lv else 0.0,
        "leaf_cov": evidence_hit(ev, leaf_text),
        "ctx_cov": evidence_hit(ev, all_text),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--control", type=Path, required=True, help="cap-200 label run (e.g. labels_dev)"
    )
    p.add_argument("--treat", type=Path, required=True, help="cap-512 label run")
    p.add_argument("--split", default="dev")
    p.add_argument("--control-cap", type=int, default=200)
    p.add_argument("--treat-cap", type=int, default=512)
    p.add_argument("--out", type=Path, default=Path("outputs/diagnostics/summary_cap"))
    a = p.parse_args()
    qfile = Path(f"data/processed/qasper/v0.3/{a.split}/questions.jsonl")
    cfgs = configs()
    ctl = Split(a.control, None, qfile, configs=cfgs)
    trt = Split(a.treat, None, qfile, configs=cfgs)
    qs = sorted(set(ctl.qs) & set(trt.qs))
    ctl.qs = trt.qs = qs
    docs = {trt.doc[q] for q in qs}
    ctl.index_hash = {d: h for d, h in ctl.index_hash.items() if d in docs}
    trt.index_hash = {d: h for d, h in trt.index_hash.items() if d in docs}
    print(f"{len(qs)} paired questions over {len(docs)} papers")
    rep: dict = {"n_questions": len(qs), "n_papers": len(docs)}

    # Q1 summary quality
    ctrees, rep["summaries_control"] = summary_stats(ctl, a.control_cap)
    ttrees, rep["summaries_treat"] = summary_stats(trt, a.treat_cap)

    def best_summary_recall(split, trees, q):
        tree = trees[split.doc[q]]
        texts = [n.text for k, v in tree.layer_to_nodes.items() if int(k) >= 1 for n in v]
        return max((recall(split.evidence.get(q, []), t) for t in texts), default=0.0)

    with_ev = [q for q in qs if any(not e.startswith("FLOAT") for e in ctl.evidence.get(q, []))]
    d = {
        q: best_summary_recall(trt, ttrees, q) - best_summary_recall(ctl, ctrees, q)
        for q in with_ev
    }
    rep["best_summary_evidence_recall"] = {
        "control": round(float(np.mean([best_summary_recall(ctl, ctrees, q) for q in with_ev])), 4),
        "treat": round(float(np.mean([best_summary_recall(trt, ttrees, q) for q in with_ev])), 4),
        "delta_ci": boot(d, ctl.doc),
        "n": len(with_ev),
    }

    # Q2 displacement + Q3 F1, per config
    per = {}
    for c in cfgs:
        row = {}
        for name, s, trees in (("control", ctl, ctrees), ("treat", trt, ttrees)):
            comp = [composition(s, trees, q, c) for q in qs]
            row[name] = {
                **system(s, {q: c for q in qs}, name=f"{name}:{c}"),
                "context_tokens": round(float(np.mean([s.ctx_tokens[q][c] for q in qs])), 1),
                "nodes": round(float(np.mean([x["nodes"] for x in comp])), 2),
                "summary_node_share": round(float(np.mean([x["summary_share"] for x in comp])), 3),
                "leaf_evidence_coverage": round(
                    float(np.mean([x["leaf_cov"] for x in comp if x["leaf_cov"] is not None])), 3
                ),
                "context_evidence_coverage": round(
                    float(np.mean([x["ctx_cov"] for x in comp if x["ctx_cov"] is not None])), 3
                ),
                "response_cache_hit_rate": round(
                    float(np.mean([s.gen_cache_hit[q][c] for q in qs])), 3
                ),
            }
        row["delta_f1_treat_minus_control"] = boot(
            {q: trt.f1[q][c] - ctl.f1[q][c] for q in qs}, ctl.doc
        )
        row["identical_answers"] = round(
            float(np.mean([trt.answer[q][c] == ctl.answer[q][c] for q in qs])), 3
        )
        per[c] = row
    rep["per_config"] = per
    leaf_drift = {c: per[c]["identical_answers"] for c in cfgs if c.startswith("leaf_only")}
    rep["control_check_leaf_identical_answers"] = leaf_drift

    # Q3 gap to leaf and difference-in-differences
    did = {}
    for b in BUDGETS:
        leaf = f"leaf_only@{b}"
        for st in ("collapsed", "summary_only"):
            c = f"{st}@{b}"
            g_c = {q: ctl.f1[q][c] - ctl.f1[q][leaf] for q in qs}
            g_t = {q: trt.f1[q][c] - trt.f1[q][leaf] for q in qs}
            did[c] = {
                "gap_to_leaf_control": boot(g_c, ctl.doc),
                "gap_to_leaf_treat": boot(g_t, ctl.doc),
                "difference_in_differences": boot({q: g_t[q] - g_c[q] for q in qs}, ctl.doc),
                # primary endpoints (collapsed@1000/2000): Bonferroni over two -> 97.5%
                "difference_in_differences_97_5": [
                    round(x, 4)
                    for x in paired_document_bootstrap(
                        {q: g_t[q] - g_c[q] for q in qs}, ctl.doc, 2000, 0.975
                    )
                ],
            }
    rep["gap_to_leaf"] = did

    # Q4 strong baselines at matched cost (leaf answers from the treatment run = cached)
    orig_t = {q: ORIG for q in qs}
    mix = fixed_mix(
        trt,
        float(np.mean([trt.cost(q, ORIG) for q in qs])),
        ladder=["leaf_only@1000", "leaf_only@2000"],
    )
    rep["original_treat_vs_matched_leaf_mix"] = {
        "original": system(trt, orig_t, name="original cap512"),
        "leaf_mix": system(trt, mix, name="leaf mix"),
        "delta": boot({q: trt.f1[q][ORIG] - trt.f1[q][mix[q]] for q in qs}, trt.doc),
    }

    # Q5 cost
    def summarizer(run: Path) -> dict:
        st = json.loads((run / "status.json").read_text())
        u = st.get("metrics", {}).get("summarizer_usage", {})
        n = max(1, st.get("metrics", {}).get("indexes_built", 0))
        usd = (u.get("input_tokens", 0) * PRICE_IN + u.get("output_tokens", 0) * PRICE_OUT) / 1e6
        return {**u, "indexes_built": n, "usd": round(usd, 4), "usd_per_paper": round(usd / n, 5)}

    rep["index_build_cost"] = {
        "control": summarizer(a.control),
        "treat": summarizer(a.treat),
        "note": "control run built more papers/configs; compare per paper",
    }

    # Pre-registered decision flags
    t = rep["summaries_treat"]
    rep["decision"] = {
        "summaries_complete (unfinished<20% and at-cap<20%)": bool(
            t["share_unfinished"] is not None
            and t["share_unfinished"] < 0.2
            and t["share_at_cap"] < 0.2
        ),
        "hierarchical_competitive@1000 (collapsed - leaf CI lower bound > -0.01)": did[
            "collapsed@1000"
        ]["gap_to_leaf_treat"][1]
        > -0.01,
        "hierarchical_competitive@2000": did["collapsed@2000"]["gap_to_leaf_treat"][1] > -0.01,
        "did_collapsed@1000_supported (97.5% CI > 0)": did["collapsed@1000"][
            "difference_in_differences_97_5"
        ][1]
        > 0,
        "did_collapsed@2000_supported (97.5% CI > 0)": did["collapsed@2000"][
            "difference_in_differences_97_5"
        ][1]
        > 0,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary_cap_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(
        json.dumps(
            {
                k: rep[k]
                for k in (
                    "summaries_control",
                    "summaries_treat",
                    "best_summary_evidence_recall",
                    "decision",
                )
            },
            indent=1,
        )
    )
    for c in cfgs:
        r = per[c]
        print(
            f"{c:18} F1 {r['control']['f1']:.3f} -> {r['treat']['f1']:.3f}  "
            f"delta {r['delta_f1_treat_minus_control']}  ctx {r['control']['context_tokens']:.0f}->"
            f"{r['treat']['context_tokens']:.0f}  summ.share {r['control']['summary_node_share']}->"
            f"{r['treat']['summary_node_share']}  leaf cov "
            f"{r['control']['leaf_evidence_coverage']}->{r['treat']['leaf_evidence_coverage']}"
        )
    print(f"wrote {a.out / 'summary_cap_report.json'}")


if __name__ == "__main__":
    main()

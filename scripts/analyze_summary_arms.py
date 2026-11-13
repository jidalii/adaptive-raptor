"""Compare several summary arms on the same questions (offline): e.g. cap 200 (upstream
prompt), cap 512 (upstream prompt), cap 512 + evidence_dense_v1 prompt.

    uv run python scripts/analyze_summary_arms.py \
        --arm cap200=outputs/pilot_catalog/20261109T195138Z:200 \
        --arm cap512=outputs/summary_cap512_pilot/<run>:512 \
        --arm dense=outputs/summary_dense_pilot/<run>:512 \
        --reference dense --out outputs/diagnostics/summary_arms_pilot

Per arm: summary completeness (median tokens, share at the cap, share ending mid-sentence),
evidence recall (best gold-evidence recall of any summary node), evidence density (that
recall per 100 summary tokens), per-config F1 / context tokens / summary share / leaf- and
context-evidence coverage, (strategy - leaf_only) gaps, index-build and inference USD.
Paired deltas (document bootstrap, 95%) of the reference arm against every other arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).parent))
from analyze_summary_cap import composition, configs, recall, summary_stats  # noqa: E402
from offline_diagnostics import PRICE_IN, PRICE_OUT, Split, boot, system  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")


def best_summary(split: Split, tree, q: str) -> tuple[float, int]:
    """(recall, tokens) of the summary node with the highest gold-evidence recall."""
    best = (0.0, 0)
    for k, nodes in tree.layer_to_nodes.items():
        if int(k) >= 1:
            for n in nodes:
                r = recall(split.evidence.get(q, []), n.text)
                if r > best[0]:
                    best = (r, len(ENC.encode(n.text)))
    return best


def index_cost(run: Path) -> dict:
    st = json.loads((run / "status.json").read_text())["metrics"]
    u = st.get("summarizer_usage", {})
    n = max(1, st.get("indexes_built", 0) + st.get("indexes_reused", 0))
    usd = (u.get("input_tokens", 0) * PRICE_IN + u.get("output_tokens", 0) * PRICE_OUT) / 1e6
    return {
        "calls": u.get("calls"),
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "usd": round(usd, 4),
        "usd_per_paper": round(usd / n, 5),
        "note": "0 if the run reused cached indexes (see the run that built them)",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--arm", action="append", required=True, help="name=run_dir:cap")
    p.add_argument("--reference", required=True, help="arm compared against all others")
    p.add_argument("--split", default="dev")
    p.add_argument(
        "--index-cost-run",
        action="append",
        default=[],
        help="name=run_dir whose status.json holds the index-build usage",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/diagnostics/summary_arms"))
    a = p.parse_args()
    qfile = Path(f"data/processed/qasper/v0.3/{a.split}/questions.jsonl")
    cfgs = configs()
    arms = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        run, cap = rest.rsplit(":", 1)
        arms[name] = (Split(Path(run), None, qfile, configs=cfgs), int(cap), Path(run))
    qs = sorted(set.intersection(*(set(s.qs) for s, _, _ in arms.values())))
    docs = {arms[a.reference][0].doc[q] for q in qs}
    for s, _, _ in arms.values():
        s.qs = qs
        s.index_hash = {d: h for d, h in s.index_hash.items() if d in docs}
    ref = arms[a.reference][0]
    with_ev = [q for q in qs if any(not e.startswith("FLOAT") for e in ref.evidence.get(q, []))]
    print(f"{len(qs)} questions, {len(docs)} papers, {len(with_ev)} with text evidence")

    rep: dict = {
        "n_questions": len(qs),
        "n_papers": len(docs),
        "reference": a.reference,
        "arms": {},
    }
    best = {}
    trees_by_arm = {}
    for name, (s, cap, run) in arms.items():
        trees, stats = summary_stats(s, cap)
        trees_by_arm[name] = trees
        best[name] = {q: best_summary(s, trees[s.doc[q]], q) for q in with_ev}
        rec = np.mean([best[name][q][0] for q in with_ev])
        dens = np.mean([best[name][q][0] / max(1, best[name][q][1]) * 100 for q in with_ev])
        arm = {
            "summaries": stats,
            "best_summary_evidence_recall": round(float(rec), 4),
            "evidence_density_per_100_tokens": round(float(dens), 4),
            "per_config": {},
        }
        for c in cfgs:
            comp = [composition(s, trees, q, c) for q in qs]
            arm["per_config"][c] = {
                **system(s, {q: c for q in qs}, name=f"{name}:{c}"),
                "context_tokens": round(float(np.mean([s.ctx_tokens[q][c] for q in qs])), 1),
                "summary_node_share": round(float(np.mean([x["summary_share"] for x in comp])), 3),
                "leaf_evidence_coverage": round(
                    float(np.mean([x["leaf_cov"] for x in comp if x["leaf_cov"] is not None])), 3
                ),
                "context_evidence_coverage": round(
                    float(np.mean([x["ctx_cov"] for x in comp if x["ctx_cov"] is not None])), 3
                ),
            }
        arm["gap_to_leaf"] = {
            f"{st}@{b}": boot(
                {q: s.f1[q][f"{st}@{b}"] - s.f1[q][f"leaf_only@{b}"] for q in qs}, s.doc
            )
            for st in ("collapsed", "summary_only")
            for b in (1000, 2000)
        }
        arm["index_build_cost"] = index_cost(run)
        rep["arms"][name] = arm

    for spec in a.index_cost_run:
        name, run = spec.split("=", 1)
        rep["arms"][name]["index_build_cost"] = index_cost(Path(run))

    # paired deltas: reference vs every other arm
    rep["reference_minus"] = {}
    for name, (s, _, _) in arms.items():
        if name == a.reference:
            continue
        d = {
            "best_summary_evidence_recall": boot(
                {q: best[a.reference][q][0] - best[name][q][0] for q in with_ev}, ref.doc
            )
        }
        for c in cfgs:
            d[f"f1 {c}"] = boot({q: ref.f1[q][c] - s.f1[q][c] for q in qs}, ref.doc)
        for st in ("collapsed", "summary_only"):
            for b in (1000, 2000):
                c, lf = f"{st}@{b}", f"leaf_only@{b}"
                d[f"DiD {c}"] = boot(
                    {q: (ref.f1[q][c] - ref.f1[q][lf]) - (s.f1[q][c] - s.f1[q][lf]) for q in qs},
                    ref.doc,
                )
        rep["reference_minus"][name] = d

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary_arms_report.json").write_text(json.dumps(rep, indent=1, default=str))
    for name, arm in rep["arms"].items():
        sm = arm["summaries"]
        print(
            f"[{name}] median {sm['median_tokens']} at-cap {sm['share_at_cap']} unfinished "
            f"{sm['share_unfinished']} recall {arm['best_summary_evidence_recall']} density "
            f"{arm['evidence_density_per_100_tokens']} index ${arm['index_build_cost']['usd']}"
        )
        for c in cfgs:
            r = arm["per_config"][c]
            print(
                f"    {c:18} F1 {r['f1']:.3f} ctx {r['context_tokens']:6.0f} summ "
                f"{r['summary_node_share']:.2f} leafcov {r['leaf_evidence_coverage']:.3f} "
                f"ctxcov {r['context_evidence_coverage']:.3f} $/1k {r['usd_per_1k_q']}"
            )
    for name, d in rep["reference_minus"].items():
        print(f"{a.reference} - {name}:")
        for k, v in d.items():
            if not k.startswith("f1 leaf"):
                print(f"    {k:28} {v}")
    print(f"wrote {a.out / 'summary_arms_report.json'}")


if __name__ == "__main__":
    main()

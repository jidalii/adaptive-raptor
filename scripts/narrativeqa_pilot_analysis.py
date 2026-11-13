"""NarrativeQA pilot analysis (pre-registered in NOTES.md, 2026-11-11). Offline, no API.

    uv run python scripts/narrativeqa_pilot_analysis.py --run-a DIR --run-b DIR [--out DIR]

Metric: ROUGE-L F1 (max over the two references) from traces; token F1 as secondary.
Cost per question and config = generation input + output tokens (as in the QASPER E1).

1. Tree structure: layers and nodes per layer per story (indexes.jsonl).
2. RAPTOR claim, per run: each collapsed/summary_only config vs the leaf-only fixed
   mixture (hash-mixed upper hull of leaf@400/1000/2000) at the config's own mean cost;
   paired document bootstrap.
3. Headroom decomposition (as E1): per-question oracle and best fixed policy chosen on run A
   at cost budgets, scored on run A (in-sample) and run B (replicated), for the leaf-budget
   space and for all 9 configs.
4. Go/no-go rule from the pre-registration.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import ALL9, COLL, LEAF, SUMM, boot  # noqa: E402
from section5_diagnostics import choice_at, lagrangian_path, value_at  # noqa: E402

BUDGETS = [700, 1000, 1500, 2000]


class Run:
    def __init__(self, run: Path, metric: str) -> None:
        self.f: dict[str, dict[str, float]] = defaultdict(dict)
        self.c: dict[str, dict[str, float]] = defaultdict(dict)
        self.doc: dict[str, str] = {}
        self.errors = 0
        for line in (run / "traces.jsonl").open():
            t = json.loads(line)
            if t["error"] is not None:
                self.errors += 1
                continue
            q, cfg = t["question_id"], t["config_id"]
            self.f[q][cfg] = float(t[metric])
            self.c[q][cfg] = float(t["input_tokens"] + t["output_tokens"])
            self.doc[q] = t["doc_id"]
        self.run = run

    def mats(self, qs, cfgs):
        F = np.array([[self.f[q][c] for c in cfgs] for q in qs])
        C = np.array([[self.c[q][c] for c in cfgs] for q in qs])
        return F, C


def tree_structure(run: Path) -> dict:
    rows = [json.loads(line) for line in (run / "indexes.jsonl").open()]
    out = []
    for r in rows:
        b = r.get("build_stats") or {}
        out.append(
            {
                "doc_id": r["doc_id"],
                "error": r["error"],
                "num_layers": b.get("num_layers"),
                "nodes_per_layer": b.get("nodes_per_layer"),
                "leaf_tokens": b.get("leaf_tokens"),
                "summary_tokens": b.get("summary_tokens"),
            }
        )
    hist: dict[str, int] = defaultdict(int)
    for r in out:
        hist[str(r["num_layers"])] += 1
    return {"histogram_num_summary_layers": dict(hist), "stories": out}


def raptor_claim(r: Run, qs) -> list[dict]:
    FL, CL = r.mats(qs, LEAF)
    leaf_path = lagrangian_path(FL, CL, np.zeros(len(qs), int))
    rows = []
    for cfg in COLL + SUMM:
        F, C = r.mats(qs, [cfg])
        cost = float(C.mean())
        pick = choice_at(leaf_path, cost, qs)
        if pick is None or cost > leaf_path[-1][0] + 1e-9:
            rows.append({"config": cfg, "mean_cost": round(cost, 1), "comparable": False})
            continue
        delta = {q: float(F[i, 0] - FL[i, pick[i]]) for i, q in enumerate(qs)}
        rows.append(
            {
                "config": cfg,
                "mean_cost": round(cost, 1),
                "score": round(float(F.mean()), 4),
                "leaf_mix_score": round(float(value_at(leaf_path, cost)), 4),
                "delta_ci": boot(delta, r.doc),
                "comparable": True,
            }
        )
    return rows


def headroom(a: Run, b: Run, qs) -> dict:
    out = {}
    for name, cfgs in (("leaf_budgets", LEAF), ("all9", ALL9)):
        FA, CA = a.mats(qs, cfgs)
        FB, _ = b.mats(qs, cfgs)
        orc = lagrangian_path(FA, CA, np.arange(len(qs)))
        fix = lagrangian_path(FA, CA, np.zeros(len(qs), int))
        rows = []
        for bud in BUDGETS:
            po, pf = choice_at(orc, bud, qs), choice_at(fix, bud, qs)
            if po is None or pf is None or bud > fix[-1][0] + 1e-9:
                continue
            i = np.arange(len(qs))
            rows.append(
                {
                    "budget": bud,
                    "in_sample_A": round(float(FA[i, po].mean() - FA[i, pf].mean()), 4),
                    "replicated_B": boot(
                        {q: float(FB[k, po[k]] - FB[k, pf[k]]) for k, q in enumerate(qs)}, a.doc
                    ),
                }
            )
        out[name] = {
            "rows": rows,
            "oracle_max_A": round(orc[-1][1], 4),
            "oracle_saturation_cost_A": round(orc[-1][0], 1),
        }
    return out


def fixed_table(r: Run, qs) -> list[dict]:
    return [
        {
            "config": c,
            "score": round(float(np.mean([r.f[q][c] for q in qs])), 4),
            "mean_cost": round(float(np.mean([r.c[q][c] for q in qs])), 1),
        }
        for c in ALL9
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-a", type=Path, required=True)
    p.add_argument("--run-b", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or Path("outputs/diagnostics/narrativeqa_pilot") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    rep: dict = {
        "note": "offline; NarrativeQA dev pilot (20 stories); metric ROUGE-L F1 max over refs "
        "(primary) and token F1; cost = generation in+out tokens; doc bootstrap 2000x 95%",
        "runs": {"A": str(a.run_a), "B": str(a.run_b)},
        "trees": tree_structure(a.run_a),
    }
    for metric in ("answer_rouge_l", "answer_f1"):
        ra, rb = Run(a.run_a, metric), Run(a.run_b, metric)
        qs = sorted(
            q
            for q in ra.f
            if all(c in ra.f[q] for c in ALL9) and q in rb.f and all(c in rb.f[q] for c in ALL9)
        )
        rep[metric] = {
            "n_questions": len(qs),
            "n_stories": len({ra.doc[q] for q in qs}),
            "errors": {"A": ra.errors, "B": rb.errors},
            "fixed_A": fixed_table(ra, qs),
            "fixed_B": fixed_table(rb, qs),
            "raptor_claim_A": raptor_claim(ra, qs),
            "raptor_claim_B": raptor_claim(rb, qs),
            "headroom": headroom(ra, rb, qs),
        }
    m = rep["answer_rouge_l"]
    a_ok = {r["config"]: r["delta_ci"][0] for r in m["raptor_claim_A"] if r["comparable"]}
    b_ok = {r["config"]: r["delta_ci"][0] for r in m["raptor_claim_B"] if r["comparable"]}
    crit_a = [c for c in a_ok if c in b_ok and a_ok[c] >= 0.01 and b_ok[c] >= 0.01]
    h = m["headroom"]
    leaf1000 = next((r for r in h["leaf_budgets"]["rows"] if r["budget"] == 1000), None)
    all1000 = next((r for r in h["all9"]["rows"] if r["budget"] == 1000), None)
    crit_b = (
        leaf1000 is not None
        and all1000 is not None
        and all1000["replicated_B"][0] - leaf1000["replicated_B"][0] >= 0.02
    )
    rep["decision"] = {
        "a_configs_beating_leaf_mix_by_1pt_in_both_runs": crit_a,
        "b_all9_minus_leaf_replicated_headroom_at_1000_ge_2pt": bool(crit_b),
        "go_full_dev": bool(crit_a) or bool(crit_b),
    }
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["trees"]["histogram_num_summary_layers"]))
    for metric in ("answer_rouge_l", "answer_f1"):
        mm = rep[metric]
        print(f"== {metric}: n={mm['n_questions']} stories={mm['n_stories']} errors={mm['errors']}")
        for run in ("A", "B"):
            print(
                f"  fixed {run}: "
                + "  ".join(
                    f"{r['config']}={r['score']:.3f}@{r['mean_cost']:.0f}"
                    for r in mm[f"fixed_{run}"]
                )
            )
            for r in mm[f"raptor_claim_{run}"]:
                if r["comparable"]:
                    print(
                        f"  claim {run} {r['config']:18} {r['score']:.4f} vs leaf mix "
                        f"{r['leaf_mix_score']:.4f}  delta {r['delta_ci']}"
                    )
                else:
                    print(f"  claim {run} {r['config']:18} cost {r['mean_cost']} not comparable")
        for name, v in mm["headroom"].items():
            print(f"  headroom {name}: max {v['oracle_max_A']} @ {v['oracle_saturation_cost_A']}")
            for r in v["rows"]:
                print(
                    f"    {r['budget']}: in-sample {r['in_sample_A']:+.4f}  "
                    f"replicated B {r['replicated_B']}"
                )
    print("decision:", rep["decision"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

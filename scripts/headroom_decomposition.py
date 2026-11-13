"""E1: where does the oracle headroom come from? Offline, NO API CALLS, test untouched.

    uv run python scripts/headroom_decomposition.py [--out DIR]

1. Frontiers per action space (dev): leaf budgets only (leaf@400/1000/2000), hierarchical
   only (collapsed + summary_only, 6 configs), strategy at a fixed budget (3 strategies at
   @1000 and at @2000), and all 9. For each, the best fixed policy (mixing allowed) and
   the per-question oracle at the same mean tokens, and the headroom between them.
   Hierarchical configs are shown both without and with index-build cost: summarizer
   tokens per question, amortized over the dev questions. That cost is charged to a
   question whenever its chosen config needs the tree.
2. Oracle sanity checks: where the oracle saturates (above that cost the budget
   constraint is slack, so the F1 is flat by construction); tie structure (questions whose
   max F1 is reached by several configs, by every config, or where everything scores 0);
   the oracle F1 if ties go to the most expensive config instead of the cheapest.
3. Replicability on the pilot (87 questions x 9 configs, two generations A/B): per
   action space, the headroom with the oracle chosen on run A and scored on run B. Also
   per config: how often its gain over leaf@1000 has the same sign in both runs
   ("stable gains").
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import ALL9, COLL, LEAF, SUMM, Split, boot, latest  # noqa: E402
from section5_diagnostics import choice_at, lagrangian_path, value_at  # noqa: E402

BUDGETS = [425, 500, 700, 1000, 1500, 2043]
SPACES = {
    "leaf_budgets": LEAF,
    "hierarchical_only": COLL + SUMM,
    "strategy@1000": ["leaf_only@1000", "collapsed@1000", "summary_only@1000"],
    "strategy@2000": ["leaf_only@2000", "collapsed@2000", "summary_only@2000"],
    "all9": ALL9,
}


def mats(s: Split, qs, cfgs, idx_per_q: float = 0.0):
    F = np.array([[s.f1[q][c] for c in cfgs] for q in qs])
    C = np.array(
        [
            [s.cost(q, c) + (0.0 if c.startswith("leaf_only") else idx_per_q) for c in cfgs]
            for q in qs
        ],
        dtype=float,
    )
    return F, C


def space_report(s: Split, qs, cfgs, idx_per_q: float) -> dict:
    F, C = mats(s, qs, cfgs, idx_per_q)
    fixed = lagrangian_path(F, C, np.zeros(len(qs), int))
    orc = lagrangian_path(F, C, np.arange(len(qs)))
    rows = []
    for b in BUDGETS:
        fb, ob = value_at(fixed, b), value_at(orc, b)
        r = {"budget": b, "fixed_f1": fb and round(fb, 4), "oracle_f1": ob and round(ob, 4)}
        if fb is not None and ob is not None:
            po, pf = choice_at(orc, b, qs), choice_at(fixed, b, qs)
            r["headroom_ci"] = boot(
                {q: float(F[i, po[i]] - F[i, pf[i]]) for i, q in enumerate(qs)}, s.doc
            )
        rows.append(r)
    return {
        "configs": cfgs,
        "rows": rows,
        "oracle_max_f1": round(orc[-1][1], 4),
        "oracle_saturation_cost": round(orc[-1][0], 1),
        "fixed_path": [[round(c), round(f, 4)] for c, f, _ in fixed],
        "oracle_path": [[round(c), round(f, 4)] for c, f, _ in orc],
    }


def ties(s: Split, qs) -> dict:
    n_all_zero = n_unique = n_multi = n_all9 = 0
    exp_tie_f1, costs_cheap, costs_exp = [], [], []
    for q in qs:
        f = np.array([s.f1[q][c] for c in ALL9])
        c = np.array([s.cost(q, x) for x in ALL9])
        top = f.max()
        win = np.flatnonzero(f >= top - 1e-12)
        if top <= 0:
            n_all_zero += 1
        elif len(win) == 1:
            n_unique += 1
        else:
            n_multi += 1
            n_all9 += len(win) == 9
        exp_tie_f1.append(top)
        costs_cheap.append(c[win].min())
        costs_exp.append(c[win].max())
    n = len(qs)
    return {
        "n": n,
        "all_configs_zero": round(n_all_zero / n, 3),
        "unique_best_config": round(n_unique / n, 3),
        "max_reached_by_several": round(n_multi / n, 3),
        "max_reached_by_all_9": round(n_all9 / n, 3),
        "oracle_mean_f1_unconstrained": round(float(np.mean(exp_tie_f1)), 4),
        "oracle_mean_tokens_ties_to_cheapest": round(float(np.mean(costs_cheap)), 1),
        "oracle_mean_tokens_ties_to_most_expensive": round(float(np.mean(costs_exp)), 1),
        "note": "the unconstrained oracle F1 is identical under both tie rules; only its cost "
        "changes. Budgets above the cheapest-tie cost leave the constraint slack, hence a "
        "flat oracle F1",
    }


def pilot(pa: Split, pb: Split) -> dict:
    qs = [q for q in pa.qs if q in pb.f1 and all(c in pb.f1[q] for c in ALL9)]
    out = {"n": len(qs), "spaces": {}, "stable_gain_vs_leaf1000": {}}
    for name, cfgs in SPACES.items():
        FA, CA = mats(pa, qs, cfgs)
        FB = np.array([[pb.f1[q][c] for c in cfgs] for q in qs])
        o, f = (
            lagrangian_path(FA, CA, np.arange(len(qs))),
            lagrangian_path(FA, CA, np.zeros(len(qs), int)),
        )
        rows = []
        for b in (700, 1000, 1500):
            po, pf = choice_at(o, b, qs), choice_at(f, b, qs)
            if po is None or pf is None:
                continue
            i = np.arange(len(qs))
            ins = float(FA[i, po].mean() - FA[i, pf].mean())
            oos = float(FB[i, po].mean() - FB[i, pf].mean())
            rows.append(
                {
                    "budget": b,
                    "in_sample_run_A": round(ins, 4),
                    "scored_on_run_B": round(oos, 4),
                    "ci_run_B": boot(
                        {q: float(FB[k, po[k]] - FB[k, pf[k]]) for k, q in enumerate(qs)},
                        pa.doc,
                    ),
                }
            )
        out["spaces"][name] = rows
    base = "leaf_only@1000"
    for c in ALL9:
        if c == base:
            continue
        da = np.array([pa.f1[q][c] - pa.f1[q][base] for q in qs])
        db = np.array([pb.f1[q][c] - pb.f1[q][base] for q in qs])
        nz = da != 0
        out["stable_gain_vs_leaf1000"][c] = {
            "share_gain_in_A": round(float((da > 0).mean()), 3),
            "share_loss_in_A": round(float((da < 0).mean()), 3),
            "share_gain_in_both": round(float(((da > 0) & (db > 0)).mean()), 3),
            "share_loss_in_both": round(float(((da < 0) & (db < 0)).mean()), 3),
            "sign_agreement_when_A_nonzero": round(
                float((np.sign(da[nz]) == np.sign(db[nz])).mean()) if nz.any() else 0.0, 3
            ),
            "mean_gain_A": round(float(da.mean()), 4),
            "mean_gain_B": round(float(db.mean()), 4),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or Path("outputs/diagnostics/headroom_decomposition") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    dq = Path("data/processed/qasper/v0.3/dev/questions.jsonl")
    run = latest("outputs/labels_dev/*/")
    dv = Split(run, None, dq)
    st = json.loads((run / "status.json").read_text())["metrics"]["summarizer_usage"]
    idx = (st["input_tokens"] + st["output_tokens"]) / len(dv.qs)
    rep = {
        "note": "offline; dev; no API calls; baseline = best fixed policy with mixing (upper "
        "hull of single configs); costs = generation in+out tokens",
        "index_tokens_per_question": round(idx, 1),
        "spaces": {},
        "spaces_with_index_cost": {},
        "ties": ties(dv, dv.qs),
        "pilot": pilot(
            Split(Path("outputs/pilot_catalog/20261109T193103Z"), None, dq),
            Split(Path("outputs/pilot_catalog/20261109T194522Z"), None, dq),
        ),
    }
    for name, cfgs in SPACES.items():
        rep["spaces"][name] = space_report(dv, dv.qs, cfgs, 0.0)
        if name != "leaf_budgets":
            rep["spaces_with_index_cost"][name] = space_report(dv, dv.qs, cfgs, idx)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["ties"], indent=1))
    for key in ("spaces", "spaces_with_index_cost"):
        print(f"== {key} ==")
        for name, r in rep[key].items():
            cells = " ".join(
                f"{x['budget']}:{x['fixed_f1']}/{x['oracle_f1']}"
                + (f"({x['headroom_ci'][0]:+.3f})" if "headroom_ci" in x else "")
                for x in r["rows"]
            )
            print(f"  {name:18} max {r['oracle_max_f1']} @ {r['oracle_saturation_cost']}  {cells}")
    print("== pilot ==")
    for name, rows in rep["pilot"]["spaces"].items():
        print(f"  {name}: {rows}")
    for c, v in rep["pilot"]["stable_gain_vs_leaf1000"].items():
        print(f"  {c}: {v}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

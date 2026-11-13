"""Section 5 diagnostics from existing data (offline, NO API CALLS; test split untouched).

    uv run --group plots python scripts/section5_diagnostics.py [--out DIR] [--no-plots]

A. Oracle headroom. Delta_headroom(B) = F1_oracle(B) - F1_fixed(B) at the same mean
   per-question token budget B.
   - fixed(B): the best fixed policy at mean cost B. Every question gets the same
     configuration; two neighbouring configurations are mixed by a question hash, so
     this is the upper concave hull of the 9 catalog configs.
   - oracle(B): per-question choice using the TRUE F1 of every config, under the
     constraint mean cost <= B, solved with a Lagrangian over the 9 configs. This is an
     upper bound for any router over this catalog.
   - cell oracle(B): one choice per LLM-profile cell, in hindsight on dev. This is an
     upper bound for any router that sees only the profile, charged its profiler tokens.
   - train-fit cell policy: the same per-cell choice fitted on train, with lambda
     calibrated on train, then scored on dev. This is an honest profile router.
   - Noise: the in-sample oracle exploits generator noise (about 20% of (question,
     config) F1 values change on a rerun). On the 87 pilot questions with two runs (A, B),
     the oracle chooses on run A and is scored on run B. That gives the share of the
     headroom that replicates.
   Paired document-bootstrap CIs (95%) compare per-question choices at budget B:
   oracle or policy vs the fixed mix at the same budget.
B. Profiler predictive power (train and dev separately):
   - contingency of each profile field vs the per-question best config (ties go to the
     cheaper config; all-zero questions excluded), as chi-square and Cramer's V;
   - mean F1 of every config per category, and the best config per category;
   - interaction tests for the assumptions behind RULES_V1, each as a difference of
     mean gains between categories with a document-bootstrap CI:
       H1 multi-fact / cross-section questions gain more from more leaf context;
       H2 whole-paper questions gain more from summaries (collapsed / summary_only);
       H3 single-fact yes/no or short-span questions need less context;
   - profiler answer_form vs gold answer type (is the profile itself accurate?).
C. Unified token accounting for every system: generation input / output, profiler
   input / output, total, USD (off-peak), and index-build summarizer tokens amortized
   per question. leaf_only needs no summaries; every other strategy needs the RAPTOR tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import (  # noqa: E402
    ALL9,
    ORIG,
    PRICE_IN,
    PRICE_OUT,
    Split,
    boot,
    latest,
)

BUDGETS = [500, 700, 1000, 1200, 1500, 1800, 2000]
LAMBDAS = np.concatenate([[0.0], np.geomspace(1e-8, 1.0, 1500), [10.0, 1e3]])
CHEAPEST_TIE = 1e-9


def rank(q: str) -> float:
    return int(hashlib.md5(q.encode()).hexdigest(), 16) % 10_000 / 10_000


# ---------------------------------------------------------------------------- A: frontiers


def lagrangian_path(F: np.ndarray, C: np.ndarray, groups: np.ndarray):
    """For each lambda, each group picks argmax(sum F - lambda * sum C). Returns the
    vertices of the upper hull as (mean cost, mean F1, per-question choice index)."""
    gids = np.unique(groups)
    sumF = np.stack([F[groups == g].sum(0) for g in gids])
    sumC = np.stack([C[groups == g].sum(0) for g in gids])
    pos = {g: i for i, g in enumerate(gids)}
    gi = np.array([pos[g] for g in groups])
    scale = float(np.median(C)) or 1.0
    pts = {}
    for lam in LAMBDAS / scale:
        pick_g = np.argmax(sumF - lam * sumC - CHEAPEST_TIE * sumC, axis=1)
        pick = pick_g[gi]
        c = float(C[np.arange(len(F)), pick].mean())
        f = float(F[np.arange(len(F)), pick].mean())
        pts.setdefault(round(c, 6), (c, f, pick))
    hull = []
    for c, f, pick in sorted(pts.values(), key=lambda x: x[0]):
        if not hull or f > hull[-1][1] + 1e-12:
            hull.append((c, f, pick))
    # keep only concave-hull vertices (randomised mixing dominates interior points)
    out = []
    for p in hull:
        while len(out) >= 2:
            (c0, f0, _), (c1, f1, _) = out[-2], out[-1]
            if (f1 - f0) * (p[0] - c0) <= (p[1] - f0) * (c1 - c0):
                out.pop()
            else:
                break
        out.append(p)
    return out


def choice_at(path, budget: float, qs: list[str]) -> np.ndarray | None:
    """Per-question config index of the hash-mixed policy with mean cost = budget."""
    if budget < path[0][0] - 1e-9:
        return None
    if budget >= path[-1][0]:
        return path[-1][2]
    for (c0, _, p0), (c1, _, p1) in zip(path, path[1:], strict=False):
        if c0 <= budget <= c1:
            w = (budget - c0) / (c1 - c0) if c1 > c0 else 0.0
            return np.array([p1[i] if rank(q) < w else p0[i] for i, q in enumerate(qs)])
    return path[-1][2]


def value_at(path, budget: float) -> float | None:
    if budget < path[0][0] - 1e-9:
        return None
    if budget >= path[-1][0]:
        return path[-1][1]
    for (c0, f0, _), (c1, f1, _) in zip(path, path[1:], strict=False):
        if c0 <= budget <= c1:
            return f0 + (f1 - f0) * ((budget - c0) / (c1 - c0) if c1 > c0 else 0.0)
    return path[-1][1]


def matrices(s: Split, qs: list[str], configs: list[str]):
    F = np.array([[s.f1[q][c] for c in configs] for q in qs])
    C = np.array([[s.cost(q, c) for c in configs] for q in qs], dtype=float)
    return F, C


def cell_key(p: dict | None, level: str) -> str:
    if p is None:
        return "no_profile"
    nf = "1" if p["n_facts"] == 1 else "2" if p["n_facts"] == 2 else "3+"
    full = f"{p['answer_form']}|{p['granularity']}|{nf}|{'x' if p['cross_section'] else '-'}"
    return {
        "full": full,
        "answer_form": p["answer_form"],
        "granularity": p["granularity"],
        "n_facts": nf,
        "cross_section": str(p["cross_section"]),
    }[level]


def headroom(tr: Split, dv: Split, pilot: tuple[Split, Split]) -> dict:
    qs = dv.qs
    F, C = matrices(dv, qs, ALL9)
    prof = np.array([sum(dv.prof_tok.get(q, (0, 0))) for q in qs], dtype=float)
    doc = dv.doc
    fixed = lagrangian_path(F, C, np.zeros(len(qs), int))
    oracle = lagrangian_path(F, C, np.arange(len(qs)))
    rep: dict = {"n": len(qs), "budgets": BUDGETS, "rows": [], "cell_oracles": {}}

    def delta_ci(pick_a, pick_b, Fm=F) -> list[float]:
        d = {q: float(Fm[i, pick_a[i]] - Fm[i, pick_b[i]]) for i, q in enumerate(qs)}
        return boot(d, doc)

    for b in BUDGETS:
        fb, ob = value_at(fixed, b), value_at(oracle, b)
        row = {"budget": b, "fixed_f1": fb, "oracle_f1": ob}
        if fb is not None and ob is not None:
            row["headroom"] = round(ob - fb, 4)
            row["headroom_ci"] = delta_ci(choice_at(oracle, b, qs), choice_at(fixed, b, qs))
        rep["rows"].append(row)

    # cell oracles in hindsight (dev), charged profiler tokens
    for lv, charge in [
        (lv, ch)
        for lv in ("full", "answer_form", "granularity", "n_facts", "cross_section")
        for ch in (True, False)
    ]:
        groups = np.array([cell_key(dv.profile.get(q), lv) for q in qs])
        path = lagrangian_path(F, C + (prof[:, None] if charge else 0), groups)
        level = lv if charge else f"{lv} (profiler tokens free)"
        rows = []
        for b in BUDGETS:
            v, fb = value_at(path, b), value_at(fixed, b)
            r = {"budget": b, "cell_oracle_f1": None if v is None else round(v, 4)}
            if v is not None and fb is not None:
                r["minus_fixed"] = round(v - fb, 4)
                r["ci"] = delta_ci(choice_at(path, b, qs), choice_at(fixed, b, qs))
            rows.append(r)
        rep["cell_oracles"][level] = {
            "profiler_tokens_charged": charge,
            "n_cells": int(len(np.unique(groups))),
            "rows": rows,
            "path": [[round(c), round(f, 4)] for c, f, _ in path],
        }

    # honest profile router: per-cell choice fitted on train, lambda calibrated on train
    tq = tr.qs
    Ft, Ct = matrices(tr, tq, ALL9)
    pt = np.array([sum(tr.prof_tok.get(q, (0, 0))) for q in tq], dtype=float)
    gt = np.array([cell_key(tr.profile.get(q), "full") for q in tq])
    gd = np.array([cell_key(dv.profile.get(q), "full") for q in qs])
    cells = sorted(set(gt))
    scale = float(np.median(Ct)) or 1.0
    fitted = []
    for lam in LAMBDAS / scale:
        glob = int(np.argmax(Ft.sum(0) - lam * (Ct + pt[:, None]).sum(0)))
        pick_cell = {}
        for g in cells:
            m = gt == g
            sc = Ft[m].sum(0) - lam * (Ct[m] + pt[m, None]).sum(0) - CHEAPEST_TIE * Ct[m].sum(0)
            pick_cell[g] = int(np.argmax(sc))
        tr_pick = np.array([pick_cell[g] for g in gt])
        dv_pick = np.array([pick_cell.get(g, glob) for g in gd])
        fitted.append(
            (
                float((Ct[np.arange(len(tq)), tr_pick] + pt).mean()),
                dv_pick,
            )
        )
    honest = []
    for b in BUDGETS:
        ok = [x for x in fitted if x[0] <= b]
        if not ok:
            continue
        _, pick = max(ok, key=lambda x: x[0])
        cost = float((C[np.arange(len(qs)), pick] + prof).mean())
        f1 = float(F[np.arange(len(qs)), pick].mean())
        fb = value_at(fixed, cost)
        fx = choice_at(fixed, cost, qs)
        honest.append(
            {
                "train_budget": b,
                "dev_mean_tokens_incl_profiler": round(cost, 1),
                "dev_f1": round(f1, 4),
                "fixed_f1_at_same_tokens": None if fb is None else round(fb, 4),
                "minus_fixed_ci": None if fx is None else delta_ci(pick, fx),
            }
        )
    rep["train_fit_cell_policy_dev"] = honest

    # replicability on the pilot (two generations A, B of the same 87 x 9 answers)
    pa, pb = pilot
    pq = [q for q in pa.qs if q in pb.f1 and all(c in pb.f1[q] for c in ALL9)]
    FA, CA = matrices(pa, pq, ALL9)
    FB = np.array([[pb.f1[q][c] for c in ALL9] for q in pq])
    o_path = lagrangian_path(FA, CA, np.arange(len(pq)))
    f_path = lagrangian_path(FA, CA, np.zeros(len(pq), int))
    rows = []
    for b in BUDGETS:
        po, pf = choice_at(o_path, b, pq), choice_at(f_path, b, pq)
        if po is None or pf is None:
            continue
        idx = np.arange(len(pq))
        ins = float(FA[idx, po].mean() - FA[idx, pf].mean())
        oos = float(FB[idx, po].mean() - FB[idx, pf].mean())
        dd = {q: float(FB[i, po[i]] - FB[i, pf[i]]) for i, q in enumerate(pq)}
        rows.append(
            {
                "budget": b,
                "in_sample_headroom_run_A": round(ins, 4),
                "out_of_sample_headroom_run_B": round(oos, 4),
                "oos_ci": boot(dd, pa.doc),
                "share_replicating": round(oos / ins, 3) if ins > 0 else None,
            }
        )
    rep["pilot_replicability"] = {"n": len(pq), "rows": rows}
    rep["paths"] = {
        "fixed": [[round(c), round(f, 4)] for c, f, _ in fixed],
        "oracle": [[round(c), round(f, 4)] for c, f, _ in oracle],
    }
    return rep


# ---------------------------------------------------------------------------- B: profiles


def gold_types(path: Path, qs: set[str]) -> dict[str, str]:
    out = {}
    for line in path.open():
        r = json.loads(line)
        if r["question_id"] in qs:
            t = Counter(a["answer_type"] for a in r["answers"]).most_common(1)
            out[r["question_id"]] = t[0][0] if t else "unknown"
    return out


def group_gain_ci(s: Split, gain: dict[str, float], cat: dict[str, str], a: str, b: str):
    """mean gain in category a minus category b, document bootstrap (2000, 95%)."""
    qa = [q for q in gain if cat.get(q) == a]
    qb = [q for q in gain if cat.get(q) == b]
    if not qa or not qb:
        return None
    point = float(np.mean([gain[q] for q in qa]) - np.mean([gain[q] for q in qb]))
    by_doc = defaultdict(list)
    for q in qa + qb:
        by_doc[s.doc[q]].append(q)
    docs = sorted(by_doc)
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(2000):
        sample = [q for d in rng.choice(docs, len(docs)) for q in by_doc[d]]
        xa = [gain[q] for q in sample if cat[q] == a]
        xb = [gain[q] for q in sample if cat[q] == b]
        if xa and xb:
            stats.append(np.mean(xa) - np.mean(xb))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        "a": a,
        "b": b,
        "n_a": len(qa),
        "n_b": len(qb),
        "mean_gain_a": round(float(np.mean([gain[q] for q in qa])), 4),
        "mean_gain_b": round(float(np.mean([gain[q] for q in qb])), 4),
        "difference": [round(point, 4), round(float(lo), 4), round(float(hi), 4)],
    }


def best_config(s: Split, q: str) -> str | None:
    """Cheapest config reaching the question's max F1; None if every config scores 0."""
    f = {c: s.f1[q][c] for c in ALL9}
    top = max(f.values())
    if top <= 0:
        return None
    ties = [c for c in ALL9 if f[c] >= top - 1e-12]
    return min(ties, key=lambda c: (s.cost(q, c), ALL9.index(c)))


def winner(s: Split, q: str, key) -> str | None:
    """Tie-aware winner over groups of configs (by strategy, budget or config): the group
    whose best F1 is highest; 'tie' if several groups reach it; None if all scores are 0."""
    best: dict[str, float] = {}
    for c in ALL9:
        best[key(c)] = max(best.get(key(c), 0.0), s.f1[q][c])
    top = max(best.values())
    if top <= 0:
        return None
    w = [k for k, v in best.items() if v >= top - 1e-12]
    return w[0] if len(w) == 1 else "tie"


def profile_power(s: Split, qfile: Path) -> dict:
    from scipy.stats import chi2_contingency

    qs = [q for q in s.qs if q in s.profile]
    rep: dict = {"n_with_profile": len(qs), "fields": {}}
    best = {q: best_config(s, q) for q in qs}
    for field in ("answer_form", "granularity", "n_facts", "cross_section"):
        cat = {q: cell_key(s.profile[q], field) for q in qs}
        cats = sorted(set(cat.values()))
        per = {}
        for c in cats:
            cq = [q for q in qs if cat[q] == c]
            means = {k: round(float(np.mean([s.f1[q][k] for q in cq])), 4) for k in ALL9}
            per[c] = {
                "n": len(cq),
                "mean_f1": means,
                "best_config_by_mean": max(means, key=means.get),
                "best_config_by_mean_f1": max(means.values()),
                "all_fail_share": round(float(np.mean([best[q] is None for q in cq])), 3),
            }
        tables = {}
        for unit, key in (
            ("config", lambda c: c),
            ("strategy", lambda c: c.split("@")[0]),
            ("budget", lambda c: c.split("@")[1]),
        ):
            win = {q: winner(s, q, key) for q in qs}
            cols = sorted({w for w in win.values() if w and w != "tie"}) + ["tie"]
            M = np.array(
                [[sum(1 for q in qs if cat[q] == r and win[q] == k) for k in cols] for r in cats]
            )
            entry = {"rows": cats, "cols": cols, "counts": M.tolist()}
            for label, Mx in (("with_ties", M), ("decisive_only", M[:, :-1])):
                keep_r, keep_c = Mx.sum(1) > 0, Mx.sum(0) > 0
                Mx = Mx[keep_r][:, keep_c]
                if min(Mx.shape) < 2:
                    entry[label] = None
                    continue
                chi2, pv, dof, _ = chi2_contingency(Mx)
                entry[label] = {
                    "n": int(Mx.sum()),
                    "chi2": round(float(chi2), 2),
                    "dof": int(dof),
                    "p_value": float(pv),
                    "cramers_v": round(float(np.sqrt(chi2 / (Mx.sum() * (min(Mx.shape) - 1)))), 3),
                }
            tables[unit] = entry
        rep["fields"][field] = {"categories": per, "best_config_contingency": tables}

    # assumptions behind RULES_V1 as interaction tests
    nf = {q: cell_key(s.profile[q], "n_facts") for q in qs}
    xs = {q: str(s.profile[q]["cross_section"]) for q in qs}
    gran = {q: s.profile[q]["granularity"] for q in qs}
    single = {
        q: "single_fact"
        if s.profile[q]["answer_form"] in ("yes_no", "short_span")
        and s.profile[q]["n_facts"] == 1
        and not s.profile[q]["cross_section"]
        else "other"
        for q in qs
    }

    def g(a: str, b: str) -> dict[str, float]:
        return {q: s.f1[q][a] - s.f1[q][b] for q in qs}

    leaf_more = g("leaf_only@2000", "leaf_only@400")
    leaf_more_1k = g("leaf_only@2000", "leaf_only@1000")
    rep["interactions"] = {
        "H1 n_facts 3+ vs 1: gain leaf@2000 - leaf@400": group_gain_ci(s, leaf_more, nf, "3+", "1"),
        "H1 n_facts 3+ vs 1: gain leaf@2000 - leaf@1000": group_gain_ci(
            s, leaf_more_1k, nf, "3+", "1"
        ),
        "H1 cross_section True vs False: gain leaf@2000 - leaf@400": group_gain_ci(
            s, leaf_more, xs, "True", "False"
        ),
        "H2 whole_paper vs specific_detail: gain collapsed@1000 - leaf@1000": group_gain_ci(
            s, g("collapsed@1000", "leaf_only@1000"), gran, "whole_paper", "specific_detail"
        ),
        "H2 whole_paper vs specific_detail: gain summary_only@1000 - leaf@1000": group_gain_ci(
            s, g("summary_only@1000", "leaf_only@1000"), gran, "whole_paper", "specific_detail"
        ),
        "H3 single_fact vs other: gain leaf@1000 - leaf@400": group_gain_ci(
            s, g("leaf_only@1000", "leaf_only@400"), single, "single_fact", "other"
        ),
    }
    gt = gold_types(qfile, set(qs))
    forms = ["yes_no", "short_span", "explanation", "unanswerable_likely"]
    types = ["boolean", "extractive", "abstractive", "none"]
    M = [
        [sum(1 for q in qs if s.profile[q]["answer_form"] == f and gt.get(q) == t) for t in types]
        for f in forms
    ]
    agree = sum(M[i][i] for i in range(4)) / max(1, sum(map(sum, M)))
    rep["answer_form_vs_gold_type"] = {
        "rows_profile": forms,
        "cols_gold": types,
        "counts": M,
        "diagonal_agreement": round(agree, 3),
    }
    return rep


# ---------------------------------------------------------------------------- C: accounting


def accounting(dv: Split, labels_run: Path) -> dict:
    from adaptive_raptor.config.loader import load_config
    from adaptive_raptor.experiments import _label_table, _replay
    from adaptive_raptor.optimizer import Router, load_checkpoint
    from adaptive_raptor.optimizer.profiler import ProfileStore, build_profiler
    from adaptive_raptor.profiling import merge_label_tables

    status = json.loads((labels_run / "status.json").read_text())["metrics"]
    su = status["summarizer_usage"]
    n_q = len(dv.qs)
    idx_in, idx_out = su["input_tokens"] / n_q, su["output_tokens"] / n_q
    store = ProfileStore.load([latest("outputs/profile_dev/*/") / "profiles.jsonl"])
    records, _ = merge_label_tables([labels_run / "labels.jsonl"])

    def row(name, choice, prof=None, needs_tree=True, note=""):
        qs = list(choice)
        gi = float(np.mean([dv.tin[q][choice[q]] for q in qs]))
        go = float(np.mean([dv.tout[q][choice[q]] for q in qs]))
        pi = float(np.mean([prof[q][0] for q in qs])) if prof else 0.0
        po = float(np.mean([prof[q][1] for q in qs])) if prof else 0.0
        tot_in, tot_out = gi + pi, go + po
        return {
            "system": name,
            "n": len(qs),
            "f1": round(float(np.mean([dv.f1[q][choice[q]] for q in qs])), 4),
            "generation_input": round(gi, 1),
            "generation_output": round(go, 1),
            "profiler_input": round(pi, 1),
            "profiler_output": round(po, 1),
            "total_tokens": round(tot_in + tot_out, 1),
            "usd_per_1k_q": round((tot_in * PRICE_IN + tot_out * PRICE_OUT) / 1e3, 4),
            "index_summarizer_tokens_per_q": round(idx_in + idx_out, 1) if needs_tree else 0.0,
            "usd_per_1k_q_incl_index": round(
                (
                    (tot_in + (idx_in if needs_tree else 0)) * PRICE_IN
                    + (tot_out + (idx_out if needs_tree else 0)) * PRICE_OUT
                )
                / 1e3,
                4,
            ),
            "note": note,
        }

    rows = [
        row(c, {q: c for q in dv.qs}, needs_tree=not c.startswith("leaf_only"))
        for c in [ORIG, *ALL9]
    ]
    for kind in ("llm_profile_table", "llm_profile_rules"):
        run = latest(f"outputs/router_train_{kind}/*/")
        rcfg = load_config(run / "resolved_config.yaml", [])
        catalog = rcfg.retrieval.catalog()
        ck = load_checkpoint(run / "router")
        table = _label_table(records, ck.artifacts["label_cell"], [c.id for c in catalog])
        router = Router.from_checkpoint(
            run / "router", rcfg.optimizer, catalog, build_profiler(rcfg.optimizer.profiler)
        )
        choice, prof, _, missing = _replay(router, {q: table[q] for q in dv.qs}, store)
        rows.append(
            row(
                {"llm_profile_table": "A1 router", "llm_profile_rules": "A2 router"}[kind],
                choice,
                prof,
                note=f"replayed from stored profiles; theta={ck.theta}; missing={len(missing)}",
            )
        )
    casc = json.loads(Path("outputs/diagnostics/cascade/cascade_report.json").read_text())
    for srow in casc["prompt_controlled"]["systems"]:
        if srow["system"].startswith("prompt-matched cascade"):
            rows.append(
                {
                    "system": "sufficiency cascade (both stages)",
                    "n": casc["prompt_controlled"]["n"],
                    "f1": srow["f1"],
                    "generation_input": srow["input_tokens"],
                    "generation_output": srow["output_tokens"],
                    "profiler_input": 0.0,
                    "profiler_output": 0.0,
                    "total_tokens": srow["total_tokens"],
                    "usd_per_1k_q": srow["usd_per_1k_q_no_cache"],
                    "index_summarizer_tokens_per_q": 0.0,
                    "usd_per_1k_q_incl_index": srow["usd_per_1k_q_no_cache"],
                    "note": "new sufficiency prompt; stage-1 flag tokens included; no cache",
                }
            )
    return {
        "prices_usd_per_mtok": {"input": PRICE_IN, "output": PRICE_OUT},
        "index_build": {
            "summarizer_input_tokens_per_q": round(idx_in, 1),
            "summarizer_output_tokens_per_q": round(idx_out, 1),
            "note": "one-off per paper, amortized over the dev questions; leaf_only needs "
            "no summaries (embeddings are local)",
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------- output


def write_md(out: Path, rep: dict) -> None:
    L = ["## A. Oracle headroom (dev)", ""]
    L.append(
        "| Budget (mean tokens) | Fixed F1 | Oracle F1 | Headroom [95% CI] | "
        "Profile-cell oracle − fixed | Pilot: share of headroom replicating |"
    )
    L.append("|---|---|---|---|---|---|")
    cell = {r["budget"]: r for r in rep["A"]["cell_oracles"]["full"]["rows"]}
    free = {r["budget"]: r for r in rep["A"]["cell_oracles"]["full (profiler tokens free)"]["rows"]}
    pil = {r["budget"]: r for r in rep["A"]["pilot_replicability"]["rows"]}
    for r in rep["A"]["rows"]:
        if r.get("headroom") is None:
            continue
        b, ci, c = r["budget"], r["headroom_ci"], cell.get(r["budget"], {})
        cc = c.get("ci")
        p = pil.get(b, {})
        L.append(
            f"| {b} | {r['fixed_f1']:.3f} | {r['oracle_f1']:.3f} | "
            f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}] | "
            + (f"{cc[0]:+.3f} [{cc[1]:+.3f}, {cc[2]:+.3f}]" if cc else "n/a")
            + " | "
            + (
                "{:+.3f} [{:+.3f}, {:+.3f}]".format(*free[b]["ci"])
                if free.get(b, {}).get("ci")
                else "n/a"
            )
            + f" | {p.get('share_replicating', 'n/a')} |"
        )
    L += ["", "## C. Token accounting (dev, per question)", ""]
    L.append(
        "| System | F1 | Gen in | Gen out | Profiler in | Profiler out | Total | "
        "$/1k q | Index tokens/q | $/1k q incl. index |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rep["C"]["rows"]:
        L.append(
            f"| {r['system']} | {r['f1']:.3f} | {r['generation_input']:.0f} | "
            f"{r['generation_output']:.0f} | {r['profiler_input']:.0f} | "
            f"{r['profiler_output']:.0f} | {r['total_tokens']:.0f} | {r['usd_per_1k_q']:.3f} | "
            f"{r['index_summarizer_tokens_per_q']:.0f} | {r['usd_per_1k_q_incl_index']:.3f} |"
        )
    (out / "tables.md").write_text("\n".join(L) + "\n")


def plots(out: Path, rep: dict) -> None:
    import matplotlib.pyplot as plt
    from make_dev_charts import INK, INK_2, MUTED, save, style

    style()
    A = rep["A"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    fx = np.array(A["paths"]["fixed"])
    orc = np.array(A["paths"]["oracle"])
    cel = np.array(A["cell_oracles"]["full"]["path"])
    ax.plot(orc[:, 0], orc[:, 1], color="#d6762a", lw=2, label="per-question oracle (true F1)")
    ax.plot(
        cel[:, 0],
        cel[:, 1],
        color="#3a9a5b",
        lw=2,
        label="profile-cell oracle (hindsight, incl. profiler tokens)",
    )
    free = np.array(A["cell_oracles"]["full (profiler tokens free)"]["path"])
    ax.plot(
        free[:, 0],
        free[:, 1],
        color="#3a9a5b",
        lw=1.5,
        ls="--",
        label="profile-cell oracle, profiler tokens not charged",
    )
    ax.plot(fx[:, 0], fx[:, 1], color=INK, lw=2, marker="s", ms=5, label="best fixed (mixing)")
    hon = [r for r in A["train_fit_cell_policy_dev"]]
    ax.plot(
        [r["dev_mean_tokens_incl_profiler"] for r in hon],
        [r["dev_f1"] for r in hon],
        color="#3a9a5b",
        ls=":",
        marker="o",
        ms=5,
        label="profile-cell policy fitted on train",
    )
    for r in rep["C"]["rows"]:
        if r["system"] in ("A1 router", "A2 router"):
            ax.plot(r["total_tokens"], r["f1"], "D", color="#2a78d6", ms=8)
            ax.annotate(
                r["system"],
                (r["total_tokens"], r["f1"]),
                textcoords="offset points",
                xytext=(8, 6) if r["system"] == "A2 router" else (8, -14),
                fontsize=8.5,
                color=INK_2,
            )
    ax.set_xlim(300, 2150)
    ax.set_xlabel("mean tokens per question (generation + profiler)")
    ax.set_ylabel("dev F1")
    ax.set_title(f"Oracle headroom over the 9-config catalog (dev, n={A['n']})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out, "headroom_curve")

    # profile interactions, train and dev
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels, y = [], 0
    for split, col in (("train", MUTED), ("dev", "#2a78d6")):
        for k, v in rep["B"][split]["interactions"].items():
            if v is None:
                continue
            d = v["difference"]
            ax.errorbar(
                d[0],
                y,
                xerr=[[d[0] - d[1]], [d[2] - d[0]]],
                fmt="o",
                color=col,
                ms=6,
                lw=2,
                capsize=0,
            )
            labels.append(f"[{split}] {k.split(': ')[0]}: {k.split(': ')[1]}")
            y += 1
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("difference in mean F1 gain between profile categories (95% doc. bootstrap)")
    ax.set_title("Do profile categories change which config helps? (RULES_V1 assumptions)")
    fig.tight_layout()
    save(fig, out, "profile_interactions")

    # best-config strategy share per granularity / n_facts (dev)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {
        "leaf_only": "#2a78d6",
        "collapsed": "#3a9a5b",
        "summary_only": "#d6762a",
        "tie": "#c9c8c3",
    }
    for ax, field in zip(axes, ("granularity", "n_facts"), strict=True):
        t = rep["B"]["dev"]["fields"][field]["best_config_contingency"]["strategy"]
        M = np.array(t["counts"], dtype=float)
        share = M / M.sum(1, keepdims=True)
        left = np.zeros(len(t["rows"]))
        for j, c in enumerate(t["cols"]):
            ax.barh(
                t["rows"],
                share[:, j],
                left=left,
                color=colors.get(c, MUTED),
                label=c,
                height=0.6,
                edgecolor="white",
                linewidth=1,
            )
            left += share[:, j]
        for i in range(len(t["rows"])):
            ax.annotate(
                f"n={int(M[i].sum())}",
                (1.01, i),
                fontsize=8,
                color=INK_2,
                va="center",
                annotation_clip=False,
            )
        ax.set_xlim(0, 1)
        dec = t["decisive_only"]
        ax.set_title(
            f"{field} (decisive questions: χ² p={dec['p_value']:.2g}, V={dec['cramers_v']})",
            fontsize=9.5,
        )
        ax.set_xlabel("share of questions: strategy with the highest F1 (tie = several)")
    axes[0].legend(
        fontsize=8, loc="upper center", bbox_to_anchor=(1.1, -0.18), ncol=3, frameon=False
    )
    fig.suptitle(
        "Which strategy wins, by profile category (dev; all-zero questions excluded)",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "best_strategy_by_profile")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    out = a.out or Path("outputs/diagnostics/section5") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    tq = Path("data/processed/qasper/v0.3/train/questions.jsonl")
    dq = Path("data/processed/qasper/v0.3/dev/questions.jsonl")
    labels_dev = latest("outputs/labels_dev/*/")
    train_run = latest("outputs/labels_train/*/")
    tr = Split(train_run, latest("outputs/profile_train/*/") / "profiles.jsonl", tq)
    dv = Split(labels_dev, latest("outputs/profile_dev/*/") / "profiles.jsonl", dq)
    dv.qs = [q for q in dv.qs if ORIG in dv.f1[q]]
    pilot = (
        Split(Path("outputs/pilot_catalog/20261109T193103Z"), None, dq),
        Split(Path("outputs/pilot_catalog/20261109T194522Z"), None, dq),
    )
    rep = {
        "note": "offline; no API calls; test split untouched; gold F1/evidence used only "
        "for oracles and diagnosis",
        "runs": {"train": str(train_run), "dev": str(labels_dev)},
        "A": headroom(tr, dv, pilot),
        "B": {"train": profile_power(tr, tq), "dev": profile_power(dv, dq)},
        "C": accounting(dv, labels_dev),
    }
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    write_md(out, rep)
    if not a.no_plots:
        plots(out, rep)
    print((out / "tables.md").read_text())
    for split in ("train", "dev"):
        print(f"== B {split} ==")
        for k, v in rep["B"][split]["interactions"].items():
            print(f"  {k}: {v and v['difference']}  (n {v and v['n_a']}/{v and v['n_b']})")
        for f, d in rep["B"][split]["fields"].items():
            t = d["best_config_contingency"]
            print(
                f"  {f}: "
                + " | ".join(
                    f"{u} decisive p={t[u]['decisive_only']['p_value']:.3g} "
                    f"V={t[u]['decisive_only']['cramers_v']} n={t[u]['decisive_only']['n']}"
                    for u in ("config", "strategy", "budget")
                    if t[u]["decisive_only"]
                )
                + "; best by mean: "
                + ", ".join(
                    f"{c}:{v['best_config_by_mean']}(n={v['n']})"
                    for c, v in d["categories"].items()
                )
            )
        print("  answer_form vs gold:", rep["B"][split]["answer_form_vs_gold_type"])
    print(json.dumps(rep["A"]["pilot_replicability"], indent=0))
    print(json.dumps(rep["A"]["train_fit_cell_policy_dev"], indent=0))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

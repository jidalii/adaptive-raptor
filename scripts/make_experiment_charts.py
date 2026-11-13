"""Charts for the summary-cap ablation and the sufficiency cascade (offline).

    uv run --group plots python scripts/make_experiment_charts.py \
        [--summary-cap outputs/diagnostics/summary_cap/summary_cap_report.json] \
        [--cascade outputs/diagnostics/cascade/cascade_report.json] [--out outputs/figures/next]

Either report may be omitted. Writes PNG + SVG:
  summary_cap_quality    summary length, share at the cap, share ending mid-sentence
  summary_cap_f1         F1 per config, cap 200 vs 512 (95% CI of the paired delta)
  summary_cap_gap        (strategy - leaf_only) at the same budget, cap 200 vs 512
  summary_cap_displace   summary share of the context and leaf-evidence coverage
  cascade_tradeoff       F1 vs tokens: leaf_only line, matched mixes, cascades
  cascade_signal         mean F1 gain from escalating, by stage-1 flag and by abstention
  cascade_deltas         primary cascade minus each comparator (95% CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_dev_charts import GRID, INK, INK_2, MUTED, SURFACE, save, style  # noqa: E402

ARM = {"control": ("cap 200", "#8a8984"), "treat": ("cap 512", "#2a78d6")}
CFG_ORDER = [
    "leaf_only@1000",
    "leaf_only@2000",
    "collapsed@1000",
    "collapsed@2000",
    "summary_only@1000",
    "summary_only@2000",
    "raptor_original",
]


def short(c: str) -> str:
    return c.replace("_only", "").replace("raptor_original", "original RAPTOR")


def summary_cap(rep: dict, out: Path) -> None:
    n = rep["n_questions"]
    # 1. quality
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    metrics = [
        ("median_tokens", "median summary tokens", "{:.0f}"),
        ("share_at_cap", "share at the cap", "{:.0%}"),
        ("share_unfinished", "share ending mid-sentence", "{:.0%}"),
    ]
    for ax, (k, title, fmt) in zip(axes, metrics, strict=True):
        vals = [rep[f"summaries_{arm}"][k] for arm in ARM]
        ax.bar(range(2), vals, color=[c for _, c in ARM.values()], width=0.6)
        for i, v in enumerate(vals):
            ax.annotate(
                fmt.format(v),
                (i, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK_2,
            )
        ax.set_xticks(range(2), [lbl for lbl, _ in ARM.values()])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.25 or 1)
    fig.suptitle(
        f"Summary completeness ({rep['summaries_treat']['summaries']} summaries, "
        f"{rep['n_papers']} papers)",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "summary_cap_quality")

    # 2. F1 per config
    per = rep["per_config"]
    cfgs = [c for c in CFG_ORDER if c in per]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.38
    for k, arm in enumerate(ARM):
        xs = [i + (k - 0.5) * w for i in range(len(cfgs))]
        vals = [per[c][arm]["f1"] for c in cfgs]
        ax.bar(xs, vals, width=w - 0.04, color=ARM[arm][1], label=ARM[arm][0])
    for i, c in enumerate(cfgs):
        d = per[c]["delta_f1_treat_minus_control"]
        top = max(per[c]["control"]["f1"], per[c]["treat"]["f1"])
        ax.annotate(
            f"{d[0]:+.3f}\n[{d[1]:+.3f}, {d[2]:+.3f}]",
            (i, top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK_2,
        )
    ax.axhline(per["leaf_only@1000"]["control"]["f1"], color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(cfgs)), [short(c) for c in cfgs], fontsize=9, rotation=20, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("answer F1")
    ax.set_ylim(0, max(per[c][a]["f1"] for c in cfgs for a in ARM) * 1.3)
    ax.set_title(
        f"F1 by config: {ARM['control'][0]} vs {ARM['treat'][0]}\n"
        f"(n={n}; label: Δ treat−control, 95% CI; dashed: leaf@1000)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    save(fig, out, "summary_cap_f1")

    # 3. gap to leaf
    gaps = rep["gap_to_leaf"]
    rows = [
        c
        for c in ("collapsed@1000", "collapsed@2000", "summary_only@1000", "summary_only@2000")
        if c in gaps
    ]
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(rows) + 1.3))
    for i, c in enumerate(rows):
        for k, (arm, key) in enumerate(
            (("control", "gap_to_leaf_control"), ("treat", "gap_to_leaf_treat"))
        ):
            m, lo, hi = gaps[c][key]
            y = i + (k - 0.5) * 0.3
            ax.plot([lo, hi], [y, y], color=ARM[arm][1], lw=2, solid_capstyle="round")
            ax.scatter(
                m,
                y,
                s=50,
                color=ARM[arm][1],
                zorder=3,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=ARM[arm][0] if i == 0 else None,
            )
        did = gaps[c]["difference_in_differences"]
        ax.annotate(
            f"DiD {did[0]:+.3f} [{did[1]:+.3f}, {did[2]:+.3f}]",
            (max(gaps[c]["gap_to_leaf_treat"][2], gaps[c]["gap_to_leaf_control"][2]), i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [f"{short(c)} − leaf@{c.split('@')[1]}" for c in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ F1 vs leaf_only at the same budget (95% paired CI)")
    ax.set_title("Does hierarchical retrieval catch up with leaf-only?")
    xmin = min(gaps[c][k][1] for c in rows for k in ("gap_to_leaf_control", "gap_to_leaf_treat"))
    xmax = max(gaps[c][k][2] for c in rows for k in ("gap_to_leaf_control", "gap_to_leaf_treat"))
    ax.set_xlim(xmin - 0.01, xmax + 0.09)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    save(fig, out, "summary_cap_gap")

    # 4. displacement
    hier = [c for c in ("collapsed@1000", "collapsed@2000", "raptor_original") if c in per]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title in (
        (a, "summary_node_share", "summary share of retrieved nodes"),
        (b, "leaf_evidence_coverage", "gold evidence found in leaf chunks"),
    ):
        for k, arm in enumerate(ARM):
            xs = [i + (k - 0.5) * w for i in range(len(hier))]
            vals = [per[c][arm][key] for c in hier]
            ax.bar(xs, vals, width=w - 0.04, color=ARM[arm][1], label=ARM[arm][0])
            for x, v in zip(xs, vals, strict=True):
                ax.annotate(
                    f"{v:.2f}",
                    (x, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=INK_2,
                )
        if key == "leaf_evidence_coverage":
            for c, ls in (("leaf_only@1000", "--"), ("leaf_only@2000", ":")):
                ax.axhline(per[c]["treat"][key], color=MUTED, lw=1, ls=ls, label=f"{short(c)}")
        ax.set_xticks(range(len(hier)), [short(c) for c in hier], fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.15)
    b.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(
        "Do longer summaries displace leaf evidence? (same budget cap)",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "summary_cap_displace")


def cascade(rep: dict, out: Path) -> None:
    rows = {r["system"]: r for r in rep["dev_primary"]}
    sec = rep["dev_secondary_all_candidates"]
    # 1. trade-off
    fig, ax = plt.subplots(figsize=(8.5, 5))
    leaf = [
        rows[k]
        for k in ("fixed leaf_only@400", "fixed leaf_only@1000", "fixed leaf_only@2000")
        if k in rows
    ]
    ax.plot(
        [r["total_tokens"] for r in leaf],
        [r["f1"] for r in leaf],
        color="#2a78d6",
        lw=2,
        marker="o",
        ms=8,
        mec=SURFACE,
        mew=2,
        label="fixed leaf_only (400/1000/2000)",
        zorder=3,
    )
    ax.scatter(
        [r["total_tokens"] for r in sec],
        [r["f1"] for r in sec],
        s=25,
        color=MUTED,
        alpha=0.7,
        label="other cascade candidates (exploratory)",
        zorder=2,
    )
    marks = [
        ("abstention cascade (run A, 400->2000)", "X", "#eb6834", "abstention cascade"),
        ("primary cascade", "*", INK, "primary cascade (chosen on train)"),
    ]
    for key, m, color, label in marks:
        r = rows[key]
        ax.scatter(
            r["total_tokens"],
            r["f1"],
            marker=m,
            s=220 if m == "*" else 110,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=5,
            label=label,
        )
    ax.set_xlabel("total tokens per question (both calls, generation in + out)")
    ax.set_ylabel("answer F1 (dev)")
    ax.set_title(f"Cascades vs fixed leaf-only budgets (dev, n={rep['n']['dev']})")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, out, "cascade_tradeoff")

    # 2. signal quality (stage 1 = 400 -> 2000)
    sq = next(
        x
        for x in rep["signal_quality"]["dev"]
        if x["stage1"] == "leaf_only@400" and x["target"] == "leaf_only@2000"
    )
    levels = [
        lv for lv in ("sufficient", "partial", "insufficient", "None") if lv in sq["by_level"]
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    vals = [sq["by_level"][lv]["mean_gain"] for lv in levels]
    a.bar(
        range(len(levels)),
        vals,
        color=["#1baf7a", "#eda100", "#eb6834", MUTED][: len(levels)],
        width=0.6,
    )
    for i, lv in enumerate(levels):
        a.annotate(
            f"{vals[i]:+.3f}\n({sq['by_level'][lv]['share']:.0%} of q)",
            (i, vals[i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_2,
        )
    a.axhline(0, color=MUTED, lw=1)
    a.set_xticks(range(len(levels)), ["flag missing" if lv == "None" else lv for lv in levels])
    a.set_ylabel("mean F1 gain, leaf@2000 − stage 1")
    a.set_title("Gain from escalating, by stage-1 flag", fontsize=10)
    a.grid(axis="x", visible=False)
    a.set_ylim(min(0, min(vals)) - 0.02, max(vals) * 1.35)
    aucs = [
        ("flag", sq["auc_flag"]),
        ("abstention", sq["auc_abstain"]),
        ("flag + abstention", sq["auc_flag_plus_abstain"]),
    ]
    b.barh(
        range(3), [v or 0 for _, v in aucs], color=["#2a78d6", "#eb6834", "#6250d6"], height=0.55
    )
    for i, (_, v) in enumerate(aucs):
        b.annotate(
            f"{v:.3f}" if v else "n/a",
            (v or 0, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    b.axvline(0.5, color=MUTED, lw=1, ls="--")
    b.set_yticks(range(3), [k for k, _ in aucs])
    b.invert_yaxis()
    b.set_xlim(0.4, 1)
    b.grid(axis="y", visible=False)
    b.set_title("AUC: predicts 'escalating helps'", fontsize=10)
    fig.suptitle(
        "Is the sufficiency flag a better trigger than abstention? (dev, 400 → 2000)",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "cascade_signal")

    # 3. deltas
    comps = [r for r in rep["dev_primary"] if "primary_minus_this" in r]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(comps) + 1.3))
    for i, r in enumerate(comps):
        m, lo, hi = r["primary_minus_this"]
        ax.plot([lo, hi], [i, i], color=INK_2, lw=2, solid_capstyle="round")
        ax.scatter(m, i, s=55, color=INK, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(
            f"{m:+.3f}  ({r['total_tokens']:,.0f} tok, ${r['usd_per_1k_q_no_cache']:.3f}/1k q)",
            (hi, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(comps)), [r["system"] for r in comps], fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    p = rows["primary cascade"]
    ax.set_xlabel(
        f"Δ F1, primary cascade − comparator (primary: F1 {p['f1']:.3f}, "
        f"{p['total_tokens']:,.0f} tok)"
    )
    ax.set_title("Primary cascade vs comparators (dev, 95% paired CI)")
    lo = min(r["primary_minus_this"][1] for r in comps)
    hi = max(r["primary_minus_this"][2] for r in comps)
    ax.set_xlim(lo - 0.01, hi + 0.12)
    save(fig, out, "cascade_deltas")


ARM3 = {
    "cap200": ("cap 200, upstream prompt", "#8a8984"),
    "cap512": ("cap 512, upstream prompt", "#86b6ef"),
    "dense": ("cap 512, evidence-dense prompt", "#2a78d6"),
}


def summary_arms(rep: dict, out: Path) -> None:
    arms = [a for a in ARM3 if a in rep["arms"]]
    # 1. summary quality, 5 small panels
    metrics = [
        (lambda a: a["summaries"]["median_tokens"], "median summary tokens", "{:.0f}"),
        (lambda a: a["summaries"]["share_at_cap"], "share at the cap", "{:.0%}"),
        (lambda a: a["summaries"]["share_unfinished"], "share ending mid-sentence", "{:.0%}"),
        (lambda a: a["best_summary_evidence_recall"], "evidence recall (best summary)", "{:.2f}"),
        (
            lambda a: a["evidence_density_per_100_tokens"],
            "evidence recall per 100 tokens",
            "{:.2f}",
        ),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    for ax, (get, title, fmt) in zip(axes, metrics, strict=True):
        vals = [get(rep["arms"][a]) for a in arms]
        ax.bar(range(len(arms)), vals, color=[ARM3[a][1] for a in arms], width=0.65)
        for i, v in enumerate(vals):
            ax.annotate(
                fmt.format(v),
                (i, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK_2,
            )
        ax.set_xticks(range(len(arms)), arms, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.25 or 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARM3[a][1]) for a in arms]
    fig.legend(
        handles,
        [ARM3[a][0] for a in arms],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize=9,
        frameon=False,
    )
    fig.suptitle(
        f"Summary quality by arm ({rep['n_papers']} papers, {rep['n_questions']} questions)",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "summary_arms_quality")

    # 2. F1 and leaf-evidence coverage per config
    cfgs = [c for c in CFG_ORDER if c in rep["arms"][arms[0]]["per_config"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    w = 0.8 / len(arms)
    for k, arm in enumerate(arms):
        xs = [i + (k - (len(arms) - 1) / 2) * w for i in range(len(cfgs))]
        pc = rep["arms"][arm]["per_config"]
        a1.bar(
            xs, [pc[c]["f1"] for c in cfgs], width=w - 0.03, color=ARM3[arm][1], label=ARM3[arm][0]
        )
        a2.bar(
            xs, [pc[c]["leaf_evidence_coverage"] for c in cfgs], width=w - 0.03, color=ARM3[arm][1]
        )
        for x, c in zip(xs, cfgs, strict=True):
            a1.annotate(
                f"{pc[c]['f1']:.2f}",
                (x, pc[c]["f1"]),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=INK_2,
            )
    a1.set_ylabel("answer F1")
    a1.legend(fontsize=8.5, loc="upper left", ncol=3)
    a1.set_ylim(0, max(rep["arms"][a]["per_config"][c]["f1"] for a in arms for c in cfgs) * 1.3)
    a2.set_ylabel("leaf-evidence coverage")
    a2.set_ylim(0, 1.1)
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
    a2.set_xticks(range(len(cfgs)), [short(c) for c in cfgs], fontsize=9, rotation=20, ha="right")
    a1.set_title("F1 and leaf evidence kept in the context, by summary arm (same budgets)")
    save(fig, out, "summary_arms_f1")

    # 3. reference vs others: paired deltas for the hierarchical configs
    refn = rep["reference"]
    keys = [
        "best_summary_evidence_recall",
        "f1 collapsed@1000",
        "f1 collapsed@2000",
        "f1 raptor_original",
        "DiD collapsed@1000",
        "DiD collapsed@2000",
    ]
    others = list(rep["reference_minus"])
    fig, ax = plt.subplots(figsize=(9.5, 0.75 * len(keys) + 1.4))
    for i, k in enumerate(keys):
        for j, o in enumerate(others):
            m, lo, hi = rep["reference_minus"][o][k]
            y = i + (j - (len(others) - 1) / 2) * 0.28
            ax.plot([lo, hi], [y, y], color=ARM3[o][1], lw=2, solid_capstyle="round")
            ax.scatter(
                m,
                y,
                s=45,
                color=ARM3[o][1],
                zorder=3,
                edgecolor=SURFACE,
                linewidth=1.2,
                label=f"{refn} − {o}" if i == 0 else None,
            )
            ax.annotate(
                f"{m:+.3f}",
                (hi, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=INK_2,
            )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(keys)), keys, fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired difference (95% document bootstrap)")
    ax.set_title(f"Evidence-dense prompt ({refn}) vs the other arms")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out, "summary_arms_deltas")


def cascade_prompt_control(pc: dict, out: Path) -> None:
    rows = pc["systems"]
    cas = rows[0]
    comps = [r for r in rows if "cascade_minus_this" in r]
    fig, (a, b) = plt.subplots(
        1,
        2,
        figsize=(13, 0.5 * len(comps) + 1.8),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.55},
    )
    for i, r in enumerate(comps):
        m, lo, hi = r["cascade_minus_this"]
        a.plot([lo, hi], [i, i], color=INK_2, lw=2, solid_capstyle="round")
        a.scatter(m, i, s=50, color=INK, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        a.annotate(
            f"{m:+.3f}",
            (hi, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    a.axvline(0, color=MUTED, lw=1)
    a.set_yticks(
        range(len(comps)),
        [
            f"{r['system']}  ({r['total_tokens']:,.0f} tok, ${r['usd_per_1k_q_no_cache']:.3f}/1k)"
            for r in comps
        ],
        fontsize=8,
    )
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.set_xlabel(
        f"Δ F1, prompt-matched cascade − comparator (cascade: F1 {cas['f1']:.3f}, "
        f"{cas['total_tokens']:,.0f} tok)"
    )
    a.set_title("Adaptive effect: cascade vs fixed baselines (dev, 95% CI)")
    eff = pc["prompt_effect_new_minus_old"]
    cs = list(eff)
    for i, c in enumerate(cs):
        m, lo, hi = eff[c]
        b.plot([lo, hi], [i, i], color="#2a78d6", lw=2, solid_capstyle="round")
        b.scatter(m, i, s=50, color="#2a78d6", zorder=3, edgecolor=SURFACE, linewidth=1.2)
        b.annotate(
            f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            (hi, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    b.axvline(0, color=MUTED, lw=1)
    b.set_yticks(range(len(cs)), [short(c) for c in cs])
    b.invert_yaxis()
    b.grid(axis="y", visible=False)
    b.set_xlim(min(eff[c][1] for c in cs) - 0.01, max(eff[c][2] for c in cs) + 0.05)
    b.set_xlabel("Δ F1, sufficiency prompt − original prompt")
    b.set_title("Prompt effect (same context)")
    save(fig, out, "cascade_prompt_control")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--summary-cap",
        type=Path,
        default=Path("outputs/diagnostics/summary_cap/summary_cap_report.json"),
    )
    p.add_argument(
        "--cascade", type=Path, default=Path("outputs/diagnostics/cascade/cascade_report.json")
    )
    p.add_argument(
        "--arms",
        type=Path,
        default=Path("outputs/diagnostics/summary_arms_pilot/summary_arms_report.json"),
    )
    p.add_argument("--out", type=Path, default=Path("outputs/figures/next"))
    p.add_argument("--control-label", default="cap 200", help="summary-cap control arm label")
    p.add_argument("--treat-label", default="cap 512", help="summary-cap treatment arm label")
    a = p.parse_args()
    ARM["control"] = (a.control_label, ARM["control"][1])
    ARM["treat"] = (a.treat_label, ARM["treat"][1])
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    plt.rcParams["grid.color"] = GRID
    if a.summary_cap.exists():
        summary_cap(json.loads(a.summary_cap.read_text()), a.out)
    else:
        print(f"skip summary-cap charts: {a.summary_cap} not found")
    if a.cascade.exists():
        crep = json.loads(a.cascade.read_text())
        cascade(crep, a.out)
        if "prompt_controlled" in crep:
            cascade_prompt_control(crep["prompt_controlled"], a.out)
    if a.arms.exists():
        summary_arms(json.loads(a.arms.read_text()), a.out)
    else:
        print(f"skip cascade charts: {a.cascade} not found")


if __name__ == "__main__":
    main()

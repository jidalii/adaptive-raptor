"""Original RAPTOR (baseline) vs RAPTOR + optimizer, on dev (offline; calls no API).

    uv run --group plots python scripts/make_baseline_charts.py
    uv run --group plots python scripts/make_baseline_charts.py --a1 <router_train run> \
        --a2 <router_train run> --dev-labels <labels.jsonl> --dev-profiles <profiles.jsonl>

Every router is replayed from its frozen checkpoint and the stored dev profiles; its
answer on each question is the recorded answer of the config it picks. Systems:
  original RAPTOR       upstream collapsed-tree retrieval (the baseline)
  A1 (θ frozen)         profile-table router at the θ chosen in router_train
  A1 (θ = baseline cost)  same router, θ re-set so its dev cost is closest to the baseline
  A2 rules              rule-map ablation (no cost knob)
Writes PNG + SVG to --out:
  baseline_headline     F1 (95% paired CI of the difference) and total tokens per question
  baseline_by_type      F1 by gold answer type
  baseline_tokens       where the tokens go: context, prompt overhead, output, profiler
  baseline_win_loss     per question: optimizer better / same / worse than the baseline
Also writes baseline_comparison.csv with the numbers behind the charts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adaptive_raptor.config.loader import load_config  # noqa: E402
from adaptive_raptor.evaluation.research import (  # noqa: E402
    paired_document_bootstrap,
    score_fixed,
    score_routed,
)
from adaptive_raptor.experiments import _label_table, _replay  # noqa: E402
from adaptive_raptor.optimizer import Router, load_checkpoint  # noqa: E402
from adaptive_raptor.optimizer.profiler import ProfileStore, build_profiler  # noqa: E402
from adaptive_raptor.profiling import merge_label_tables  # noqa: E402

# Validated categorical slots (light): baseline in neutral ink, optimizer systems in slots 1-3.
COLORS = {
    "original RAPTOR": "#52514e",
    "A1 (θ frozen)": "#2a78d6",
    "A1 (θ = baseline cost)": "#eb6834",
    "A2 rules": "#1baf7a",
}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
WIN, TIE, LOSS = "#2a78d6", "#d3d2cc", "#eb6834"  # diverging poles + neutral midpoint
BASE = "original RAPTOR"


def latest(pattern: str) -> Path:
    runs = sorted(Path().glob(pattern))
    if not runs:
        raise SystemExit(f"no run matches {pattern}; pass the path explicitly")
    return runs[-1]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png/.svg")


def baseline_cost_theta(run: Path) -> float | None:
    with (run / "operating_points.csv").open() as f:
        for row in csv.DictReader(f):
            if row["fixed"] == "raptor_original" and row["theta"]:
                return float(row["theta"])
    return None


def build_systems(a1: Path, a2: Path, labels: Path, profiles: Path, prices):
    records, _ = merge_label_tables([labels])
    store = ProfileStore.load([profiles])
    cfg = load_config(a1 / "resolved_config.yaml", [])
    catalog = cfg.retrieval.catalog()
    cell = load_checkpoint(a1 / "router").artifacts["label_cell"]
    table = _label_table(records, cell, [c.id for c in catalog])
    original = _label_table(records, cell, ["raptor_original"])
    table = {q: row for q, row in table.items() if q in original}
    q2d = {q: next(iter(row.values())).doc_id for q, row in table.items()}
    atype = {q: next(iter(row.values())).answer_type for q, row in table.items()}

    systems = {BASE: score_fixed(BASE, {"raptor_original": 1.0}, original, prices)}
    variants = [
        (a1, "A1 (θ frozen)", None),
        (a1, "A1 (θ = baseline cost)", baseline_cost_theta(a1)),
        (a2, "A2 rules", None),
    ]
    thetas = {}
    for run, name, theta in variants:
        rcfg = load_config(run / "resolved_config.yaml", [])
        router = Router.from_checkpoint(
            run / "router", rcfg.optimizer, catalog, build_profiler(rcfg.optimizer.profiler)
        )
        if theta is not None:
            router.theta = theta
        thetas[name] = router.theta
        choices, prof, fallback, missing = _replay(router, table, store)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} dev questions have no stored profile")
        systems[name] = score_routed(name, choices, table, prof, fallback, prices)
    return systems, q2d, atype, thetas


def summary_rows(systems, q2d, thetas):
    base = systems[BASE]
    rows = []
    for name, s in systems.items():
        row = {"system": name, "theta": thetas.get(name), **s.row()}
        if name != BASE:
            delta = {q: s.per_query_f1[q] - base.per_query_f1[q] for q in base.per_query_f1}
            mean, lo, hi = paired_document_bootstrap(delta, q2d, 2000, 0.95)
            d = np.array(list(delta.values()))
            row.update(
                delta_f1_vs_baseline=round(mean, 4),
                ci_lo=round(lo, 4),
                ci_hi=round(hi, 4),
                better=int((d > 1e-9).sum()),
                same=int((np.abs(d) <= 1e-9).sum()),
                worse=int((d < -1e-9).sum()),
            )
        rows.append(row)
    return rows


def headline(rows, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw={"wspace": 0.12})
    names = [r["system"] for r in rows]
    ys = range(len(rows))
    base_f1 = rows[0]["answer_f1"]
    for i, r in enumerate(rows):
        c = COLORS[r["system"]]
        a.barh(i, r["answer_f1"], color=c, height=0.62)
        if "ci_lo" in r:  # CI of the paired difference, drawn around this system's F1
            a.plot([base_f1 + r["ci_lo"], base_f1 + r["ci_hi"]], [i, i], color=INK, lw=1.5)
            txt = f"{r['answer_f1']:.3f}  ({r['delta_f1_vs_baseline']:+.3f})"
        else:
            txt = f"{r['answer_f1']:.3f}  baseline"
        a.annotate(
            txt,
            (max(r["answer_f1"], r.get("ci_hi", -1) + base_f1, base_f1), i),  # past the dashed line
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
        b.barh(i, r["total_tokens"], color=c, height=0.62)
        pct = "" if i == 0 else f"  ({r['total_tokens'] / rows[0]['total_tokens'] - 1:+.0%})"
        b.annotate(
            f"{r['total_tokens']:,.0f}{pct}",
            (r["total_tokens"], i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    a.set_yticks(list(ys), names)
    a.invert_yaxis()  # shared with b
    b.tick_params(axis="y", left=False)
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
    a.axvline(base_f1, color=MUTED, lw=1, ls="--")
    a.set_xlim(0, max(r["answer_f1"] for r in rows) * 1.45)
    b.set_xlim(0, max(r["total_tokens"] for r in rows) * 1.35)
    a.set_xlabel("answer F1 (black line: 95% paired CI vs baseline)")
    b.set_xlabel("total tokens per question (generation + profiler)")
    a.set_title("Answer F1 on dev")
    b.set_title("Cost per question")
    fig.suptitle(
        f"Original RAPTOR vs RAPTOR + optimizer (dev, n={rows[0]['n']})",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(top=0.8)
    save(fig, out, "baseline_headline")


def by_type(systems, atype, out: Path) -> None:
    types = ["extractive", "abstractive", "boolean", "none"]
    n = {t: sum(1 for q in atype if atype[q] == t) for t in types}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(systems)
    for k, (name, s) in enumerate(systems.items()):
        vals = []
        for t in types:
            qs = [q for q in s.per_query_f1 if atype[q] == t]
            vals.append(sum(s.per_query_f1[q] for q in qs) / len(qs) if qs else 0)
        xs = [i + (k - (len(systems) - 1) / 2) * w for i in range(len(types))]
        ax.bar(xs, vals, width=w - 0.03, color=COLORS[name], label=name)
        for x, v in zip(xs, vals, strict=True):
            ax.annotate(
                f"{v:.2f}",
                (x, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=INK_2,
            )
    ax.set_xticks(range(len(types)), [f"{t}\n(n={n[t]})" for t in types])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("answer F1")
    ax.set_title("F1 by gold answer type (dev)")
    ax.legend(ncol=4, loc="upper left", fontsize=8.5, bbox_to_anchor=(0, -0.18))
    save(fig, out, "baseline_by_type")


def tokens(rows, out: Path) -> None:
    parts = [
        ("retrieved context", "context_tokens", "#2a78d6"),
        ("prompt + question", None, "#86b6ef"),
        ("answer output", "generation_output_tokens", "#eb6834"),
        ("profiler (optimizer)", "profiler_tokens", "#1baf7a"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, r in enumerate(rows):
        left = 0.0
        for label, key, color in parts:
            v = r["generation_input_tokens"] - r["context_tokens"] if key is None else r[key]
            ax.barh(
                i,
                v,
                left=left,
                color=color,
                height=0.6,
                edgecolor=SURFACE,
                linewidth=2,
                label=label if i == 0 else None,
            )
            if v > 120:
                ax.text(
                    left + v / 2,
                    i,
                    f"{v:,.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if color in ("#2a78d6", "#eb6834") else INK,
                )
            left += v
        ax.annotate(
            f"{left:,.0f}",
            (left, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_yticks(range(len(rows)), [r["system"] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("tokens per question (mean)")
    ax.set_title("Where the tokens go (dev)")
    ax.legend(ncol=4, loc="upper left", fontsize=8.5, bbox_to_anchor=(0, -0.22))
    save(fig, out, "baseline_tokens")


def win_loss(rows, out: Path) -> None:
    rows = [r for r in rows if "better" in r]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.4))
    for i, r in enumerate(rows):
        n = r["better"] + r["same"] + r["worse"]
        segs = [(-r["worse"] / n, LOSS, "worse"), (r["better"] / n, WIN, "better")]
        same = r["same"] / n
        ax.barh(
            i,
            -same / 2,
            left=0,
            color=TIE,
            height=0.6,
            edgecolor=SURFACE,
            linewidth=2,
            label="same F1" if i == 0 else None,
        )
        ax.barh(i, same / 2, left=0, color=TIE, height=0.6, edgecolor=SURFACE, linewidth=2)
        for v, color, label in segs:
            start = -same / 2 if v < 0 else same / 2
            ax.barh(
                i,
                v,
                left=start,
                color=color,
                height=0.6,
                edgecolor=SURFACE,
                linewidth=2,
                label=f"optimizer {label}" if i == 0 else None,
            )
            ax.text(
                start + v / 2,
                i,
                f"{abs(v):.0%}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white",
            )
        ax.text(0, i, f"{same:.0%}", ha="center", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [r["system"] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["100%", "50%", "0", "50%", "100%"])
    ax.set_xlabel("share of dev questions: ← optimizer worse    optimizer better →")
    ax.set_title("Per question: optimizer vs original RAPTOR (dev)")
    ax.legend(ncol=3, loc="upper left", fontsize=8.5, bbox_to_anchor=(0, -0.3))
    save(fig, out, "baseline_win_loss")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--a1", type=Path, help="router_train run dir for llm_profile_table")
    p.add_argument("--a2", type=Path, help="router_train run dir for llm_profile_rules")
    p.add_argument("--dev-labels", type=Path, help="labels.jsonl that includes raptor_original")
    p.add_argument("--dev-profiles", type=Path)
    p.add_argument("--price-in", type=float, default=0.15, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=0.60, help="USD per 1M output tokens")
    p.add_argument("--out", type=Path, default=Path("outputs/figures/t12_dev"))
    a = p.parse_args()
    a1 = a.a1 or latest("outputs/router_train_llm_profile_table/*/")
    a2 = a.a2 or latest("outputs/router_train_llm_profile_rules/*/")
    labels = a.dev_labels or latest("outputs/labels_dev/*/") / "labels.jsonl"
    profiles = a.dev_profiles or latest("outputs/profile_dev/*/") / "profiles.jsonl"
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"A1={a1}\nA2={a2}\nlabels={labels}\nprofiles={profiles}")

    systems, q2d, atype, thetas = build_systems(a1, a2, labels, profiles, (a.price_in, a.price_out))
    rows = summary_rows(systems, q2d, thetas)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (a.out / "baseline_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out / 'baseline_comparison.csv'}")
    for r in rows:
        print(
            json.dumps(
                {
                    k: r.get(k)
                    for k in (
                        "system",
                        "theta",
                        "answer_f1",
                        "total_tokens",
                        "usd_per_query",
                        "delta_f1_vs_baseline",
                        "ci_lo",
                        "ci_hi",
                        "fallback_rate",
                    )
                }
            )
        )
    style()
    headline(rows, a.out)
    by_type(systems, atype, a.out)
    tokens(rows, a.out)
    win_loss(rows, a.out)


if __name__ == "__main__":
    main()

"""Charts for the T12 dev evaluation (offline: reads run outputs, calls no API).

    uv run --group plots python scripts/make_dev_charts.py
    uv run --group plots python scripts/make_dev_charts.py \
        --a1 outputs/router_train_llm_profile_table/<run> \
        --a2 outputs/router_train_llm_profile_rules/<run> \
        --dev-labels outputs/labels_dev/<run>/labels.jsonl \
        --dev-profiles outputs/profile_dev/<run>/profiles.jsonl --out outputs/figures/t12_dev

Defaults pick the latest run of each kind under outputs/. Writes PNG + SVG:
  pareto        F1 vs total tokens: fixed configs, original RAPTOR, A1 theta sweep, A1/A2
  deltas        paired-bootstrap delta F1 (95% CI) of each router vs its comparators
  profile_groups  dev F1 of each config within each LLM-profile group (heatmap)
  config_choice   how often A1 / A2 route to each config
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots 1-3 (all-pairs safe; aqua < 3:1 -> lines are direct-labeled).
STRATEGY_COLOR = {"leaf_only": "#2a78d6", "summary_only": "#eb6834", "collapsed": "#1baf7a"}
STRATEGY_MARKER = {"leaf_only": "o", "summary_only": "s", "collapsed": "^"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CONFIGS = [
    f"{s}@{b}" for s in ("leaf_only", "collapsed", "summary_only") for b in (400, 1000, 2000)
]


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
            "axes.titlesize": 12,
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


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def pareto(a1: Path, a2: Path, out: Path) -> None:
    r1 = json.loads((a1 / "dev_report.json").read_text())
    r2 = json.loads((a2 / "dev_report.json").read_text())
    fixed = {p["config"]: (p["total_tokens"], p["f1"]) for p in r1["fixed_points"]}
    fig, ax = plt.subplots(figsize=(8, 5.2))

    sweep = read_csv(a1 / "theta_sweep.csv")
    xs = [float(r["mean_total_tokens"]) for r in sweep]
    ys = [float(r["mean_f1"]) for r in sweep]
    ax.plot(xs, ys, color=MUTED, lw=1.5, ls="--", zorder=2, label="A1 θ sweep")

    for strat, color in STRATEGY_COLOR.items():
        pts = sorted(v for k, v in fixed.items() if k.startswith(strat))
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            lw=2,
            marker=STRATEGY_MARKER[strat],
            ms=8,
            mec=SURFACE,
            mew=2,
            zorder=3,
            label=strat,
        )
        x, y = pts[-1]
        dy = -12 if strat == "collapsed" else 0  # keep clear of the A1 sweep line
        ax.annotate(
            strat,
            (x, y),
            xytext=(8, dy),
            textcoords="offset points",
            va="center",
            color=INK_2,
            fontsize=9,
        )

    orig = r1["raptor_original"]
    systems = [
        (orig["total_tokens"], orig["answer_f1"], "D", "original RAPTOR"),
        (
            r1["router"]["total_tokens"],
            r1["router"]["answer_f1"],
            "*",
            f"A1 table (θ={r1['theta']})",
        ),
        (r2["router"]["total_tokens"], r2["router"]["answer_f1"], "X", "A2 rules"),
    ]
    offsets = {"original RAPTOR": (8, 6), "A1": (8, -12), "A2 rules": (-10, 10)}
    for x, y, m, name in systems:
        if m == "X":  # A2 sits almost on A1: open ring so the A1 star stays visible inside
            ax.scatter(
                x,
                y,
                marker="o",
                s=260,
                facecolor="none",
                edgecolor=INK,
                linewidth=1.5,
                zorder=5,
                label=name,
            )
        else:
            ax.scatter(
                x,
                y,
                marker=m,
                s=170 if m == "*" else 80,
                color=INK,
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=4,
                label=name,
            )
        dx, dy = offsets.get(name.split(" (")[0].split(" table")[0], (8, 6))
        ax.annotate(
            name,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            color=INK,
            fontsize=9,
        )

    ax.set_xlabel("Total tokens per question (generation in + out + profiler)")
    ax.set_ylabel("Answer F1 (dev)")
    ax.set_title(f"F1 vs cost on dev (n={r1['router']['n']} questions)")
    ax.set_xlim(right=max(xs + [p[0] for p in fixed.values()]) * 1.15)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    fig.text(
        0.01,
        -0.02,
        "θ and comparators are chosen on these dev questions, so router numbers are optimistic.",
        color=MUTED,
        fontsize=8,
    )
    save(fig, out, "pareto")


def deltas(a1: Path, a2: Path, out: Path) -> None:
    rows = []
    for run, name in ((a1, "A1 table"), (a2, "A2 rules")):
        rep = json.loads((run / "dev_report.json").read_text())
        for d in rep["deltas_vs_matched_fixed"]:
            comp = next(iter(d["comparator"]["weights"]))
            rows.append((f"{name} vs {comp}\n({d['matching']})", d))
        # Only A1 has a cost knob (θ); A2's 0/1 rule probabilities cannot be cost-matched.
        for op in read_csv(run / "operating_points.csv") if run == a1 else []:
            if op["fixed"] == "raptor_original" and op["delta_f1"]:
                lo, hi = json.loads(op["ci"])
                d = {
                    "delta_f1": float(op["delta_f1"]),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "verdict": op["verdict"],
                }
                rows.append((f"{name} (θ={op['theta']}) vs original RAPTOR\n(matched cost)", d))
    rows.reverse()
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.2))
    for i, (_label, d) in enumerate(rows):
        ax.plot([d["ci_lo"], d["ci_hi"]], [i, i], color=INK_2, lw=2, solid_capstyle="round")
        ax.scatter(d["delta_f1"], i, s=60, color=INK, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(
            f"{d['delta_f1']:+.3f}  {d['verdict']}",
            (d["ci_hi"], i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
            fontsize=8.5,
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ answer F1, router − comparator (95% paired bootstrap CI by paper)")
    ax.set_title("Router vs fixed configs at matched cost (dev)")
    lo = min(d["ci_lo"] for _, d in rows)
    hi = max(d["ci_hi"] for _, d in rows)
    ax.set_xlim(lo - 0.01, hi + 0.06)
    save(fig, out, "deltas")


def profile_groups(labels: Path, profiles: Path, out: Path) -> None:
    f1: dict[str, dict[str, float]] = defaultdict(dict)
    for line in labels.open():
        r = json.loads(line)
        f1[r["question_id"]][r["config_id"]] = r["answer_f1"]
    prof = {}
    for line in profiles.open():
        r = json.loads(line)
        if r.get("profile"):
            prof[r["question_id"]] = r["profile"]
    cols = CONFIGS + (
        ["raptor_original"] if any("raptor_original" in v for v in f1.values()) else []
    )
    fields = [
        ("answer_form", lambda p: p["answer_form"]),
        ("granularity", lambda p: p["granularity"]),
        ("n_facts", lambda p: str(min(p["n_facts"], 3)) + ("+" if p["n_facts"] >= 3 else "")),
        ("cross_section", lambda p: "cross" if p["cross_section"] else "single"),
    ]
    row_labels, grid = [], []
    for field, key in fields:
        groups: dict[str, list[str]] = defaultdict(list)
        for q in f1:
            if q in prof:
                groups[key(prof[q])].append(q)
        for g, qs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            row_labels.append(f"{field}: {g}  (n={len(qs)})")
            grid.append([sum(f1[q].get(c, 0.0) for q in qs) / len(qs) for c in cols])

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 3.5, 0.42 * len(grid) + 1.6))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=max(max(r) for r in grid))
    for i, row in enumerate(grid):
        best = max(range(len(row)), key=row.__getitem__)
        for j, v in enumerate(row):
            dark = v > 0.6 * im.norm.vmax
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if dark else INK,
                fontweight="bold" if j == best else "normal",
            )
    ax.set_xticks(
        range(len(cols)),
        [c.replace("_only", "").replace("raptor_original", "orig.") for c in cols],
        rotation=40,
        ha="right",
        fontsize=8.5,
    )
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("mean answer F1", color=INK_2)
    ax.set_title("Dev F1 by LLM-profile group (bold = best config in the row)")
    save(fig, out, "profile_groups")


def config_choice(a1: Path, a2: Path, out: Path) -> None:
    freq = {}
    for run, name in ((a1, "A1 table"), (a2, "A2 rules")):
        rep = json.loads((run / "dev_report.json").read_text())
        freq[name] = rep["router"]["config_frequency"]
    cols = [c for c in CONFIGS if any(c in f for f in freq.values())]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(cols) + 1.4))
    h = 0.38
    colors = {"A1 table": "#2a78d6", "A2 rules": "#eb6834"}
    for k, (name, f) in enumerate(freq.items()):
        ys = [i + (k - 0.5) * h for i in range(len(cols))]
        vals = [f.get(c, 0) for c in cols]
        ax.barh(ys, vals, height=h - 0.04, color=colors[name], label=name)
        for y, v in zip(ys, vals, strict=True):
            if v:
                ax.annotate(
                    str(v),
                    (v, y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=INK_2,
                )
    ax.set_yticks(range(len(cols)), cols, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("dev questions routed to the config")
    ax.set_title("Where the routers send questions (dev)")
    ax.legend(loc="lower right")
    save(fig, out, "config_choice")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--a1", type=Path, help="router_train run dir for llm_profile_table")
    p.add_argument("--a2", type=Path, help="router_train run dir for llm_profile_rules")
    p.add_argument("--dev-labels", type=Path)
    p.add_argument("--dev-profiles", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/figures/t12_dev"))
    a = p.parse_args()
    a1 = a.a1 or latest("outputs/router_train_llm_profile_table/*/")
    a2 = a.a2 or latest("outputs/router_train_llm_profile_rules/*/")
    labels = a.dev_labels or latest("outputs/labels_dev/*/") / "labels.jsonl"
    profiles = a.dev_profiles or latest("outputs/profile_dev/*/") / "profiles.jsonl"
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"A1={a1}\nA2={a2}\nlabels={labels}\nprofiles={profiles}")
    style()
    pareto(a1, a2, a.out)
    deltas(a1, a2, a.out)
    profile_groups(labels, profiles, a.out)
    config_choice(a1, a2, a.out)


if __name__ == "__main__":
    main()

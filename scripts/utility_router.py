"""E2/E3: retrieval-aware utility predictor over leaf budgets. Offline, NO API CALLS.

    uv run python scripts/utility_router.py --stage train            # select + freeze (train only)
    uv run --group plots python scripts/utility_router.py --stage dev --frozen <dir>   # ONE look

The pre-registration is in NOTES.md ("Pre-registration: retrieval-aware utility
predictor"). The test split is never read.

Action space: leaf_only@{400,1000,2000}. Base a0 = leaf@1000. Two regressions predict the
gains D400 = F1(@400) - F1(@1000) and D2000 = F1(@2000) - F1(@1000) from features known
at retrieval time. Policy: argmax_a [D^(q,a) - lambda * (C(q,a) - C(q,a0))], where C is the
prompt input tokens of that context plus the config's mean train output tokens.

Features (no gold data):
  Q  SBERT query embedding (768; PCA-16 for GBDT) + surface question features
  R  leaf similarity stats (top-k, gaps, std, entropy, near-ties), per added block
     (400->1000, 1000->2000): mean / max similarity, similarity relative to top-1,
     novelty = 1 - max cosine to the chunks already in the smaller context, query
     content-word coverage of the context and its gain; positional diversity (contiguous
     runs, normalized span, mean position); paper length in leaves.
Candidates: {Q, R, Q+R} x {ridge (inner-CV alpha), small GBDT}. The best candidate by
5-fold paper-grouped cross-fitted train gain over the matched leaf mix at 1039 tokens/q
is refit on all of train. lambda is calibrated on the cross-fitted train predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import (  # noqa: E402
    ALL9,
    LEAF,
    PRICE_IN,
    PRICE_OUT,
    STOP,
    Split,
    boot,
    doc_folds,
    fixed_mix,
    latest,
    load_tree,
    question_features,
)
from section5_diagnostics import choice_at, lagrangian_path, value_at  # noqa: E402

BASE = "leaf_only@1000"
ACTIONS = LEAF  # leaf_only@400, @1000, @2000
TARGET = 1039.0
SECONDARY = 1500.0
GATE = 0.02
LAMBDAS = np.concatenate([[0.0], np.geomspace(1e-7, 1e-1, 600)])
EMB_MODEL = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
WORD = re.compile(r"[a-z0-9]+")

R_NAMES = [
    "s1", "s2", "s5", "s10", "s20", "gap12", "gap1_10", "std20", "entropy20", "n_close",
    "n_leaves",
    "add1_mean", "add1_max", "add1_rel", "add1_novelty", "add1_n",
    "add2_mean", "add2_max", "add2_rel", "add2_novelty", "add2_n",
    "cov400", "cov1000", "cov2000", "cov_gain1", "cov_gain2",
    "runs1000", "runs2000", "span1000", "span2000", "pos_top5",
]  # fmt: skip


def qterms(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


def retrieval_block(tree, qemb: np.ndarray, nodes: dict[str, list[int]], question: str):
    leaves = tree.layer_to_nodes[0]
    E = {n.index: np.asarray(n.embeddings["EMB"], dtype=np.float32) for n in leaves}
    for k in E:
        E[k] = E[k] / np.linalg.norm(E[k])
    sim = {k: float(v @ qemb) for k, v in E.items()}
    sl = np.sort(np.array(list(sim.values())))[::-1]

    def pick(i):
        return float(sl[i]) if len(sl) > i else float(sl[-1])

    top = sl[:20]
    p = np.exp((top - top.max()) / 0.05)
    p /= p.sum()
    s1 = pick(0)
    n400, n1000, n2000 = nodes["leaf_only@400"], nodes["leaf_only@1000"], nodes["leaf_only@2000"]

    def added(small, big):
        add = [i for i in big if i not in set(small)]
        if not add:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        s = np.array([sim[i] for i in add])
        base = np.stack([E[i] for i in small]) if small else None
        nov = (
            [1 - float(np.max(base @ E[i])) for i in add] if base is not None else [1.0] * len(add)
        )
        return [
            float(s.mean()),
            float(s.max()),
            float(s.mean() / s1),
            float(np.mean(nov)),
            len(add),
        ]

    q = qterms(question)
    text = {n.index: n.text for n in leaves}

    def cov(ids):
        have = set().union(*[qterms(text[i]) for i in ids]) if ids else set()
        return len(q & have) / len(q) if q else 0.0

    def runs(ids):
        s = sorted(ids)
        return float(1 + sum(b != a + 1 for a, b in zip(s, s[1:], strict=False))) if s else 0.0

    nl = len(leaves)

    def span(ids):
        return (max(ids) - min(ids)) / nl if ids else 0.0

    c4, c10, c20 = cov(n400), cov(n1000), cov(n2000)
    return [
        s1, pick(1), pick(4), pick(9), pick(19), s1 - pick(1), s1 - pick(9),
        float(top.std()), float(-(p * np.log(p + 1e-12)).sum()), float((sl >= s1 - 0.05).sum()),
        float(nl),
        *added(n400, n1000), *added(n1000, n2000),
        c4, c10, c20, c10 - c4, c20 - c10,
        runs(n1000), runs(n2000), span(n1000), span(n2000),
        float(np.mean(sorted(n2000, key=lambda i: -sim[i])[:5]) / nl) if n2000 else 0.0,
    ]  # fmt: skip


def features(s: Split, split: str, cache: Path) -> dict[str, np.ndarray]:
    f = cache / f"utility_features_{split}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z["feats"].item()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMB_MODEL, device="cpu")
    emb = model.encode([s.question[q] for q in s.qs], batch_size=64, normalize_embeddings=True)
    out = {"emb": {}, "qf": {}, "rf": {}}
    trees = {}
    for i, q in enumerate(s.qs):
        d = s.doc[q]
        if d not in trees:
            trees[d] = load_tree(d, s.index_hash[d])
        out["emb"][q] = emb[i].astype(np.float32)
        out["qf"][q] = np.array(question_features(s.question[q]), dtype=np.float32)
        out["rf"][q] = np.array(
            retrieval_block(trees[d], emb[i], s.nodes[q], s.question[q]), dtype=np.float32
        )
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, feats=np.array(out, dtype=object))
    return out


# ---------------------------------------------------------------------------- models


class Candidate:
    def __init__(self, featset: str, kind: str):
        self.featset, self.kind = featset, kind
        self.name = f"{kind}|{featset}"

    def X(self, F: dict, qs: list[str], pca=None) -> np.ndarray:
        blocks = []
        if "Q" in self.featset:
            e = np.stack([F["emb"][q] for q in qs])
            blocks += [pca.transform(e) if (self.kind == "gbdt" and pca) else e]
            blocks.append(np.stack([F["qf"][q] for q in qs]))
        if "R" in self.featset:
            blocks.append(np.stack([F["rf"][q] for q in qs]))
        return np.concatenate(blocks, axis=1)

    def fit(self, F, qs, Y):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.pca = None
        if "Q" in self.featset and self.kind == "gbdt":
            self.pca = PCA(16, random_state=0).fit(np.stack([F["emb"][q] for q in qs]))
        X = self.X(F, qs, self.pca)
        self.models = []
        for j in range(Y.shape[1]):
            if self.kind == "ridge":
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-1, 1e4, 25)))
            else:
                m = HistGradientBoostingRegressor(
                    max_iter=150,
                    learning_rate=0.05,
                    max_depth=3,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    random_state=0,
                )
            m.fit(X, Y[:, j])
            self.models.append(m)
        return self

    def predict(self, F, qs):
        X = self.X(F, qs, self.pca)
        return np.stack([m.predict(X) for m in self.models], axis=1)


def targets(s: Split, qs):
    return np.array([[s.f1[q][a] - s.f1[q][BASE] for a in ("leaf_only@400", "leaf_only@2000")]
                     for q in qs])  # fmt: skip


def cost_matrix(s: Split, qs, out_mean: dict[str, float]):
    """C(q,a): prompt input tokens of that context + the config's mean train output tokens."""
    return np.array([[s.tin[q][a] + out_mean[a] for a in ACTIONS] for q in qs], dtype=float)


def policy(pred: np.ndarray, C: np.ndarray, lam: float) -> np.ndarray:
    """pred: (n, 2) gains for @400, @2000 vs @1000 -> action index into ACTIONS."""
    util = np.stack([pred[:, 0], np.zeros(len(pred)), pred[:, 1]], axis=1)
    return np.argmax(util - lam * (C - C[:, [1]]), axis=1)


def calibrate(pred, C, target):
    """lambda whose mean cost is <= target and closest to it (largest F1-seeking)."""
    for lam in LAMBDAS:
        pick = policy(pred, C, lam)
        if C[np.arange(len(pick)), pick].mean() <= target:
            return float(lam)
    return float(LAMBDAS[-1])


def realized(s: Split, qs, pick) -> dict:
    ch = {q: ACTIONS[i] for q, i in zip(qs, pick, strict=True)}
    gin = float(np.mean([s.tin[q][ch[q]] for q in qs]))
    gout = float(np.mean([s.tout[q][ch[q]] for q in qs]))
    return {
        "f1": round(float(np.mean([s.f1[q][ch[q]] for q in qs])), 4),
        "input_tokens": round(gin, 1),
        "output_tokens": round(gout, 1),
        "total_tokens": round(gin + gout, 1),
        "profiler_tokens": 0.0,
        "usd_per_1k_q": round((gin * PRICE_IN + gout * PRICE_OUT) / 1e3, 4),
        "mix": {a: int(sum(ch[q] == a for q in qs)) for a in ACTIONS},
        "choice": ch,
    }


def vs_mix(s: Split, qs, row) -> dict:
    s_qs = s.qs
    s.qs = qs
    mix = fixed_mix(s, row["total_tokens"])
    s.qs = s_qs
    d = {q: s.f1[q][row["choice"][q]] - s.f1[q][mix[q]] for q in qs}
    return {
        "mix_f1": round(float(np.mean([s.f1[q][mix[q]] for q in qs])), 4),
        "delta_ci": boot(d, s.doc),
    }


# ---------------------------------------------------------------------------- stages


def load_split(split: str):
    q = Path(f"data/processed/qasper/v0.3/{split}/questions.jsonl")
    return Split(latest(f"outputs/labels_{split}/*/"), None, q, configs=ALL9)


def stage_train(out: Path, cache: Path) -> None:
    tr = load_split("train")
    qs = tr.qs
    F = features(tr, "train", cache)
    Y = targets(tr, qs)
    out_mean = {a: float(np.mean([tr.tout[q][a] for q in qs])) for a in ACTIONS}
    C = cost_matrix(tr, qs, out_mean)
    folds = doc_folds(tr, qs)
    rep = {"n_train": len(qs), "candidates": {}, "out_mean": out_mean}
    for fs in ("Q", "R", "Q+R"):
        for kind in ("ridge", "gbdt"):
            cand = Candidate(fs, kind)
            cf = np.zeros_like(Y)
            for k in range(5):
                tr_q = [q for q, f in zip(qs, folds, strict=True) if f != k]
                te_q = [q for q, f in zip(qs, folds, strict=True) if f == k]
                m = Candidate(fs, kind).fit(F, tr_q, Y[folds != k])
                cf[folds == k] = m.predict(F, te_q)
            row = {}
            for tgt in (TARGET, SECONDARY):
                lam = calibrate(cf, C, tgt)
                r = realized(tr, qs, policy(cf, C, lam))
                row[str(int(tgt))] = {
                    "lambda": lam,
                    **{k: v for k, v in r.items() if k != "choice"},
                    **vs_mix(tr, qs, r),
                }
            row["corr_D400"] = round(float(np.corrcoef(cf[:, 0], Y[:, 0])[0, 1]), 3)
            row["corr_D2000"] = round(float(np.corrcoef(cf[:, 1], Y[:, 1])[0, 1]), 3)
            from sklearn.metrics import roc_auc_score

            row["auc_D2000_positive"] = round(float(roc_auc_score(Y[:, 1] > 0, cf[:, 1])), 3)
            row["auc_D400_negative"] = round(float(roc_auc_score(Y[:, 0] < 0, -cf[:, 0])), 3)
            row["_cf"] = cf
            rep["candidates"][cand.name] = row
            print(
                f"{cand.name:10} train-CV @1039: F1 {row['1039']['f1']} tok "
                f"{row['1039']['total_tokens']} vs mix {row['1039']['delta_ci']} | corr "
                f"{row['corr_D400']}/{row['corr_D2000']} auc {row['auc_D2000_positive']}",
                flush=True,
            )
    best = max(rep["candidates"], key=lambda n: rep["candidates"][n]["1039"]["delta_ci"][0])
    kind, fs = best.split("|")
    cf = rep["candidates"][best]["_cf"]
    lam = {str(int(t)): calibrate(cf, C, t) for t in (TARGET, SECONDARY)}
    lam_sweep = [float(x) for x in LAMBDAS[::10]]
    model = Candidate(fs, kind).fit(F, qs, Y)
    for r in rep["candidates"].values():
        r.pop("_cf")
    frozen = {
        "selected": best,
        "lambda": lam,
        "lambda_sweep": lam_sweep,
        "out_mean": out_mean,
        "features": {"Q": "SBERT emb + question_features", "R": R_NAMES},
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "model.pkl").write_bytes(pickle.dumps(model))
    (out / "frozen.json").write_text(json.dumps(frozen, indent=1))
    (out / "train_selection.json").write_text(json.dumps(rep, indent=1))
    h = hashlib.sha256((out / "model.pkl").read_bytes() + (out / "frozen.json").read_bytes())
    (out / "FROZEN_SHA256").write_text(h.hexdigest() + "\n")
    print(f"selected {best}; lambda {lam}; frozen to {out} sha256 {h.hexdigest()[:16]}")


def stage_dev(frozen_dir: Path, cache: Path, force: bool, plots: bool) -> None:
    done = frozen_dir / "dev_report.json"
    if done.exists() and not force:
        raise SystemExit(f"{done} exists: the dev look was already taken (pre-registered once)")
    h = hashlib.sha256(
        (frozen_dir / "model.pkl").read_bytes() + (frozen_dir / "frozen.json").read_bytes()
    ).hexdigest()
    if h + "\n" != (frozen_dir / "FROZEN_SHA256").read_text():
        raise SystemExit("frozen artifacts changed after freezing")
    fz = json.loads((frozen_dir / "frozen.json").read_text())
    model = pickle.loads((frozen_dir / "model.pkl").read_bytes())
    dv = load_split("dev")
    qs = dv.qs
    F = features(dv, "dev", cache)
    pred = model.predict(F, qs)
    C = cost_matrix(dv, qs, fz["out_mean"])
    rep: dict = {"selected": fz["selected"], "frozen_sha256": h, "n_dev": len(qs), "points": {}}
    for tgt, lam in fz["lambda"].items():
        r = realized(dv, qs, policy(pred, C, lam))
        rep["points"][tgt] = {"lambda": lam, **{k: v for k, v in r.items() if k != "choice"}}
        rep["points"][tgt].update(vs_mix(dv, qs, r))
        rep["points"][tgt]["_choice"] = r["choice"]
    prim = rep["points"][str(int(TARGET))]
    d = prim["delta_ci"]
    rep["gate"] = {
        "rule": "delta vs matched leaf mix >= +0.02 and 95% CI lower bound > 0",
        "delta_ci": d,
        "passed": bool(d[0] >= GATE and d[1] > 0),
    }
    # oracle comparison: regret and share of leaf-only oracle headroom captured
    for space, cfgs in (("leaf_only", LEAF), ("all9", ALL9)):
        Fm = np.array([[dv.f1[q][c] for c in cfgs] for q in qs])
        Cm = np.array([[dv.cost(q, c) for c in cfgs] for q in qs], dtype=float)
        orc = lagrangian_path(Fm, Cm, np.arange(len(qs)))
        fix = lagrangian_path(Fm, Cm, np.zeros(len(qs), int))
        for p in rep["points"].values():
            b = p["total_tokens"]
            o, f = value_at(orc, b), value_at(fix, b)
            po = choice_at(orc, b, qs)
            regret = {q: float(Fm[i, po[i]] - dv.f1[q][p["_choice"][q]]) for i, q in enumerate(qs)}
            p[f"oracle_{space}_f1"] = round(o, 4)
            p[f"regret_vs_oracle_{space}"] = boot(regret, dv.doc)
            if space == "leaf_only":
                p["share_leaf_oracle_headroom_captured"] = (
                    round((p["f1"] - f) / (o - f), 3) if o > f else None
                )
    # lambda sweep (frontier, reporting only)
    sweep = []
    for lam in fz["lambda_sweep"]:
        r = realized(dv, qs, policy(pred, C, lam))
        sweep.append([r["total_tokens"], r["f1"]])
    rep["frontier"] = sorted({tuple(x) for x in sweep})
    for p in rep["points"].values():
        p.pop("_choice")
    done.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "frontier"}, indent=1))
    if plots:
        plot(frozen_dir, rep, dv)


def plot(out: Path, rep: dict, dv: Split) -> None:
    import matplotlib.pyplot as plt
    from make_dev_charts import INK, INK_2, save, style

    style()
    qs = dv.qs
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for cfgs, col, lab in ((ALL9, "#d6762a", "oracle, all 9 configs"),
                           (LEAF, "#e8a96b", "oracle, leaf budgets only")):  # fmt: skip
        Fm = np.array([[dv.f1[q][c] for c in cfgs] for q in qs])
        Cm = np.array([[dv.cost(q, c) for c in cfgs] for q in qs], dtype=float)
        o = np.array([[c, f] for c, f, _ in lagrangian_path(Fm, Cm, np.arange(len(qs)))])
        ax.plot(o[:, 0], o[:, 1], color=col, lw=2, label=lab)
    Fm = np.array([[dv.f1[q][c] for c in LEAF] for q in qs])
    Cm = np.array([[dv.cost(q, c) for c in LEAF] for q in qs], dtype=float)
    fx = np.array([[c, f] for c, f, _ in lagrangian_path(Fm, Cm, np.zeros(len(qs), int))])
    ax.plot(fx[:, 0], fx[:, 1], color=INK, lw=2, marker="s", ms=5, label="best fixed leaf mix")
    fr = np.array(rep["frontier"])
    ax.plot(fr[:, 0], fr[:, 1], color="#2a78d6", lw=1.5, ls=":", marker=".",
            label="utility predictor, λ sweep (reporting only)")  # fmt: skip
    for tgt, p in rep["points"].items():
        ax.plot(p["total_tokens"], p["f1"], "D", color="#2a78d6", ms=8)
        ax.annotate(
            f"frozen @{tgt}: Δ vs mix {p['delta_ci'][0]:+.3f} "
            f"[{p['delta_ci'][1]:+.3f}, {p['delta_ci'][2]:+.3f}]",
            (p["total_tokens"], p["f1"]), textcoords="offset points", xytext=(8, -14),
            fontsize=8, color=INK_2,
        )  # fmt: skip
    ax.plot(1014, 0.336, "^", color="#8a8984", ms=8)
    ax.annotate(
        "A1/A2 (LLM profile, incl. profiler tokens)",
        (1014, 0.336),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=8,
        color=INK_2,
    )
    ax.set_xlim(350, 2150)
    ax.set_xlabel("mean tokens per question (generation; no profiler)")
    ax.set_ylabel("dev F1")
    g = "PASSED" if rep["gate"]["passed"] else "FAILED"
    ax.set_title(f"Retrieval-aware utility predictor ({rep['selected']}), dev, gate {g}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out, "utility_router_dev")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stage", choices=["train", "dev"], required=True)
    p.add_argument("--frozen", type=Path, help="frozen dir from --stage train (for --stage dev)")
    p.add_argument("--force", action="store_true", help="re-run the dev look (not pre-registered)")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    base = Path("outputs/diagnostics/utility_router")
    cache = base / "cache"
    if a.stage == "train":
        stage_train(base / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), cache)
    else:
        if not a.frozen:
            raise SystemExit("--frozen is required for --stage dev")
        stage_dev(a.frozen, cache, a.force, not a.no_plots)


if __name__ == "__main__":
    main()

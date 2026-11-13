"""Offline diagnostics for the optimizer redesign (no API calls; local SBERT only).

    uv run python scripts/offline_diagnostics.py            # all steps, latest runs
    uv run python scripts/offline_diagnostics.py --out outputs/diagnostics/<name>

Reads saved label tables, predictions, LLM profiles, cached RAPTOR trees (node
embeddings) and QASPER gold evidence. Gold answers / answer types / evidence are used
only for labels and diagnosis, never as router features.

Steps (written to <out>/report.json and printed):
  1  baselines, cost-constrained oracle frontier (tokens and USD), noise floor from
     runs A/B, where hierarchical beats leaf, A1 loss breakdown, leakage audit
  2  retrieval-aware routing: features recomputed from the cached trees; predictability
     (AUC) and routers (rule / ridge / gradient boosting) trained on train only
  3  cascades simulated from stored answers: retrieval-only and answer-based
  4  summary bottleneck: truncation statistics and summary displacement in collapsed
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from adaptive_raptor.evaluation.research import paired_document_bootstrap

LEAF = ["leaf_only@400", "leaf_only@1000", "leaf_only@2000"]
COLL = ["collapsed@400", "collapsed@1000", "collapsed@2000"]
SUMM = ["summary_only@400", "summary_only@1000", "summary_only@2000"]
ALL9 = LEAF + COLL + SUMM
ORIG = "raptor_original"
PRICE_IN, PRICE_OUT = 0.15, 0.60  # USD per 1M tokens (NOTES spend log)
STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "by",
        "with",
        "as",
        "at",
        "from",
        "that",
        "this",
        "it",
        "its",
        "we",
        "our",
        "they",
        "their",
        "which",
        "what",
        "how",
        "do",
        "does",
        "did",
        "not",
        "no",
        "can",
    ]
)


def latest(pattern: str) -> Path:
    runs = sorted(Path().glob(pattern))
    if not runs:
        raise SystemExit(f"no run matches {pattern}")
    return runs[-1]


def usd(i: float, o: float) -> float:
    return (i * PRICE_IN + o * PRICE_OUT) / 1e6


# --------------------------------------------------------------------------- data


class Split:
    """Per-question F1, tokens, answers, retrieved nodes for one label run."""

    def __init__(self, run: Path, profiles: Path | None, questions: Path, configs=ALL9):
        self.f1: dict[str, dict[str, float]] = defaultdict(dict)
        self.tin: dict[str, dict[str, int]] = defaultdict(dict)
        self.tout: dict[str, dict[str, int]] = defaultdict(dict)
        self.doc: dict[str, str] = {}
        self.atype: dict[str, str] = {}
        for line in (run / "labels.jsonl").open():
            r = json.loads(line)
            q, c = r["question_id"], r["config_id"]
            self.f1[q][c] = r["answer_f1"]
            self.tin[q][c] = r.get("input_tokens") or 0
            self.tout[q][c] = r.get("output_tokens") or 0
            self.doc[q] = r["doc_id"]
            self.atype[q] = r["answer_type"]
        self.nodes: dict[str, dict[str, list[int]]] = defaultdict(dict)
        self.levels: dict[str, dict[str, list[int]]] = defaultdict(dict)
        self.answer: dict[str, dict[str, str]] = defaultdict(dict)
        self.sufficiency: dict[str, dict[str, str | None]] = defaultdict(dict)
        self.ctx_tokens: dict[str, dict[str, int]] = defaultdict(dict)
        self.gen_cache_hit: dict[str, dict[str, bool]] = defaultdict(dict)
        self.prefix_hit: dict[str, dict[str, int]] = defaultdict(dict)
        self.question: dict[str, str] = {}
        self.index_hash: dict[str, str] = {}
        for line in (run / "traces.jsonl").open():
            t = json.loads(line)
            q, c = t["question_id"], t["config_id"]
            self.nodes[q][c] = t["node_ids"] or []
            self.levels[q][c] = t["levels"] or []
            self.answer[q][c] = t["answer"] or ""
            self.sufficiency[q][c] = t.get("sufficiency")
            self.ctx_tokens[q][c] = t.get("context_tokens") or 0
            self.gen_cache_hit[q][c] = bool(t.get("generation_cache_hit"))
            self.prefix_hit[q][c] = t.get("prompt_cache_hit_tokens") or 0
            self.question[q] = t["question"]
            self.index_hash[t["doc_id"]] = t["index_key_hash"]
        self.profile = {}
        self.prof_tok = {}
        for line in profiles.open() if profiles else []:
            r = json.loads(line)
            self.profile[r["question_id"]] = r["profile"]
            self.prof_tok[r["question_id"]] = (r["input_tokens"], r["output_tokens"])
        self.evidence: dict[str, list[str]] = {}
        for line in questions.open():
            r = json.loads(line)
            if r["question_id"] in self.f1:
                ev = {e for a in r["answers"] for e in a.get("evidence", []) if e.strip()}
                self.evidence[r["question_id"]] = sorted(ev)
        self.qs = sorted(q for q in self.f1 if all(c in self.f1[q] for c in configs))

    def cost(self, q: str, c: str) -> float:
        return self.tin[q][c] + self.tout[q][c]

    def usd(self, q: str, c: str) -> float:
        return usd(self.tin[q][c], self.tout[q][c])


def load_tree(doc: str, key_hash: str):
    path = Path("indexes/qasper/v0.3/f100/seed224") / f"{doc}__{key_hash[:16]}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- stats


def boot(delta: dict[str, float], doc: dict[str, str]) -> list[float]:
    m, lo, hi = paired_document_bootstrap(delta, doc, 2000, 0.95)
    return [round(m, 4), round(lo, 4), round(hi, 4)]


def system(s: Split, choice: dict[str, str], extra_in=None, extra_out=None, name=""):
    """Score a per-question choice; extra_* are additional tokens (profiler, cascade)."""
    qs = list(choice)
    extra_in = extra_in or {}
    extra_out = extra_out or {}
    gin = np.mean([s.tin[q][choice[q]] + extra_in.get(q, 0) for q in qs])
    gout = np.mean([s.tout[q][choice[q]] + extra_out.get(q, 0) for q in qs])
    return {
        "system": name,
        "n": len(qs),
        "f1": round(float(np.mean([s.f1[q][choice[q]] for q in qs])), 4),
        "input_tokens": round(float(gin), 1),
        "output_tokens": round(float(gout), 1),
        "total_tokens": round(float(gin + gout), 1),
        "usd_per_1k_q": round(usd(gin, gout) * 1000, 4),
    }


def compare(s: Split, choice, base_choice, extra_in=None, extra_out=None):
    extra_in = extra_in or {}
    d = {q: s.f1[q][choice[q]] - s.f1[q][base_choice[q]] for q in choice}
    return boot(d, s.doc)


def fixed(s: Split, c: str) -> dict[str, str]:
    return {q: c for q in s.qs}


def fixed_mix(s: Split, target: float, ladder=LEAF) -> dict[str, str]:
    """Best fixed comparator at a matched mean cost: the two neighbouring leaf budgets
    mixed per question by a fixed hash (no information about the question is used)."""
    import hashlib

    costs = [np.mean([s.cost(q, c) for q in s.qs]) for c in ladder]
    lo = max([i for i, c in enumerate(costs) if c <= target], default=0)
    hi = min(lo + 1, len(ladder) - 1)
    w = 0.0 if hi == lo else min(1.0, max(0.0, (target - costs[lo]) / (costs[hi] - costs[lo])))
    rank = lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16) % 10_000 / 10_000  # noqa: E731
    return {q: ladder[hi] if rank(q) < w else ladder[lo] for q in s.qs}


def vs_matched(s: Split, row: dict, choice: dict[str, str]) -> None:
    mix = fixed_mix(s, row["total_tokens"])
    row["matched_fixed"] = system(s, mix, name="leaf mix")
    row["vs_matched_fixed"] = compare(s, choice, mix)


# --------------------------------------------------------------------------- step 1


def lagrangian_frontier(s: Split, configs, unit="tokens"):
    """Upper envelope of the per-question oracle: for each lambda each question picks
    argmax F1 - lambda*cost. Returns sorted (cost, f1) points."""
    qs = s.qs
    F = np.array([[s.f1[q][c] for c in configs] for q in qs])
    if unit == "tokens":
        C = np.array([[s.cost(q, c) for c in configs] for q in qs])
    else:
        C = np.array([[s.usd(q, c) * 1000 for c in configs] for q in qs])
    pts = []
    scale = np.median(C) or 1
    for lam in np.concatenate([[0], np.geomspace(1e-6, 10, 400)]) / scale:
        pick = np.argmax(F - lam * C - 1e-9 * C, axis=1)  # ties -> cheaper
        pts.append((C[np.arange(len(qs)), pick].mean(), F[np.arange(len(qs)), pick].mean()))
    hull = []  # keep points whose F1 rises with cost
    for p in sorted(set(pts)):
        if not hull or p[1] > hull[-1][1]:
            hull.append(p)
    return hull


def at_budget(frontier, budget):
    pts = sorted(frontier)
    best = max((f for c, f in pts if c <= budget), default=None)
    # linear interpolation between neighbouring envelope points (randomised mixing)
    for (c0, f0), (c1, f1) in zip(pts, pts[1:], strict=False):
        if c0 <= budget <= c1 and c1 > c0:
            best = max(best or 0, f0 + (f1 - f0) * (budget - c0) / (c1 - c0))
    return round(best, 4) if best is not None else None


def noise_floor(run_a: Path, run_b: Path) -> dict:
    def read(run):
        out = defaultdict(dict)
        for line in (run / "labels.jsonl").open():
            r = json.loads(line)
            out[r["question_id"]][r["config_id"]] = r["answer_f1"]
        return out

    a, b = read(run_a), read(run_b)
    qs = [q for q in a if q in b and all(c in a[q] and c in b[q] for c in ALL9)]
    per = {}
    for c in ALL9:
        diff = np.mean([a[q][c] != b[q][c] for q in qs])
        self_oracle = np.mean([max(a[q][c], b[q][c]) for q in qs]) - np.mean([a[q][c] for q in qs])
        per[c] = {
            "frac_f1_differs": round(float(diff), 3),
            "self_oracle_gain": round(float(self_oracle), 4),
        }
    oracle9 = np.mean([max(a[q][c] for c in ALL9) for q in qs])
    oracle18 = np.mean([max(max(a[q][c], b[q][c]) for c in ALL9) for q in qs])
    return {
        "n_questions": len(qs),
        "per_config": per,
        "oracle9_run_a": round(float(oracle9), 4),
        "oracle9_with_run_b_resamples": round(float(oracle18), 4),
        "mean_self_oracle_gain": round(
            float(np.mean([v["self_oracle_gain"] for v in per.values()])), 4
        ),
    }


def evidence_hit(evidence: list[str], text: str, thr: float = 0.5) -> bool | None:
    """True if some gold evidence paragraph has >= thr content-token recall in text."""
    ev = [e for e in evidence if not e.startswith("FLOAT SELECTED")]
    if not ev:
        return None
    have = set(w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP)
    for e in ev:
        toks = [w for w in re.findall(r"[a-z0-9]+", e.lower()) if w not in STOP]
        if toks and sum(t in have for t in toks) / len(toks) >= thr:
            return True
    return False


# --------------------------------------------------------------------------- step 2 features

WH = [
    "what",
    "how",
    "which",
    "why",
    "who",
    "where",
    "when",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "can",
    "have",
    "has",
]


def question_features(text: str) -> list[float]:
    t = text.lower().strip()
    first = re.findall(r"[a-z]+", t)[:1]
    words = re.findall(r"[a-z0-9]+", t)
    return [
        len(words),
        *[float(bool(first) and first[0] == w) for w in WH],
        float(" and " in t),
        float(" or " in t),
        float(bool(re.search(r"\b(compare|differ|better|outperform|versus|vs)\b", t))),
        float(bool(re.search(r"\b(how many|how much|number|size|percent)\b", t))),
        float(bool(re.search(r"\b(dataset|corpus|data)\b", t))),
        float(bool(re.search(r"\b(baseline|model|method|approach|architecture)\b", t))),
        float(bool(re.search(r"\b(result|performance|accuracy|score|f1|bleu)\b", t))),
        float(t.endswith("?")),
    ]


QF_NAMES = [
    "n_words",
    *[f"wh_{w}" for w in WH],
    "has_and",
    "has_or",
    "comparative",
    "quantity",
    "mentions_data",
    "mentions_method",
    "mentions_result",
    "has_qmark",
]
RF_NAMES = [
    "leaf_s1",
    "leaf_s2",
    "leaf_s5",
    "leaf_s10",
    "leaf_gap12",
    "leaf_gap1_10",
    "leaf_std20",
    "leaf_entropy20",
    "leaf_n_close",
    "n_leaves",
    "summ_s1",
    "summ_minus_leaf",
    "coll_summary_frac10",
    "summ_n",
]


def retrieval_features(tree, qemb: np.ndarray) -> list[float]:
    def sims(nodes):
        if not nodes:
            return np.array([])
        m = np.stack([np.asarray(n.embeddings["EMB"], dtype=np.float32) for n in nodes])
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        return np.sort(m @ qemb)[::-1]

    leaves = tree.layer_to_nodes[0]
    summ = [n for k, v in tree.layer_to_nodes.items() if int(k) >= 1 for n in v]
    sl, ss = sims(leaves), sims(summ)
    top = sl[:20]
    p = np.exp((top - top.max()) / 0.05)
    p /= p.sum()
    pick = lambda a, i: float(a[i]) if len(a) > i else float(a[-1]) if len(a) else 0.0  # noqa: E731
    s1 = pick(sl, 0)
    allsims = sorted([(v, 0) for v in sl] + [(v, 1) for v in ss], reverse=True)[:10]
    return [
        s1,
        pick(sl, 1),
        pick(sl, 4),
        pick(sl, 9),
        s1 - pick(sl, 1),
        s1 - pick(sl, 9),
        float(top.std()),
        float(-(p * np.log(p + 1e-12)).sum()),
        float((sl >= s1 - 0.05).sum()),
        float(len(sl)),
        pick(ss, 0),
        pick(ss, 0) - s1,
        float(np.mean([lvl for _, lvl in allsims])) if allsims else 0.0,
        float(len(ss)),
    ]


def build_features(s: Split, model, cache: Path) -> dict[str, dict[str, np.ndarray]]:
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["features"].item()
    qs = s.qs
    emb = model.encode([s.question[q] for q in qs], batch_size=64, normalize_embeddings=True)
    by_doc = defaultdict(list)
    for i, q in enumerate(qs):
        by_doc[s.doc[q]].append(i)
    feats = {"qf": {}, "rf": {}, "emb": {}}
    for doc, idx in by_doc.items():
        tree = load_tree(doc, s.index_hash[doc])
        for i in idx:
            q = qs[i]
            feats["qf"][q] = np.array(question_features(s.question[q]), dtype=np.float32)
            feats["rf"][q] = np.array(retrieval_features(tree, emb[i]), dtype=np.float32)
            feats["emb"][q] = emb[i].astype(np.float32)
    np.savez_compressed(cache, features=np.array(feats, dtype=object))
    return feats


def check_retrieval_reproduces(s: Split, model, n: int = 40) -> float:
    """Share of sampled questions whose recomputed leaf ranking reproduces the stored
    leaf_only@400 node order (validates that the features see the same scores)."""
    ok = 0
    qs = s.qs[:: max(1, len(s.qs) // n)][:n]
    for q in qs:
        tree = load_tree(s.doc[q], s.index_hash[s.doc[q]])
        e = model.encode([s.question[q]], normalize_embeddings=True)[0]
        leaves = tree.layer_to_nodes[0]
        m = np.stack([np.asarray(x.embeddings["EMB"], dtype=np.float32) for x in leaves])
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        order = [leaves[i].index for i in np.argsort(-(m @ e))]
        stored = s.nodes[q]["leaf_only@400"]
        ok += order[: len(stored)] == stored
    return ok / len(qs)


def matrix(s: Split, feats, names: list[str], qs=None, pca=None, extra_prof=False):
    qs = qs or s.qs
    blocks = []
    if "qf" in names:
        blocks.append(np.stack([feats["qf"][q] for q in qs]))
    if "rf" in names:
        blocks.append(np.stack([feats["rf"][q] for q in qs]))
    if "emb" in names:
        e = np.stack([feats["emb"][q] for q in qs])
        blocks.append(pca.transform(e) if pca is not None else e)
    if extra_prof:
        forms = ["yes_no", "short_span", "explanation"]
        grans = ["specific_detail", "section_level", "whole_paper"]
        rows = []
        for q in qs:
            p = s.profile.get(q) or {}
            rows.append(
                [float(p.get("answer_form") == f) for f in forms]
                + [float(p.get("granularity") == g) for g in grans]
                + [float(p.get("n_facts", 1)), float(bool(p.get("cross_section")))]
            )
        blocks.append(np.array(rows, dtype=np.float32))
    return np.concatenate(blocks, axis=1)


# --------------------------------------------------------------------------- step 2 routers


def doc_folds(s: Split, qs, k=5):
    docs = sorted({s.doc[q] for q in qs})
    rng = np.random.default_rng(0)
    rng.shuffle(docs)
    fold_of = {d: i % k for i, d in enumerate(docs)}
    return np.array([fold_of[s.doc[q]] for q in qs])


def fit_predict(kind: str, X, Y, Xte):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    preds = []
    for j in range(Y.shape[1]):
        if kind == "ridge":
            m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
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
        preds.append(m.predict(Xte))
    return np.stack(preds, axis=1)


def policy(pred, cost_vec, lam):
    return np.argmax(pred - lam * cost_vec[None, :], axis=1)


def calibrate_lambda(pred_tr, cost_tr_mat, cost_vec, target):
    """Largest-F1 lambda whose (cross-fitted) train mean cost <= target."""
    best = None
    for lam in np.concatenate([[0], np.geomspace(1e-7, 1e-2, 300)]):
        pick = policy(pred_tr, cost_vec, lam)
        c = cost_tr_mat[np.arange(len(pick)), pick].mean()
        if c <= target:
            return lam
        best = lam
    return best


def run_router(tr: Split, dv: Split, ftr, fdv, configs, featset, kind, target, prof=False):
    from sklearn.decomposition import PCA

    pca = (
        PCA(16, random_state=0).fit(np.stack([ftr["emb"][q] for q in tr.qs]))
        if "emb" in featset
        else None
    )
    Xtr = matrix(tr, ftr, featset, pca=pca, extra_prof=prof)
    Xdv = matrix(dv, fdv, featset, pca=pca, extra_prof=prof)
    Ytr = np.array([[tr.f1[q][c] for c in configs] for q in tr.qs])
    Ctr = np.array([[tr.cost(q, c) for c in configs] for q in tr.qs])
    cost_vec = Ctr.mean(axis=0)
    folds = doc_folds(tr, tr.qs)
    cf = np.zeros_like(Ytr)
    for k in range(5):
        cf[folds == k] = fit_predict(kind, Xtr[folds != k], Ytr[folds != k], Xtr[folds == k])
    prof_cost = np.mean([sum(tr.prof_tok[q]) for q in tr.qs]) if prof else 0.0
    lam = calibrate_lambda(cf, Ctr, cost_vec, target - prof_cost)
    pdv = fit_predict(kind, Xtr, Ytr, Xdv)
    pick = policy(pdv, cost_vec, lam)
    choice = {q: configs[i] for q, i in zip(dv.qs, pick, strict=True)}
    ein = {q: dv.prof_tok[q][0] for q in dv.qs} if prof else None
    eout = {q: dv.prof_tok[q][1] for q in dv.qs} if prof else None
    return choice, ein, eout, lam


def auc_table(tr, dv, ftr, fdv):
    """Can features predict that a bigger budget / hierarchical retrieval helps?"""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    targets = {
        "leaf2000_beats_leaf400": ("leaf_only@2000", "leaf_only@400"),
        "leaf1000_beats_leaf400": ("leaf_only@1000", "leaf_only@400"),
        "coll1000_beats_leaf1000": ("collapsed@1000", "leaf_only@1000"),
        "coll2000_beats_leaf2000": ("collapsed@2000", "leaf_only@2000"),
    }
    pca = PCA(16, random_state=0).fit(np.stack([ftr["emb"][q] for q in tr.qs]))
    out = {}
    for name, (a, b) in targets.items():
        ytr = np.array([tr.f1[q][a] > tr.f1[q][b] for q in tr.qs])
        ydv = np.array([dv.f1[q][a] > dv.f1[q][b] for q in dv.qs])
        row = {"base_rate_dev": round(float(ydv.mean()), 3)}
        for fs_name, fs, prof in [
            ("question", ["qf"], False),
            ("question+emb", ["qf", "emb"], False),
            ("retrieval", ["rf"], False),
            ("question+retrieval", ["qf", "rf"], False),
            ("all+profile", ["qf", "rf", "emb"], True),
        ]:
            Xtr = matrix(tr, ftr, fs, pca=pca, extra_prof=prof)
            Xdv = matrix(dv, fdv, fs, pca=pca, extra_prof=prof)
            for kind, m in [
                (
                    "logreg",
                    make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
                ),
                (
                    "gbdt",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        learning_rate=0.05,
                        max_depth=3,
                        min_samples_leaf=40,
                        random_state=0,
                    ),
                ),
            ]:
                m.fit(Xtr, ytr)
                row[f"{fs_name}/{kind}"] = round(
                    float(roc_auc_score(ydv, m.predict_proba(Xdv)[:, 1])), 3
                )
        out[name] = row
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or Path("outputs/diagnostics") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {
        "note": "offline; dev numbers; test untouched; gold labels used only for "
        "labels and diagnosis"
    }

    tr = Split(
        latest("outputs/labels_train/*/"),
        latest("outputs/profile_train/*/") / "profiles.jsonl",
        Path("data/processed/qasper/v0.3/train/questions.jsonl"),
    )
    dv = Split(
        latest("outputs/labels_dev/*/"),
        latest("outputs/profile_dev/*/") / "profiles.jsonl",
        Path("data/processed/qasper/v0.3/dev/questions.jsonl"),
    )
    dv.qs = [q for q in dv.qs if ORIG in dv.f1[q]]
    print(f"train {len(tr.qs)} questions, dev {len(dv.qs)} questions")

    # ---------------- step 1: baselines + oracle frontier
    base = {c: system(dv, fixed(dv, c), name=c) for c in [ORIG, *ALL9]}
    rep["fixed_dev"] = base
    rep["leaf1000_vs_original"] = compare(dv, fixed(dv, "leaf_only@1000"), fixed(dv, ORIG))
    rep["leaf2000_vs_original"] = compare(dv, fixed(dv, "leaf_only@2000"), fixed(dv, ORIG))
    fr = {}
    for name, cfgs in [
        ("all9", ALL9),
        ("all9+original", ALL9 + [ORIG]),
        ("leaf_budgets", LEAF),
        ("strategy@1000", ["leaf_only@1000", "collapsed@1000", "summary_only@1000"]),
        ("leaf+collapsed", LEAF + COLL),
    ]:
        tok = lagrangian_frontier(dv, cfgs, "tokens")
        dol = lagrangian_frontier(dv, cfgs, "usd")
        fr[name] = {
            "tokens_at_1039": at_budget(tok, 1039),
            "tokens_at_1164": at_budget(tok, 1164),
            "tokens_at_425": at_budget(tok, 425),
            "tokens_at_700": at_budget(tok, 700),
            "tokens_at_2043": at_budget(tok, 2043),
            "usd_per_1k_at_leaf1000": at_budget(dol, base["leaf_only@1000"]["usd_per_1k_q"]),
            "usd_per_1k_at_original": at_budget(dol, base[ORIG]["usd_per_1k_q"]),
            "token_frontier": [[round(c), round(f, 4)] for c, f in tok[:: max(1, len(tok) // 15)]],
        }
    rep["oracle_frontier_dev"] = fr
    rep["noise_floor_pilot_runs_A_B"] = noise_floor(
        Path("outputs/pilot_catalog/20261109T193103Z"),
        Path("outputs/pilot_catalog/20261109T194522Z"),
    )

    # where hierarchical beats leaf (same budget), by predicted profile and gold type (diagnostic)
    wins = {}
    for bud in (400, 1000, 2000):
        lc, cc = f"leaf_only@{bud}", f"collapsed@{bud}"
        groups = defaultdict(list)
        for q in dv.qs:
            p = dv.profile.get(q) or {}
            for g in (
                f"form={p.get('answer_form')}",
                f"gran={p.get('granularity')}",
                f"cross={p.get('cross_section')}",
                f"gold_type={dv.atype[q]}",
                "all",
            ):
                groups[g].append(dv.f1[q][cc] - dv.f1[q][lc])
        wins[f"collapsed_vs_leaf@{bud}"] = {
            g: {
                "n": len(v),
                "mean_delta": round(float(np.mean(v)), 4),
                "coll_better": round(float(np.mean([x > 0 for x in v])), 3),
                "leaf_better": round(float(np.mean([x < 0 for x in v])), 3),
            }
            for g, v in sorted(groups.items())
        }
    rep["hierarchical_vs_leaf_dev"] = wins

    # ---------------- trees: evidence coverage + summary statistics
    print("loading trees for evidence coverage and summary statistics ...")
    texts: dict[str, dict[int, str]] = {}
    summary_tokens, summary_unfinished = [], []
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    for doc, h in dv.index_hash.items():
        tree = load_tree(doc, h)
        texts[doc] = {i: n.text for i, n in tree.all_nodes.items()}
        for k, nodes in tree.layer_to_nodes.items():
            if int(k) >= 1:
                for n in nodes:
                    summary_tokens.append(len(enc.encode(n.text)))
                    summary_unfinished.append(
                        not n.text.rstrip().endswith((".", "!", "?", '"', ")"))
                    )

    def hit(q, c):
        nodes = dv.nodes[q].get(c) or []
        return evidence_hit(
            dv.evidence.get(q, []), " ".join(texts[dv.doc[q]].get(i, "") for i in nodes)
        )

    cov = {}
    for c in [ORIG, *ALL9]:
        h = [hit(q, c) for q in dv.qs]
        h = [x for x in h if x is not None]
        cov[c] = round(float(np.mean(h)), 3)
    rep["evidence_coverage_dev"] = {
        "definition": "share of questions with text evidence where some "
        "gold paragraph has >=50% content-token recall in the context "
        "(abstractive summaries are undercounted)",
        **cov,
    }

    # ---------------- A1 loss breakdown (replay the frozen A1 checkpoint)
    from adaptive_raptor.config.loader import load_config
    from adaptive_raptor.experiments import _label_table, _replay
    from adaptive_raptor.optimizer import Router, load_checkpoint
    from adaptive_raptor.optimizer.profiler import ProfileStore, build_profiler
    from adaptive_raptor.profiling import merge_label_tables

    a1 = latest("outputs/router_train_llm_profile_table/*/")
    rcfg = load_config(a1 / "resolved_config.yaml", [])
    catalog = rcfg.retrieval.catalog()
    records, _ = merge_label_tables([latest("outputs/labels_dev/*/") / "labels.jsonl"])
    table = _label_table(
        records, load_checkpoint(a1 / "router").artifacts["label_cell"], [c.id for c in catalog]
    )
    router = Router.from_checkpoint(
        a1 / "router", rcfg.optimizer, catalog, build_profiler(rcfg.optimizer.profiler)
    )
    a1_choice, _, _, _ = _replay(
        router,
        {q: table[q] for q in dv.qs},
        ProfileStore.load([latest("outputs/profile_dev/*/") / "profiles.jsonl"]),
    )
    breakdown = {}
    for comp in (ORIG, "leaf_only@1000"):
        cats = Counter()
        losses = [q for q in dv.qs if dv.f1[q][a1_choice[q]] < dv.f1[q][comp]]
        for q in losses:
            c = a1_choice[q]
            strat, bud = c.split("@")
            budget = "budget<1000" if int(bud) < 1000 else "budget>=1000"
            hc, hb = hit(q, c), hit(q, comp)
            ev = (
                "no_text_evidence"
                if hc is None
                else "chosen_misses_evidence_comp_has"
                if (not hc and hb)
                else "both_have_evidence"
                if (hc and hb)
                else "neither_has_evidence"
                if (not hc and not hb)
                else "chosen_has_comp_misses"
            )
            cats[(budget, strat, ev)] += 1
        n = len(losses)
        breakdown[f"A1_vs_{comp}"] = {
            "losses": n,
            "wins": sum(dv.f1[q][a1_choice[q]] > dv.f1[q][comp] for q in dv.qs),
            "by_budget": {k: v for k, v in Counter(b for (b, _, _) in cats.elements()).items()},
            "by_strategy": dict(Counter(s for (_, s, _) in cats.elements())),
            "by_evidence": dict(Counter(e for (_, _, e) in cats.elements())),
            "lost_f1_sum_by_budget_strategy": {
                f"{b}|{st}": round(
                    sum(
                        dv.f1[q][comp] - dv.f1[q][a1_choice[q]]
                        for q in losses
                        if (
                            (
                                "budget<1000"
                                if int(a1_choice[q].split("@")[1]) < 1000
                                else "budget>=1000"
                            )
                            == b
                            and a1_choice[q].split("@")[0] == st
                        )
                    )
                    / len(dv.qs),
                    4,
                )
                for b in ("budget<1000", "budget>=1000")
                for st in ("leaf_only", "collapsed", "summary_only")
            },
        }
    rep["a1_loss_breakdown_dev"] = breakdown
    rep["leakage_audit"] = {
        "A1/A2 inference features": "LLM profile of (question, title) only; prompt profile_v3",
        "gold answers / answer_type / evidence used as features": False,
        "gold used for": "training labels (F1) and diagnosis tables only",
        "earlier 'leaf budget by answer_form' check": "used the LLM-predicted answer_form "
        "(available at inference, costs the profiler call), fit on train labels; the gold "
        "answer_type table was diagnostic only",
        "dev use": "A1 theta and comparators were selected on dev (optimistic)",
    }

    # ---------------- step 4: summary statistics
    st = np.array(summary_tokens)
    rep["summary_bottleneck_dev_trees"] = {
        "n_summaries": int(len(st)),
        "median_tokens": float(np.median(st)),
        "share_ge_190_tokens": round(float((st >= 190).mean()), 3),
        "share_not_ending_in_sentence_punct": round(float(np.mean(summary_unfinished)), 3),
    }
    disp = defaultdict(list)
    for q in dv.qs:
        lv = dv.levels[q]["collapsed@1000"]
        frac = np.mean([x >= 1 for x in lv]) if lv else 0
        b = "0" if frac == 0 else "<=20%" if frac <= 0.2 else "<=40%" if frac <= 0.4 else ">40%"
        disp[b].append(
            (
                dv.f1[q]["collapsed@1000"] - dv.f1[q]["leaf_only@1000"],
                hit(q, "collapsed@1000"),
                hit(q, "leaf_only@1000"),
            )
        )
    rep["collapsed1000_summary_share_vs_leaf1000"] = {
        b: {
            "n": len(v),
            "mean_f1_delta": round(float(np.mean([x[0] for x in v])), 4),
            "evidence_cov_collapsed": round(
                float(np.mean([x[1] for x in v if x[1] is not None])), 3
            ),
            "evidence_cov_leaf": round(float(np.mean([x[2] for x in v if x[2] is not None])), 3),
        }
        for b, v in sorted(disp.items())
    }

    # ---------------- step 2: features + predictability + routers
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1", device="cpu")
    rep["retrieval_feature_reproduction_rate"] = check_retrieval_reproduces(dv, model)
    print("retrieval reproduction:", rep["retrieval_feature_reproduction_rate"])
    ftr = build_features(tr, model, out / "features_train.npz")
    fdv = build_features(dv, model, out / "features_dev.npz")
    rep["feature_names"] = {
        "question": QF_NAMES,
        "retrieval": RF_NAMES,
        "emb": "SBERT query, PCA-16 (fit on train)",
    }
    rep["predictability_auc_dev"] = auc_table(tr, dv, ftr, fdv)
    print(json.dumps(rep["predictability_auc_dev"], indent=1))

    leaf1000 = fixed(dv, "leaf_only@1000")
    routers = []
    for target in (1039, 1164):
        for cfg_name, cfgs in [
            ("leaf_budgets", LEAF),
            ("leaf+collapsed", LEAF + COLL),
            ("all9", ALL9),
        ]:
            for fs_name, fs, prof in [
                ("question", ["qf"], False),
                ("retrieval", ["rf"], False),
                ("question+retrieval+emb", ["qf", "rf", "emb"], False),
                ("all+llm_profile", ["qf", "rf", "emb"], True),
            ]:
                for kind in ("ridge", "gbdt"):
                    ch, ein, eout, lam = run_router(tr, dv, ftr, fdv, cfgs, fs, kind, target, prof)
                    row = system(
                        dv, ch, ein, eout, name=f"{kind}|{cfg_name}|{fs_name}|target{target}"
                    )
                    row["vs_leaf1000"] = compare(dv, ch, leaf1000)
                    vs_matched(dv, row, ch)
                    row["vs_original"] = compare(dv, ch, fixed(dv, ORIG))
                    row["config_mix"] = dict(Counter(ch.values()))
                    routers.append(row)
    rep["routers_dev"] = routers

    # rule-based: escalate leaf budget by top-1 leaf similarity (threshold set on train)
    def rule(split, feats, t_lo, t_hi):
        return {
            q: (
                "leaf_only@2000"
                if feats["rf"][q][0] < t_lo
                else "leaf_only@1000"
                if feats["rf"][q][0] < t_hi
                else "leaf_only@400"
            )
            for q in split.qs
        }

    rules = []
    s1_tr = np.array([ftr["rf"][q][0] for q in tr.qs])
    for lo_q, hi_q in [(0.2, 0.6), (0.33, 0.67), (0.5, 0.9), (0.25, 0.95)]:
        lo, hi = np.quantile(s1_tr, lo_q), np.quantile(s1_tr, hi_q)
        for sign in ("low_sim_escalates", "high_sim_escalates"):
            if sign == "low_sim_escalates":
                ch = rule(dv, fdv, lo, hi)
            else:
                ch = {
                    q: (
                        "leaf_only@2000"
                        if fdv["rf"][q][0] > hi
                        else "leaf_only@1000"
                        if fdv["rf"][q][0] > lo
                        else "leaf_only@400"
                    )
                    for q in dv.qs
                }
            row = system(dv, ch, name=f"rule top1-sim {sign} q=({lo_q},{hi_q})")
            row["vs_leaf1000"] = compare(dv, ch, leaf1000)
            vs_matched(dv, row, ch)
            rules.append(row)
    rep["rule_routers_dev"] = rules

    # ---------------- step 3: cascades from stored answers
    def unans(ans: str) -> bool:
        return bool(
            re.search(
                r"unanswerable|not (mentioned|specified|stated|provided|found)|"
                r"no information|cannot be determined|does not (say|mention|specify)",
                ans.lower(),
            )
        )

    casc = []
    for first, nxt in [
        ("leaf_only@400", "leaf_only@1000"),
        ("leaf_only@400", "leaf_only@2000"),
        ("leaf_only@1000", "leaf_only@2000"),
        ("leaf_only@400", "collapsed@2000"),
    ]:
        ch, ein, eout = {}, {}, {}
        for q in dv.qs:
            if unans(dv.answer[q][first]):
                ch[q] = nxt
                ein[q], eout[q] = dv.tin[q][first], dv.tout[q][first]
            else:
                ch[q] = first
        row = system(dv, ch, ein, eout, name=f"answer cascade: {first} -> {nxt} if abstains")
        row["escalation_rate"] = round(len(ein) / len(dv.qs), 3)
        row["vs_leaf1000"] = compare(dv, ch, leaf1000)
        row["vs_original"] = compare(dv, ch, fixed(dv, ORIG))
        vs_matched(dv, row, ch)
        casc.append(row)
    # three-stage 400 -> 1000 -> 2000
    ch, ein, eout = {}, defaultdict(int), defaultdict(int)
    for q in dv.qs:
        cur = "leaf_only@400"
        for nxt in ("leaf_only@1000", "leaf_only@2000"):
            if unans(dv.answer[q][cur]):
                ein[q] += dv.tin[q][cur]
                eout[q] += dv.tout[q][cur]
                cur = nxt
        ch[q] = cur
    row = system(dv, ch, ein, eout, name="answer cascade: 400 -> 1000 -> 2000 while abstaining")
    row["escalation_rate"] = round(
        sum(1 for q in dv.qs if ch[q] != "leaf_only@400") / len(dv.qs), 3
    )
    row["vs_leaf1000"] = compare(dv, ch, leaf1000)
    row["vs_original"] = compare(dv, ch, fixed(dv, ORIG))
    vs_matched(dv, row, ch)
    casc.append(row)
    # how good is abstention as an escalation signal?
    ab = [q for q in dv.qs if unans(dv.answer[q]["leaf_only@400"])]
    rep["abstention_signal_dev"] = {
        "leaf400_abstains": round(len(ab) / len(dv.qs), 3),
        "gold_unanswerable_among_abstentions": round(
            float(np.mean([dv.atype[q] == "none" for q in ab])), 3
        ),
        "mean_gain_leaf2000_minus_leaf400_when_abstains": round(
            float(np.mean([dv.f1[q]["leaf_only@2000"] - dv.f1[q]["leaf_only@400"] for q in ab])), 4
        ),
        "mean_gain_when_answers": round(
            float(
                np.mean(
                    [
                        dv.f1[q]["leaf_only@2000"] - dv.f1[q]["leaf_only@400"]
                        for q in dv.qs
                        if q not in set(ab)
                    ]
                )
            ),
            4,
        ),
    }
    rep["cascades_dev"] = casc

    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()

"""Sentence-level extractive compression, Phase 1: offline feasibility. NO API CALLS.

    uv run --group plots python scripts/compression_feasibility.py \
        [--config configs/analysis/compression_feasibility.yaml] [--splits train dev] [--no-plots]

For every question of the label runs (train, dev; the test split is never read):

1. Take the stored leaf_only@2000 retrieval: node ids in rank order, from traces.jsonl.
   The leaf text comes from the cached RAPTOR tree. Nothing is retrieved, indexed or
   generated again.
2. Compressed contexts (adaptive_raptor.compression): sentence units of those leaves,
   scored by BM25 / SBERT cosine / RRF hybrid / random, selected greedily under
   600 / 1000 / 1400 tokens, with and without the predecessor rule, in document order.
3. Chunk baselines: the same ranked leaves truncated with upstream's rule (stop at the
   first chunk that no longer fits), at 400 / 600 / 1000 / 1400 / 2000 tokens. They
   equal leaf_only@B; this is checked against the stored leaf_only@400 / 1000 runs.
4. Gold-evidence coverage of every context (gold data is used only here, never to
   select sentences):
   - hl_recall: highlighted-evidence recall, the primary measure. Per annotator, the share
     of content tokens of their highlighted sentences that appear in the context; the
     best annotator counts.
   - para_recall: best single evidence-paragraph recall (as in analyze_summary_cap).
   - para_hit50: some evidence paragraph is at least 50% covered (offline_diagnostics).
   - answer_in_ctx: an extractive gold answer appears verbatim (normalized) in the context.
   - context_tokens: cl100k tokens of the context string as the generator would receive it.
5. Paired document bootstrap (95%) deltas: compressed@B vs chunk@B (same budget), and
   compressed vs leaf_only@1000 / leaf_only@2000.

The pre-registered decision rule is in the YAML config. Outputs go to a new timestamped
directory under ``out_dir``: report.json, table_<split>.csv, table_dev.md,
per_question_<split>.jsonl and plots. Sentence embeddings are cached in
``out_dir/cache`` (local SBERT, no API).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tiktoken
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import LEAF, STOP, Split, boot, evidence_hit, load_tree  # noqa: E402

from adaptive_raptor.compression import (  # noqa: E402
    BM25,
    CompressionSettings,
    Unit,
    assemble,
    pool_units,
    rrf,
    select,
    sentence_units,
)

WORD = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------- gold data


def load_gold(path: Path, qs: set[str]) -> dict[str, dict]:
    """Per question: highlighted sentences per annotator, evidence paragraphs, extractive
    answers."""
    out = {}
    for line in path.open():
        r = json.loads(line)
        if r["question_id"] not in qs:
            continue
        hl, ext = [], []
        for a in r["answers"]:
            sents = [s for s in a.get("highlighted_evidence", []) if s.strip()]
            sents = [s for s in sents if not s.startswith("FLOAT SELECTED")]
            if sents:
                hl.append(sents)
            if a.get("answer_type") == "extractive" and a.get("text", "").strip():
                ext.append(a["text"])
        out[r["question_id"]] = {"hl": hl, "extractive": ext}
    return out


def ctoks(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def hl_recall(annotators: list[list[str]], have: set[str]) -> float | None:
    best = None
    for sents in annotators:
        toks = [t for s in sents for t in ctoks(s)]
        if toks:
            r = sum(t in have for t in toks) / len(toks)
            best = r if best is None else max(best, r)
    return best


def para_recall(evidence: list[str], have: set[str]) -> float | None:
    ev = [e for e in evidence if not e.startswith("FLOAT SELECTED")]
    best = None
    for e in ev:
        toks = ctoks(e)
        if toks:
            r = sum(t in have for t in toks) / len(toks)
            best = r if best is None else max(best, r)
    return best


def norm_words(text: str) -> str:
    return " " + " ".join(WORD.findall(text.lower())) + " "


def answer_in_ctx(answers: list[str], ctx_norm: str) -> float | None:
    if not answers:
        return None
    return float(any(norm_words(a) in ctx_norm for a in answers if norm_words(a).strip()))


def evaluate(ctx: str, q: str, s: Split, gold: dict, encode) -> dict:
    have = set(ctoks(ctx))
    hit = evidence_hit(s.evidence.get(q, []), ctx)
    return {
        "context_tokens": len(encode(ctx)),
        "hl_recall": hl_recall(gold[q]["hl"], have),
        "para_recall": para_recall(s.evidence.get(q, []), have),
        "para_hit50": None if hit is None else float(hit),
        "answer_in_ctx": answer_in_ctx(gold[q]["extractive"], norm_words(ctx)),
    }


# ---------------------------------------------------------------------------- contexts


def chunk_truncate(ranked: list[int], tok: dict[int, int], budget: int) -> list[int]:
    """Upstream retrieve_information_collapse_tree: stop at the first chunk that does not fit."""
    out, used = [], 0
    for i in ranked:
        if used + tok[i] > budget:
            break
        out.append(i)
        used += tok[i]
    return out


def embed_units(split: str, s: Split, cfg: CompressionSettings, cache_dir: Path):
    """SBERT embeddings of all units of the leaves in any leaf_only@2000 pool, per paper.
    Cached on disk (local model, no API)."""
    cache = cache_dir / f"{split}_units_{cfg.embedding_model.split('/')[-1]}.npz"
    qcache = cache_dir / f"{split}_questions_{cfg.embedding_model.split('/')[-1]}.npz"
    if cache.exists() and qcache.exists():
        z, zq = np.load(cache, allow_pickle=True), np.load(qcache, allow_pickle=True)
        return z["emb"].item(), zq["emb"].item()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.embedding_model, device="cpu")
    need: dict[str, set[int]] = defaultdict(set)
    for q in s.qs:
        need[s.doc[q]].update(s.nodes[q][cfg.source_config])
    keys, texts = [], []
    for doc, leaves in sorted(need.items()):
        tree = load_tree(doc, s.index_hash[doc])
        for i in sorted(leaves):
            for u in sentence_units(i, tree.all_nodes[i].text):
                keys.append(f"{doc}|{u.leaf}|{u.pos}")
                texts.append(u.text)
    t0 = time.time()
    print(f"[{split}] encoding {len(texts)} sentence units locally ...", flush=True)
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    print(f"[{split}] encoded in {time.time() - t0:.0f}s", flush=True)
    qemb = model.encode([s.question[q] for q in s.qs], normalize_embeddings=True)
    out = {k: e.astype(np.float32) for k, e in zip(keys, emb, strict=True)}
    qout = {q: e.astype(np.float32) for q, e in zip(s.qs, qemb, strict=True)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, emb=np.array(out, dtype=object))
    np.savez_compressed(qcache, emb=np.array(qout, dtype=object))
    return out, qout


def variants(cfg: CompressionSettings) -> list[tuple[str, bool]]:
    return [(sc, nb) for sc in cfg.scorers for nb in cfg.neighbors]


def vname(sc: str, nb: bool, b: int) -> str:
    return f"{sc}{'+nb' if nb else ''}@{b}"


def run_split(split: str, cfg: CompressionSettings, cache_dir: Path) -> tuple[dict, list[dict]]:
    run = Path(cfg.runs[split])
    qfile = Path(f"data/processed/qasper/v0.3/{split}/questions.jsonl")
    s = Split(run, None, qfile, configs=LEAF)
    gold = load_gold(qfile, set(s.qs))
    enc = tiktoken.get_encoding(cfg.tokenizer).encode
    uemb, qemb = embed_units(split, s, cfg, cache_dir)
    rng = np.random.default_rng(cfg.seed)

    rows: list[dict] = []
    check = {"leaf_only@400": 0, "leaf_only@1000": 0}
    by_doc: dict[str, list[str]] = defaultdict(list)
    for q in s.qs:
        by_doc[s.doc[q]].append(q)
    for doc, qs in sorted(by_doc.items()):
        tree = load_tree(doc, s.index_hash[doc])
        all_units = [u for n in tree.layer_to_nodes[0] for u in sentence_units(n.index, n.text)]
        bm25 = BM25([u.text for u in all_units], cfg.bm25_k1, cfg.bm25_b)
        for q in qs:
            ranked = s.nodes[q][cfg.source_config]
            text = {i: tree.all_nodes[i].text for i in ranked}
            tok = {i: len(enc(text[i])) for i in ranked}
            row = {"question_id": q, "doc_id": doc, "methods": {}}
            for b in cfg.chunk_baselines:
                ids = chunk_truncate(ranked, tok, b)
                ctx = "\n\n".join(" ".join(text[i].splitlines()) for i in ids)
                row["methods"][f"chunk@{b}"] = {**evaluate(ctx, q, s, gold, enc), "units": None}
                if f"leaf_only@{b}" in check:
                    check[f"leaf_only@{b}"] += ids == s.nodes[q][f"leaf_only@{b}"]
            units: list[Unit] = pool_units([(i, text[i]) for i in ranked])
            if not units:
                continue
            ut = [u.text for u in units]
            sc = {
                "bm25": bm25.score(s.question[q], ut),
                "sbert": np.array([uemb[f"{doc}|{u.leaf}|{u.pos}"] @ qemb[q] for u in units]),
                "random": rng.random(len(units)),
            }
            sc["hybrid"] = rrf(sc["bm25"], sc["sbert"], k=cfg.rrf_k)
            for name, nb in variants(cfg):
                for b in cfg.budgets:
                    picked = select(
                        units,
                        sc[name],
                        b,
                        enc,
                        redundancy_jaccard=cfg.redundancy_jaccard,
                        neighbors=nb,
                        dependent_max_words=cfg.dependent_max_words,
                    )
                    ctx = assemble(picked)
                    row["methods"][vname(name, nb, b)] = {
                        **evaluate(ctx, q, s, gold, enc),
                        "units": len(picked),
                    }
            row["pool_units"] = len(units)
            rows.append(row)
    n = len(rows)
    meta = {
        "run": str(run),
        "n_questions": n,
        "n_papers": len(by_doc),
        "n_with_highlighted_evidence": sum(
            r["methods"]["chunk@2000"]["hl_recall"] is not None for r in rows
        ),
        "n_with_extractive_answer": sum(
            r["methods"]["chunk@2000"]["answer_in_ctx"] is not None for r in rows
        ),
        "mean_pool_units": round(float(np.mean([r["pool_units"] for r in rows])), 1),
        "chunk_baseline_reproduces_stored": {k: round(v / n, 4) for k, v in check.items()},
    }
    return meta, rows


# ---------------------------------------------------------------------------- aggregation

METRICS = ["hl_recall", "para_recall", "para_hit50", "answer_in_ctx"]


def summarize(rows: list[dict]) -> dict[str, dict]:
    ref = np.mean([r["methods"]["chunk@2000"]["context_tokens"] for r in rows])
    out = {}
    for m in rows[0]["methods"]:
        vals = [r["methods"][m] for r in rows]
        tokens = float(np.mean([v["context_tokens"] for v in vals]))
        d = {
            "context_tokens": round(tokens, 1),
            "token_reduction_vs_leaf2000": round(1 - tokens / ref, 4),
        }
        for k in METRICS:
            xs = [v[k] for v in vals if v[k] is not None]
            d[k] = round(float(np.mean(xs)), 4)
            d[f"n_{k}"] = len(xs)
        out[m] = d
    return out


def delta(rows: list[dict], a: str, b: str, metric: str) -> list[float] | None:
    d, doc = {}, {}
    for r in rows:
        x, y = r["methods"][a][metric], r["methods"][b][metric]
        if x is not None and y is not None:
            d[r["question_id"]] = x - y
            doc[r["question_id"]] = r["doc_id"]
    return boot(d, doc) if d else None


def comparisons(rows, cfg, best: tuple[str, bool]) -> dict:
    out = {}
    for sc, nb in variants(cfg):
        for b in cfg.budgets:
            m = vname(sc, nb, b)
            out[m] = {
                f"{metric} vs {ref}": delta(rows, m, ref, metric)
                for ref in (f"chunk@{b}", "chunk@1000", "chunk@2000")
                for metric in ("hl_recall", "answer_in_ctx", "para_recall")
            }
    for b in cfg.budgets:  # BM25 vs embedding scoring, same neighbor setting as best
        out[f"bm25 vs sbert @{b}"] = {
            metric: delta(rows, vname("bm25", best[1], b), vname("sbert", best[1], b), metric)
            for metric in ("hl_recall", "answer_in_ctx")
        }
    return out


def decide(summary: dict, comp: dict, best: tuple[str, bool]) -> dict:
    m1000, m1400 = vname(*best, 1000), vname(*best, 1400)
    hl_d = comp[m1000]["hl_recall vs chunk@1000"]
    ans_d = comp[m1000]["answer_in_ctx vs chunk@1000"]
    a = bool(hl_d and hl_d[1] > 0 and ans_d and ans_d[1] > -0.01)
    keep = summary[m1400]["hl_recall"] / summary["chunk@2000"]["hl_recall"]
    red = summary[m1400]["token_reduction_vs_leaf2000"]
    b = bool(keep >= 0.95 and red >= 0.25)
    return {
        "selected_variant (train, max hl_recall@1000)": vname(*best, 0)[:-2],
        "(a) compressed@1000 beats leaf_only@1000": {
            "hl_recall delta": hl_d,
            "answer_in_ctx delta": ans_d,
            "met": a,
        },
        "(b) compressed@1400 keeps >=95% of leaf_only@2000 hl_recall with >=25% fewer tokens": {
            "hl_recall share kept": round(keep, 4),
            "token reduction": red,
            "met": b,
        },
        "phase2_justified": a or b,
    }


# ---------------------------------------------------------------------------- output


def write_tables(out: Path, split: str, summary: dict) -> None:
    cols = ["context_tokens", "token_reduction_vs_leaf2000", *METRICS]
    with (out / f"table_{split}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", *cols])
        for m, d in summary.items():
            w.writerow([m, *(d[c] for c in cols)])


def write_md(out: Path, rep: dict, cfg: CompressionSettings) -> None:
    sm = rep["dev"]["summary"]
    lines = [
        "| Method | Context tokens | Reduction vs leaf@2000 | Highlighted recall | "
        "Paragraph recall | Paragraph hit ≥50% | Answer in context |",
        "|---|---|---|---|---|---|---|",
    ]
    order = [f"chunk@{b}" for b in cfg.chunk_baselines] + [
        vname(sc, nb, b) for sc, nb in variants(cfg) for b in cfg.budgets
    ]
    for m in order:
        d = sm[m]
        lines.append(
            f"| {m} | {d['context_tokens']:.0f} | {d['token_reduction_vs_leaf2000']:.1%} | "
            f"{d['hl_recall']:.3f} | {d['para_recall']:.3f} | {d['para_hit50']:.3f} | "
            f"{d['answer_in_ctx']:.3f} |"
        )
    (out / "table_dev.md").write_text("\n".join(lines) + "\n")


def plots(out: Path, rep: dict, cfg: CompressionSettings, best: tuple[str, bool]) -> None:
    import matplotlib.pyplot as plt
    from make_dev_charts import INK, INK_2, MUTED, save, style

    style()
    colors = {"bm25": "#2a78d6", "sbert": "#d6762a", "hybrid": "#3a9a5b", "random": "#8a8984"}
    for split in rep["splits"]:
        sm = rep[split]["summary"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, metric, label in (
            (axes[0], "hl_recall", "highlighted-evidence recall"),
            (axes[1], "answer_in_ctx", "extractive answer in context"),
        ):
            xs = [sm[f"chunk@{b}"]["context_tokens"] for b in cfg.chunk_baselines]
            ys = [sm[f"chunk@{b}"][metric] for b in cfg.chunk_baselines]
            ax.plot(xs, ys, color=INK, lw=2, marker="s", ms=6, label="chunk-level (leaf_only@B)")
            for b, x, y in zip(cfg.chunk_baselines, xs, ys, strict=True):
                ax.annotate(
                    f"{b}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, -12),
                    fontsize=7.5,
                    color=INK_2,
                )
            for sc in cfg.scorers:
                m = [vname(sc, best[1], b) for b in cfg.budgets]
                ax.plot(
                    [sm[k]["context_tokens"] for k in m],
                    [sm[k][metric] for k in m],
                    color=colors[sc],
                    lw=2,
                    marker="o",
                    ms=7,
                    ls="--" if sc == "random" else "-",
                    label=f"sentence-level, {sc}",
                )
            ax.set_xlabel("mean context tokens")
            ax.set_ylabel(label)
            ax.set_title(label, fontsize=10)
        axes[1].legend(fontsize=8, loc="lower right")
        nb = "with" if best[1] else "without"
        fig.suptitle(
            f"Coverage vs context size ({split}, n={rep[split]['meta']['n_questions']}; "
            f"sentence selection {nb} predecessor rule)",
            x=0.01,
            ha="left",
            fontweight="bold",
            color=INK,
        )
        fig.tight_layout()
        save(fig, out, f"coverage_vs_tokens_{split}")

    # dev: best variant vs chunk@B, chunk@1000 and chunk@2000
    comp = rep["dev"]["comparisons"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels, y = [], 0
    for b in cfg.budgets:
        m = vname(*best, b)
        for ref, col in ((f"chunk@{b}", colors["bm25"]), ("chunk@2000", MUTED)):
            d = comp[m][f"hl_recall vs {ref}"]
            ax.errorbar(
                d[0],
                y,
                xerr=[[d[0] - d[1]], [d[2] - d[0]]],
                fmt="o",
                color=col,
                ms=7,
                capsize=0,
                lw=2,
            )
            ax.annotate(
                f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]",
                (d[2], y),
                textcoords="offset points",
                xytext=(6, -3),
                fontsize=8,
                color=INK_2,
            )
            labels.append(f"{m} − {ref}")
            y += 1
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ highlighted-evidence recall (95% paired document bootstrap)")
    ax.set_title("Sentence selection vs chunk-level retrieval (dev)")
    xs = ax.get_xlim()
    ax.set_xlim(xs[0], xs[1] + (xs[1] - xs[0]) * 0.45)
    fig.tight_layout()
    save(fig, out, "compression_deltas_dev")
    plt.close(fig)

    # scorer comparison at each budget (dev)
    sm = rep["dev"]["summary"]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / (len(cfg.scorers) * len(cfg.neighbors))
    k = 0
    for sc in cfg.scorers:
        for nb in cfg.neighbors:
            xs = np.arange(len(cfg.budgets)) + k * w - 0.4 + w / 2
            ax.bar(
                xs,
                [sm[vname(sc, nb, b)]["hl_recall"] for b in cfg.budgets],
                width=w * 0.9,
                color=colors[sc],
                alpha=1.0 if nb else 0.55,
                label=f"{sc}{' + predecessor' if nb else ''}",
            )
            k += 1
    for i, b in enumerate(cfg.budgets):
        ax.hlines(
            sm[f"chunk@{b}"]["hl_recall"],
            i - 0.42,
            i + 0.42,
            color=INK,
            lw=1.5,
            label="chunk-level at the same budget" if i == 0 else None,
        )
    ax.set_xticks(range(len(cfg.budgets)), [f"budget {b}" for b in cfg.budgets])
    ax.set_ylabel("highlighted-evidence recall")
    ax.set_ylim(0, 1)
    ax.set_title("Scorers at each budget (dev)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out, "compression_scorers_dev")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--config", type=Path, default=Path("configs/analysis/compression_feasibility.yaml")
    )
    p.add_argument("--splits", nargs="+", default=["train", "dev"], choices=["train", "dev"])
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plots-only", type=Path, help="redraw plots from an existing output dir")
    a = p.parse_args()
    cfg = CompressionSettings(**yaml.safe_load(a.config.read_text()))
    if a.plots_only:
        rep = json.loads((a.plots_only / "report.json").read_text())
        name = rep["decision"]["selected_variant (train, max hl_recall@1000)"]
        plots(a.plots_only, rep, cfg, (name.removesuffix("+nb"), name.endswith("+nb")))
        return
    base = Path(cfg.out_dir)
    out = base / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    rep: dict = {"config": cfg.model_dump(), "splits": a.splits}
    for split in a.splits:
        meta, rows = run_split(split, cfg, base / "cache")
        rep[split] = {"meta": meta, "summary": summarize(rows)}
        with (out / f"per_question_{split}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        write_tables(out, split, rep[split]["summary"])
        print(f"[{split}] {json.dumps(meta)}")
    # variant selection on train (or on dev, flagged, if train was not run)
    sel_split = "train" if "train" in a.splits else "dev"
    sm = rep[sel_split]["summary"]
    cands = [(sc, nb) for sc, nb in variants(cfg) if sc != "random"]
    best = max(cands, key=lambda v: sm[vname(*v, 1000)]["hl_recall"])
    rep["selected_on"] = sel_split
    for split in a.splits:
        rows = [json.loads(x) for x in (out / f"per_question_{split}.jsonl").open()]
        rep[split]["comparisons"] = comparisons(rows, cfg, best)
    if "dev" in a.splits:
        rep["decision"] = decide(rep["dev"]["summary"], rep["dev"]["comparisons"], best)
        write_md(out, rep, cfg)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    if not a.no_plots and "dev" in a.splits:
        plots(out, rep, cfg, best)
    for split in a.splits:
        print(f"== {split} ==")
        for m, d in rep[split]["summary"].items():
            print(
                f"  {m:18} tok {d['context_tokens']:7.1f} "
                f"red {d['token_reduction_vs_leaf2000']:+.1%}"
                f"  hl {d['hl_recall']:.3f} para {d['para_recall']:.3f} hit50 "
                f"{d['para_hit50']:.3f} ans {d['answer_in_ctx']:.3f}"
            )
    print(json.dumps(rep.get("decision"), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

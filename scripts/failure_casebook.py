"""Failure casebook: questions where RAPTOR, the leaf baseline, or our adaptive router fail,
with every system's answer side by side. Offline, no API; reads stored traces only.

    uv run python scripts/failure_casebook.py [--per-category 12] [--out DIR]

Systems (all answers are the recorded ones; nothing is regenerated):
  QASPER dev (918 q, labels run 20261109T195548Z, token F1):
    raptor_original   upstream RAPTOR retrieve() (collapsed, top_k=10, 2000 tokens)
    raptor_paper      collapsed@2000, the paper's main setting
    leaf@1000         fixed leaf-only baseline at the router's cost
    adaptive_rules    our router A2 (LLM profile -> rules), replayed from its frozen
                      checkpoint; replay is checked against the stored dev F1
    adaptive_table    our router A1 (LLM profile -> profile table), same replay
  NarrativeQA pilot (594 q, run A; ROUGE-L max over 2 refs):
    raptor_paper      collapsed@2000
    leaf@2000         leaf-only at the same budget
    summary@2000      summary nodes only
    (no adaptive router has been trained on NarrativeQA; run B answers are attached to show
     how much an answer moves when only the prompt wording changes)

Fail = score < 0.3 (pre-set, not tuned). Every question failed by at least one system is
written to cases.jsonl with all answers, the chosen config, the retrieved nodes (layer +
text) and two automatic tags:
  answer_in_context  some reference answer has >= 50% content-token recall in the
                     retrieved context (else the failure is likely retrieval, not reading)
  evidence_in_context  QASPER only: a gold evidence paragraph is >= 50% recalled
  abstained          the model said the information is not in the context
casebook.md shows counts by category x tag and a fixed-seed sample per category.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import ALL9, Split, evidence_hit, load_tree  # noqa: E402

FAIL = 0.3
QASPER_RUN = Path("outputs/labels_dev/20261109T195548Z")
QASPER_Q = Path("data/processed/qasper/v0.3/dev/questions.jsonl")
PROFILES = Path("outputs/profile_dev/20261109T195035Z/profiles.jsonl")
ROUTERS = {
    "adaptive_rules": Path("outputs/router_train_llm_profile_rules/20261109T230228Z"),
    "adaptive_table": Path("outputs/router_train_llm_profile_table/20261109T230225Z"),
}
NQA_A = Path("outputs/narrativeqa_pilot/20261111T153658Z")
NQA_B = Path("outputs/narrativeqa_pilot/20261111T175627Z")
NQA_Q = Path("data/processed/narrativeqa/2e643e73/dev/questions.jsonl")
NQA_IDX = Path("indexes/narrativeqa/2e643e73/f100/seed224")
ABSTAIN = re.compile(
    r"\b(not (mentioned|provided|specified|stated|given|included|discussed|available)|"
    r"unanswerable|cannot be (determined|answered)|no information|does not (say|mention|"
    r"specify|provide|state))\b",
    re.I,
)


def replay_router(run: Path, s: Split) -> dict[str, str]:
    from adaptive_raptor.config.loader import load_config
    from adaptive_raptor.optimizer import CostModel, PlanSelector, _model_from_dict
    from adaptive_raptor.optimizer.profiler import QueryProfile

    cfg = load_config(run / "resolved_config.yaml")
    ckpt = json.loads((run / "router" / "checkpoint.json").read_text())
    catalog = cfg.retrieval.catalog()
    model = _model_from_dict(ckpt["artifacts"]["model"], catalog)
    sel = PlanSelector(
        catalog, CostModel.from_dict(ckpt["artifacts"]["cost_model"]), cfg.optimizer.fallback
    )
    prof = {}
    for line in PROFILES.open():
        r = json.loads(line)
        if r["prompt_hash"] == ckpt["artifacts"]["prompt_hash"] and r["profile"]:
            prof[r["question_id"]] = QueryProfile(**r["profile"])
    choice = {}
    for q in s.qs:
        probs = model.predict(prof[q]) if q in prof else None
        choice[q] = sel.select(probs, ckpt["theta"]).config.id
    want = json.loads((run / "dev_report.json").read_text())["router"]["answer_f1"]
    got = sum(s.f1[q][choice[q]] for q in s.qs) / len(s.qs)
    if abs(got - want) > 5e-4:
        raise SystemExit(f"router replay mismatch for {run}: {got:.4f} vs stored {want}")
    return choice


def node_texts(tree, ids, levels):
    return [
        {"node": i, "layer": lv, "text": tree.all_nodes[i].text} for i, lv in zip(ids, levels)
    ]


def tags(answer: str, ctx: str, refs: list[str], evidence: list[str] | None) -> dict:
    t = {
        "answer_in_context": evidence_hit([r for r in refs if r.strip()], ctx),
        "abstained": bool(ABSTAIN.search(answer or "")),
    }
    if evidence is not None:
        t["evidence_in_context"] = evidence_hit(evidence, ctx)
    return t


def qasper_cases() -> tuple[list[dict], dict]:
    s = Split(QASPER_RUN, None, QASPER_Q, configs=ALL9 + ["raptor_original"])
    refs = {}
    for line in QASPER_Q.open():
        r = json.loads(line)
        refs[r["question_id"]] = [
            a["text"] if a["answer_type"] != "none" else "Unanswerable" for a in r["answers"]
        ]
    systems = {
        "raptor_original": {q: "raptor_original" for q in s.qs},
        "raptor_paper": {q: "collapsed@2000" for q in s.qs},
        "leaf@1000": {q: "leaf_only@1000" for q in s.qs},
    }
    for name, run in ROUTERS.items():
        systems[name] = replay_router(run, s)
    profile = {}
    for line in PROFILES.open():
        r = json.loads(line)
        profile[r["question_id"]] = r["profile"]
    cases = []
    trees = {}
    for q in s.qs:
        scores = {n: s.f1[q][ch[q]] for n, ch in systems.items()}
        if min(scores.values()) >= FAIL:
            continue
        doc = s.doc[q]
        if doc not in trees:
            trees[doc] = load_tree(doc, s.index_hash[doc])
        out = {}
        for n, ch in systems.items():
            c = ch[q]
            nodes = node_texts(trees[doc], s.nodes[q][c], s.levels[q][c])
            ctx = "\n\n".join(x["text"] for x in nodes)
            out[n] = {
                "config": c,
                "answer": s.answer[q][c],
                "score": round(s.f1[q][c], 4),
                "failed": s.f1[q][c] < FAIL,
                "summary_node_share": round(
                    sum(x["layer"] > 0 for x in nodes) / max(1, len(nodes)), 3
                ),
                **tags(s.answer[q][c], ctx, refs[q], s.evidence.get(q, [])),
                "context": nodes,
            }
        cases.append(
            {
                "dataset": "qasper",
                "question_id": q,
                "doc_id": doc,
                "question": s.question[q],
                "gold": refs[q],
                "answer_type": s.atype[q],
                "profile": profile.get(q),
                "best_of_9": max(s.f1[q][c] for c in ALL9),
                "systems": out,
            }
        )
    meta = {
        "n": len(s.qs),
        "metric": "token F1 (QASPER official)",
        "fail_rate": {
            n: round(sum(s.f1[q][ch[q]] < FAIL for q in s.qs) / len(s.qs), 4)
            for n, ch in systems.items()
        },
        "mean_score": {
            n: round(sum(s.f1[q][ch[q]] for q in s.qs) / len(s.qs), 4)
            for n, ch in systems.items()
        },
        "pairs": [("adaptive_rules", "raptor_original"), ("raptor_paper", "leaf@1000")],
    }
    return cases, meta


def nqa_cases() -> tuple[list[dict], dict]:
    def load(run):
        t = defaultdict(dict)
        for line in (run / "traces.jsonl").open():
            r = json.loads(line)
            t[r["question_id"]][r["config_id"]] = r
        return t

    A, B = load(NQA_A), load(NQA_B)
    refs = {}
    for line in NQA_Q.open():
        r = json.loads(line)
        if r["question_id"] in A:
            refs[r["question_id"]] = [a["text"] for a in r["answers"]]
    systems = {
        "raptor_paper": "collapsed@2000",
        "leaf@2000": "leaf_only@2000",
        "summary@2000": "summary_only@2000",
    }
    qs = sorted(A)
    trees, cases = {}, []
    for q in qs:
        scores = {n: A[q][c]["answer_rouge_l"] for n, c in systems.items()}
        if min(scores.values()) >= FAIL:
            continue
        t0 = A[q]["leaf_only@2000"]
        doc = t0["doc_id"]
        if doc not in trees:
            path = NQA_IDX / f"{doc}__{t0['index_key_hash'][:16]}.pkl"
            with path.open("rb") as f:
                trees[doc] = pickle.load(f)
        out = {}
        for n, c in systems.items():
            t = A[q][c]
            nodes = node_texts(trees[doc], t["node_ids"], t["levels"])
            ctx = "\n\n".join(x["text"] for x in nodes)
            out[n] = {
                "config": c,
                "answer": t["answer"],
                "score": round(t["answer_rouge_l"], 4),
                "failed": t["answer_rouge_l"] < FAIL,
                "answer_run_b": B[q][c]["answer"],
                "score_run_b": round(B[q][c]["answer_rouge_l"], 4),
                "summary_node_share": round(
                    sum(x["layer"] > 0 for x in nodes) / max(1, len(nodes)), 3
                ),
                **tags(t["answer"], ctx, refs[q], None),
                "context": nodes,
            }
        cases.append(
            {
                "dataset": "narrativeqa",
                "question_id": q,
                "doc_id": doc,
                "question": t0["question"],
                "gold": refs[q],
                "best_of_9": max(A[q][c]["answer_rouge_l"] for c in ALL9),
                "systems": out,
            }
        )
    meta = {
        "n": len(qs),
        "metric": "ROUGE-L F1, max over 2 references",
        "fail_rate": {
            n: round(sum(A[q][c]["answer_rouge_l"] < FAIL for q in qs) / len(qs), 4)
            for n, c in systems.items()
        },
        "mean_score": {
            n: round(sum(A[q][c]["answer_rouge_l"] for q in qs) / len(qs), 4)
            for n, c in systems.items()
        },
        "pairs": [("raptor_paper", "leaf@2000"), ("raptor_paper", "summary@2000")],
    }
    return cases, meta


def category(case: dict, a: str, b: str) -> str:
    fa, fb = case["systems"][a]["failed"], case["systems"][b]["failed"]
    if fa and fb:
        return "both_fail_nothing_works" if case["best_of_9"] < FAIL else "both_fail"
    if fa:
        return f"only_{a}_fails"
    if fb:
        return f"only_{b}_fails"
    return "other_system_fails"


def short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def render(name: str, cases: list[dict], meta: dict, k: int, rng: random.Random) -> list[str]:
    sysnames = list(cases[0]["systems"]) if cases else []
    md = [f"## {name}", ""]
    md.append(f"{meta['n']} questions; metric {meta['metric']}; fail = score < {FAIL}.")
    md.append("")
    md.append("| System | Mean score | Fail rate |")
    md.append("|---|---|---|")
    for n in sysnames:
        md.append(f"| {n} | {meta['mean_score'][n]:.3f} | {meta['fail_rate'][n]:.1%} |")
    md.append("")
    for a, b in meta["pairs"]:
        cats = defaultdict(list)
        for c in cases:
            cats[category(c, a, b)].append(c)
        cats.pop("other_system_fails", None)
        md += [f"### {a} vs {b}", ""]
        md.append(
            "| Category | n | answer in A's context | answer in B's context | A abstained "
            "| B abstained |"
        )
        md.append("|---|---|---|---|---|---|")
        for cat in sorted(cats):
            cs = cats[cat]

            def rate(key, who, cs=cs):
                v = [c["systems"][who][key] for c in cs if c["systems"][who][key] is not None]
                return f"{sum(v) / len(v):.0%}" if v else "n/a"

            md.append(
                f"| {cat} | {len(cs)} | {rate('answer_in_context', a)} | "
                f"{rate('answer_in_context', b)} | {rate('abstained', a)} | {rate('abstained', b)} |"
            )
        md.append("")
        for cat in sorted(cats):
            sample = rng.sample(cats[cat], min(k, len(cats[cat])))
            md += [f"#### {a} vs {b}: {cat} ({len(cats[cat])} total, {len(sample)} shown)", ""]
            for c in sample:
                md.append(f"**{c['question_id']}**: {c['question']}")
                md.append("")
                md.append(f"- Gold: {' | '.join(short(g, 200) for g in c['gold'])}")
                if c.get("profile"):
                    p = c["profile"]
                    md.append(
                        f"- Profile: {p.get('answer_form')}, {p.get('granularity')}, "
                        f"n_facts {p.get('n_facts')}"
                    )
                md.append(f"- Best of 9 configs: {c['best_of_9']:.2f}")
                for n, v in c["systems"].items():
                    flag = "FAIL" if v["failed"] else "ok"
                    extra = []
                    if v["answer_in_context"] is not None:
                        extra.append("ans-in-ctx" if v["answer_in_context"] else "ans-NOT-in-ctx")
                    if v.get("evidence_in_context") is not None:
                        extra.append("ev-in-ctx" if v["evidence_in_context"] else "ev-NOT-in-ctx")
                    if v["abstained"]:
                        extra.append("abstained")
                    extra.append(f"{v['summary_node_share']:.0%} summary nodes")
                    line = (
                        f"- `{n}` [{v['config']}] {flag} {v['score']:.2f} ({', '.join(extra)}): "
                        f"{short(v['answer'], 300)}"
                    )
                    if "answer_run_b" in v:
                        line += f"  — run B {v['score_run_b']:.2f}: {short(v['answer_run_b'], 150)}"
                    md.append(line)
                md.append("")
    return md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--per-category", type=int, default=12)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or Path("outputs/diagnostics/failure_casebook") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    md = [
        "# Failure casebook",
        "",
        "Offline; recorded answers only; dev data (QASPER dev, NarrativeQA dev pilot). "
        "Tags are automatic heuristics (50% content-token recall), meant for triage, "
        "not as labels. Full records incl. retrieved node texts: `cases.jsonl`.",
        "",
    ]
    summary = {}
    with (out / "cases.jsonl").open("w") as f:
        for i, (name, fn) in enumerate(
            (("QASPER dev", qasper_cases), ("NarrativeQA dev pilot (run A)", nqa_cases))
        ):
            cases, meta = fn()
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            md += render(name, cases, meta, a.per_category, random.Random(i))
            summary[name] = {
                **meta,
                "n_cases": len(cases),
                "categories": {
                    f"{x}_vs_{y}": dict(Counter(category(c, x, y) for c in cases))
                    for x, y in meta["pairs"]
                },
            }
    (out / "casebook.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

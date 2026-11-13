"""Sufficiency-triggered cascade: calibrate on train, evaluate once on dev. Offline.

    uv run python scripts/analyze_cascade.py \
        --stage1-train outputs/cascade_stage1_train/<run> \
        --stage1-dev outputs/cascade_stage1_dev/<run>
    uv run python scripts/analyze_cascade.py --simulate     # dry run, no stage-1 runs needed

Stage 1 = leaf_only@400 or @1000 answered with qasper_sufficiency_v1 (answer + evidence flag,
one call). Stage 2 = the stored qasper_default answer of a larger leaf budget from the run-A
label tables (the same prompt as the fixed baselines). Costs count BOTH calls (stage-1 output
includes the flag tokens).

Protocol (pre-registered, docs/NEXT_EXPERIMENTS.md):
  * a fixed candidate list (stage-1 budget x trigger x escalation target);
  * the PRIMARY candidate is the one with the best train delta vs the matched-token leaf mix
    (train labels only); dev evaluates the primary once; other candidates are secondary;
  * the trigger uses only inference-time signals: the stage-1 flag and the stage-1 answer text;
  * comparators: leaf_only@1000, leaf_only@2000, leaf mix at matched tokens, leaf mix at matched
    USD, and the abstention cascade on the stored run-A answers (the earlier design);
  * USD without provider caching is primary; prefix-cache savings are a sensitivity analysis
    using a measured hit fraction (scripts/probe_prompt_cache.py), never assumed.
--simulate replaces the stage-1 runs by the stored run-A answers with flag = insufficient iff
the answer abstains; it only checks the code (it reproduces the abstention cascade).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import LEAF, Split, boot, latest  # noqa: E402

PRICE_MISS, PRICE_HIT, PRICE_OUT = 0.15, 0.003, 0.60  # deepseek-flash off-peak, USD per 1M
INSTRUCTION_TOKENS = 45  # qasper_default instructions + "Context:" before the chunks (tiktoken)
SCORE = {"sufficient": 0, "partial": 1, "insufficient": 2, None: 2}  # missing flag -> escalate
TRIGGERS = (
    "insufficient",
    "insufficient|partial",
    "abstain",
    "abstain|insufficient",
    "abstain|insufficient|partial",
)


def abstains(ans: str) -> bool:
    return bool(
        re.search(
            r"unanswerable|not (mentioned|specified|stated|provided|found)|"
            r"no information|cannot be determined|does not (say|mention|specify)",
            ans.lower(),
        )
    )


def fires(trigger: str, flag: str | None, answer: str) -> bool:
    parts = trigger.split("|")
    level = "insufficient" if flag is None else flag
    return ("abstain" in parts and abstains(answer)) or level in parts


def candidates():
    out = []
    for s in ("leaf_only@400", "leaf_only@1000"):
        for t in LEAF:
            if int(t.split("@")[1]) > int(s.split("@")[1]):
                out += [{"stage1": s, "trigger": tr, "target": t} for tr in TRIGGERS]
    out.append(
        {
            "stage1": "leaf_only@400",
            "trigger": "two_level",
            "target": "partial->1000, insufficient->2000",
        }
    )
    return out


class Cascade:
    """Per-question outcome of one candidate on one split."""

    def __init__(self, s1: Split, lab: Split, cand: dict, qs: list[str], cache_fraction: float):
        self.f1, self.tin, self.tout, self.hit, self.esc = {}, {}, {}, {}, {}
        st = cand["stage1"]
        for q in qs:
            flag, ans = s1.sufficiency[q].get(st), s1.answer[q][st]
            in1, out1 = s1.tin[q][st], s1.tout[q][st]
            # Recorded stage-1 cache hits are NOT used: our earlier label runs sent prompts
            # with the same instructions + context prefix, so they overstate what a fresh
            # deployment would hit. Only the retry's shared prefix is modeled (probe-measured).
            hit1 = 0
            if cand["trigger"] == "two_level":
                lvl = "insufficient" if flag is None else flag
                tgt = {"insufficient": "leaf_only@2000", "partial": "leaf_only@1000"}.get(lvl)
            else:
                tgt = cand["target"] if fires(cand["trigger"], flag, ans) else None
            if tgt is None:
                self.f1[q], self.tin[q], self.tout[q], self.hit[q], self.esc[q] = (
                    s1.f1[q][st],
                    in1,
                    out1,
                    hit1,
                    None,
                )
            else:
                prefix = s1.ctx_tokens[q][st] + INSTRUCTION_TOKENS  # shared with the retry
                self.f1[q] = lab.f1[q][tgt]
                self.tin[q] = in1 + lab.tin[q][tgt]
                self.tout[q] = out1 + lab.tout[q][tgt]
                self.hit[q] = hit1 + cache_fraction * prefix
                self.esc[q] = tgt

    def row(self, name: str) -> dict:
        qs = list(self.f1)
        tin, tout, hit = (np.mean([d[q] for q in qs]) for d in (self.tin, self.tout, self.hit))
        return {
            "system": name,
            "f1": round(float(np.mean(list(self.f1.values()))), 4),
            "input_tokens": round(float(tin), 1),
            "output_tokens": round(float(tout), 1),
            "total_tokens": round(float(tin + tout), 1),
            "usd_per_1k_q_no_cache": round((tin * PRICE_MISS + tout * PRICE_OUT) / 1e6 * 1000, 4),
            "usd_per_1k_q_with_cache": round(
                ((tin - hit) * PRICE_MISS + hit * PRICE_HIT + tout * PRICE_OUT) / 1e6 * 1000, 4
            ),
            "escalation_rate": round(float(np.mean([self.esc[q] is not None for q in qs])), 3),
        }


def fixed_rows(lab: Split, qs, c):
    class F:  # same interface as Cascade
        pass

    f = F()
    f.f1 = {q: lab.f1[q][c] for q in qs}
    f.tin = {q: lab.tin[q][c] for q in qs}
    f.tout = {q: lab.tout[q][c] for q in qs}
    f.hit = {q: 0 for q in qs}
    f.esc = {q: None for q in qs}
    f.row = lambda name: Cascade.row(f, name)
    return f


def leaf_mix(lab: Split, qs, target: float, unit: str):
    """Hash-randomised mix of the two neighbouring leaf budgets at a matched mean cost."""

    def cost(q, c):
        return (
            lab.tin[q][c] + lab.tout[q][c]
            if unit == "tokens"
            else (lab.tin[q][c] * PRICE_MISS + lab.tout[q][c] * PRICE_OUT) / 1e6 * 1000
        )

    costs = [np.mean([cost(q, c) for q in qs]) for c in LEAF]
    lo = max([i for i, c in enumerate(costs) if c <= target], default=0)
    hi = min(lo + 1, len(LEAF) - 1)
    w = 0.0 if hi == lo else min(1.0, max(0.0, (target - costs[lo]) / (costs[hi] - costs[lo])))
    rank = lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16) % 10_000 / 10_000  # noqa: E731
    pick = {q: LEAF[hi] if rank(q) < w else LEAF[lo] for q in qs}
    f = fixed_rows(lab, qs, LEAF[0])
    f.f1 = {q: lab.f1[q][pick[q]] for q in qs}
    f.tin = {q: lab.tin[q][pick[q]] for q in qs}
    f.tout = {q: lab.tout[q][pick[q]] for q in qs}
    return f


def auc(score: list[float], y: list[bool]) -> float | None:
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, score)), 3) if 0 < sum(y) < len(y) else None


def signal_quality(s1: Split, lab: Split, qs, st: str, tgt: str) -> dict:
    gain = {q: lab.f1[q][tgt] - s1.f1[q][st] for q in qs}
    y = [gain[q] > 0 for q in qs]
    flag = [SCORE[s1.sufficiency[q].get(st)] for q in qs]
    abst = [float(abstains(s1.answer[q][st])) for q in qs]
    out = {
        "stage1": st,
        "target": tgt,
        "share_gain_positive": round(float(np.mean(y)), 3),
        "auc_flag": auc(flag, y),
        "auc_abstain": auc(abst, y),
        "auc_flag_plus_abstain": auc([f + 0.5 * a for f, a in zip(flag, abst, strict=True)], y),
        "flag_missing_rate": round(
            float(np.mean([s1.sufficiency[q].get(st) is None for q in qs])), 3
        ),
        "by_level": {},
    }
    for lvl in ("sufficient", "partial", "insufficient", None):
        g = [gain[q] for q in qs if s1.sufficiency[q].get(st) == lvl]
        if g:
            out["by_level"][str(lvl)] = {
                "share": round(len(g) / len(qs), 3),
                "mean_gain": round(float(np.mean(g)), 4),
                "share_gain_positive": round(float(np.mean([x > 0 for x in g])), 3),
            }
    return out


def simulate_stage1(lab: Split) -> Split:
    """Dry run: stage 1 := stored run-A answers, flag := insufficient iff abstaining."""
    for q in lab.qs:
        for c in ("leaf_only@400", "leaf_only@1000"):
            lab.sufficiency[q][c] = "insufficient" if abstains(lab.answer[q][c]) else "sufficient"
    return lab


def merge_new_prompt(s1: Split, run2000: Path, qdir: Path) -> Split:
    """New-prompt leaf ladder 400/1000/2000: stage-1 run (400/1000) + fixed-2000 run."""
    s2 = Split(run2000, None, qdir / "dev" / "questions.jsonl", configs=["leaf_only@2000"])
    for attr in (
        "f1",
        "tin",
        "tout",
        "answer",
        "sufficiency",
        "ctx_tokens",
        "prefix_hit",
        "gen_cache_hit",
    ):
        mine, other = getattr(s1, attr), getattr(s2, attr)
        for q, row in other.items():
            if "leaf_only@2000" in row:
                mine[q]["leaf_only@2000"] = row["leaf_only@2000"]
    return s1


def prompt_controlled(s1, run2000, old, qs, primary, cache_fraction, qdir) -> dict:
    """Separate the prompt effect from the adaptive (escalation) effect.

    new = qasper_sufficiency_v1 answers (flag stripped) at leaf 400/1000/2000; old = the
    qasper_default answers. The prompt-matched cascade uses the new prompt in BOTH stages.
    """
    new = merge_new_prompt(s1, run2000, qdir)
    qs = [q for q in qs if "leaf_only@2000" in new.f1[q]]
    out: dict = {"n": len(qs)}
    out["prompt_effect_new_minus_old"] = {
        c: boot({q: new.f1[q][c] - old.f1[q][c] for q in qs}, old.doc) for c in LEAF
    }
    if primary.get("trigger") == "two_level":
        raise SystemExit("prompt control implemented for single-target primaries only")
    cas = Cascade(new, new, primary, qs, cache_fraction)
    systems = {"prompt-matched cascade (primary)": cas}
    row = cas.row("prompt-matched cascade (primary)")
    for c in LEAF:
        systems[f"fixed {c} (new prompt)"] = fixed_rows(new, qs, c)
    for c in ("leaf_only@1000", "leaf_only@2000"):
        systems[f"fixed {c} (original prompt)"] = fixed_rows(old, qs, c)
    mixes = {
        "new-prompt leaf mix @ matched USD": leaf_mix(new, qs, row["usd_per_1k_q_no_cache"], "usd"),
        "original-prompt leaf mix @ matched USD": leaf_mix(
            old, qs, row["usd_per_1k_q_no_cache"], "usd"
        ),
        "new-prompt leaf mix @ matched tokens": leaf_mix(new, qs, row["total_tokens"], "tokens"),
    }
    if cache_fraction > 0:
        mixes["new-prompt leaf mix @ matched USD (cascade with cache)"] = leaf_mix(
            new, qs, row["usd_per_1k_q_with_cache"], "usd"
        )
    systems.update(mixes)
    usd_mixes = [k for k in mixes if "matched USD" in k and "with cache" not in k]
    strongest = max(usd_mixes, key=lambda k: sum(mixes[k].f1.values()))
    rows = []
    for name, sysm in systems.items():
        r = sysm.row(name)
        if name != "prompt-matched cascade (primary)":
            r["cascade_minus_this"] = boot({q: cas.f1[q] - sysm.f1[q] for q in qs}, old.doc)
        rows.append(r)
    out["systems"] = rows
    out["strongest_fixed_mixture_at_matched_usd"] = strongest
    out["adaptive_effect_vs_strongest"] = boot(
        {q: cas.f1[q] - systems[strongest].f1[q] for q in qs}, old.doc
    )
    out["note"] = (
        "prompt effect: same context, different instructions; adaptive effect: "
        "prompt-matched cascade vs the stronger matched-USD fixed mixture"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stage1-train", type=Path)
    p.add_argument("--stage1-dev", type=Path)
    p.add_argument("--labels-train", type=Path)
    p.add_argument("--labels-dev", type=Path)
    p.add_argument(
        "--cache-fraction",
        type=float,
        default=0.0,
        help="share of the shared prefix billed as a cache hit on the retry "
        "(measure with scripts/probe_prompt_cache.py; 0 = no caching)",
    )
    p.add_argument("--simulate", action="store_true", help="dry run without stage-1 runs")
    p.add_argument(
        "--fixed2000-new-dev",
        type=Path,
        help="dev run of fixed leaf_only@2000 with the sufficiency prompt (prompt control)",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/diagnostics/cascade"))
    a = p.parse_args()
    qdir = Path("data/processed/qasper/v0.3")
    labs = {
        sp: Split(
            getattr(a, f"labels_{sp}") or latest(f"outputs/labels_{sp}/*/"),
            None,
            qdir / sp / "questions.jsonl",
            configs=LEAF,
        )
        for sp in ("train", "dev")
    }
    s1s = {}
    for sp in ("train", "dev"):
        if a.simulate:
            s1s[sp] = simulate_stage1(labs[sp])
        else:
            run = getattr(a, f"stage1_{sp}")
            if run is None:
                raise SystemExit(f"--stage1-{sp} is required (or --simulate)")
            s1s[sp] = Split(
                run,
                None,
                qdir / sp / "questions.jsonl",
                configs=["leaf_only@400", "leaf_only@1000"],
            )
    qs = {sp: sorted(set(labs[sp].qs) & set(s1s[sp].qs)) for sp in ("train", "dev")}
    rep: dict = {
        "simulate": a.simulate,
        "cache_fraction": a.cache_fraction,
        "n": {sp: len(v) for sp, v in qs.items()},
        "prices_usd_per_mtok": {"miss": PRICE_MISS, "hit": PRICE_HIT, "out": PRICE_OUT},
    }

    # stage-1 format effect: does adding the flag change answer quality?
    rep["stage1_format_effect_dev"] = {
        st: boot(
            {q: s1s["dev"].f1[q][st] - labs["dev"].f1[q][st] for q in qs["dev"]}, labs["dev"].doc
        )
        for st in ("leaf_only@400", "leaf_only@1000")
    }
    rep["stage1_recorded_cache_hit_share"] = {
        sp: round(
            float(
                np.sum([s1s[sp].prefix_hit[q].get(st, 0) for q in qs[sp] for st in s1s[sp].tin[q]])
                / max(1, np.sum([s1s[sp].tin[q][st] for q in qs[sp] for st in s1s[sp].tin[q]]))
            ),
            3,
        )
        for sp in ("train", "dev")
    }  # informational: inflated by earlier runs with identical prefixes
    rep["signal_quality"] = {
        sp: [
            signal_quality(s1s[sp], labs[sp], qs[sp], st, t)
            for st, t in (
                ("leaf_only@400", "leaf_only@1000"),
                ("leaf_only@400", "leaf_only@2000"),
                ("leaf_only@1000", "leaf_only@2000"),
            )
        ]
        for sp in ("train", "dev")
    }

    # calibrate on train: pick the primary candidate
    train_rows = []
    for cand in candidates():
        cas = Cascade(s1s["train"], labs["train"], cand, qs["train"], a.cache_fraction)
        r = cas.row(json.dumps(cand))
        mix = leaf_mix(labs["train"], qs["train"], r["total_tokens"], "tokens")
        r["train_delta_vs_mix_tokens"] = round(
            float(np.mean([cas.f1[q] - mix.f1[q] for q in qs["train"]])), 4
        )
        train_rows.append((cand, r))
    primary = max(
        train_rows, key=lambda x: (x[1]["train_delta_vs_mix_tokens"], -x[1]["total_tokens"])
    )[0]
    rep["train_candidates"] = [r for _, r in train_rows]
    rep["primary"] = primary

    # evaluate on dev
    lab, s1, dq = labs["dev"], s1s["dev"], qs["dev"]
    systems = {"primary cascade": Cascade(s1, lab, primary, dq, a.cache_fraction)}
    prow = systems["primary cascade"].row("primary cascade")
    systems["fixed leaf_only@400"] = fixed_rows(lab, dq, "leaf_only@400")
    systems["fixed leaf_only@1000"] = fixed_rows(lab, dq, "leaf_only@1000")
    systems["fixed leaf_only@2000"] = fixed_rows(lab, dq, "leaf_only@2000")
    systems["leaf mix @ matched tokens"] = leaf_mix(lab, dq, prow["total_tokens"], "tokens")
    systems["leaf mix @ matched USD"] = leaf_mix(lab, dq, prow["usd_per_1k_q_no_cache"], "usd")
    if a.cache_fraction > 0:  # fixed configs have no cacheable retry prefix
        systems["leaf mix @ matched USD (cascade with cache)"] = leaf_mix(
            lab, dq, prow["usd_per_1k_q_with_cache"], "usd"
        )
    old = simulate_stage1(
        Split(
            a.labels_dev or latest("outputs/labels_dev/*/"),
            None,
            qdir / "dev" / "questions.jsonl",
            configs=LEAF,
        )
    )
    systems["abstention cascade (run A, 400->2000)"] = Cascade(
        old,
        lab,
        {"stage1": "leaf_only@400", "trigger": "insufficient", "target": "leaf_only@2000"},
        dq,
        a.cache_fraction,
    )
    base = systems["primary cascade"]
    rows = []
    for name, s in systems.items():
        r = s.row(name)
        if name != "primary cascade":
            r["primary_minus_this"] = boot({q: base.f1[q] - s.f1[q] for q in dq}, lab.doc)
        rows.append(r)
    rep["dev_primary"] = rows
    sec = []
    for cand, _ in train_rows:
        cas = Cascade(s1, lab, cand, dq, a.cache_fraction)
        r = cas.row(json.dumps(cand))
        mix = leaf_mix(lab, dq, r["total_tokens"], "tokens")
        r["vs_mix_tokens"] = boot({q: cas.f1[q] - mix.f1[q] for q in dq}, lab.doc)
        sec.append(r)
    rep["dev_secondary_all_candidates"] = sec
    rep["note"] = (
        "primary chosen on train; dev CIs for the primary are not adjusted for "
        "selection (there was none on dev); secondary rows are exploratory"
    )
    if a.fixed2000_new_dev:
        rep["prompt_controlled"] = prompt_controlled(
            s1s["dev"], a.fixed2000_new_dev, lab, dq, primary, a.cache_fraction, qdir
        )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "cascade_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("primary (chosen on train):", primary)
    for r in rows:
        print(
            f"{r['system']:42} F1={r['f1']:.4f} tok={r['total_tokens']:7.1f} "
            f"$/1k={r['usd_per_1k_q_no_cache']:.4f} esc={r['escalation_rate']} "
            f"primary-this={r.get('primary_minus_this', '')}"
        )
    print(json.dumps(rep["signal_quality"]["dev"][1], indent=None))
    if "prompt_controlled" in rep:
        pc = rep["prompt_controlled"]
        print("prompt effect (new - old):", pc["prompt_effect_new_minus_old"])
        for r in pc["systems"]:
            print(
                f"  {r['system']:56} F1={r['f1']:.4f} tok={r['total_tokens']:7.1f} "
                f"$/1k={r['usd_per_1k_q_no_cache']:.4f} cas-this={r.get('cascade_minus_this', '')}"
            )
        print(
            "adaptive effect vs",
            pc["strongest_fixed_mixture_at_matched_usd"],
            pc["adaptive_effect_vs_strongest"],
        )
    print(f"wrote {a.out / 'cascade_report.json'}")


if __name__ == "__main__":
    main()

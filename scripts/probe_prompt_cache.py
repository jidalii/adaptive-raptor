"""Measure DeepSeek prefix caching for the cascade retry. PAID ONLY WITH --execute.

    uv run python scripts/probe_prompt_cache.py              # plan only: prompts, tokens, cost
    uv run python scripts/probe_prompt_cache.py --execute    # ~2*N real calls (< $0.01)

For N dev questions it sends the stage-1 prompt (qasper_sufficiency_v1, leaf_only@400 context)
and, after --wait seconds, the stage-2 prompt (qasper_default, leaf_only@2000 context, which
starts with the same chunks). It reports usage.prompt_cache_hit_tokens of both calls and the
retry's hit fraction of the shared prefix. Use the median fraction as
analyze_cascade.py --cache-fraction. DeepSeek documents caching as best-effort, so a low or
zero fraction is a valid result. Contexts are rebuilt from the cached trees and the stored
node ids (no retrieval, no summarisation). Run off-peak (prices double 01-04 and 06-10 UTC
on weekdays).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from offline_diagnostics import LEAF, Split, latest, load_tree  # noqa: E402

from adaptive_raptor.generation import get_template, render  # noqa: E402
from adaptive_raptor.models.chat import thinking_extra_body, visible_text  # noqa: E402
from adaptive_raptor.models.credentials import resolve  # noqa: E402


def peak_now() -> bool:
    now = datetime.now(UTC)
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--wait", type=float, default=15.0, help="seconds between stage 1 and 2")
    p.add_argument("--model", default="deepseek-flash")
    p.add_argument("--execute", action="store_true", help="actually call the API (paid)")
    p.add_argument(
        "--no-nonce",
        action="store_true",
        help="do not prefix a fresh nonce (then earlier identical requests can be cache hits)",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/diagnostics/cache_probe.json"))
    a = p.parse_args()
    lab = Split(
        latest("outputs/labels_dev/*/"),
        None,
        Path("data/processed/qasper/v0.3/dev/questions.jsonl"),
        configs=LEAF,
    )
    enc = tiktoken.get_encoding("cl100k_base")
    qs = lab.qs[:: max(1, len(lab.qs) // a.n)][: a.n]
    s1t, s2t = get_template("qasper_sufficiency_v1"), get_template("qasper_default")
    plan = []
    for q in qs:
        tree = load_tree(lab.doc[q], lab.index_hash[lab.doc[q]])
        ctx = {
            c: [tree.all_nodes[i].text for i in lab.nodes[q][c]]
            for c in ("leaf_only@400", "leaf_only@2000")
        }
        # A fresh nonce shared by both calls: neither prompt was ever sent before, so the
        # retry can only hit the prefix that stage 1 created (earlier runs sent the plain
        # prompts, and DeepSeek keeps cache units for hours to days).
        nonce = "" if a.no_nonce else f"Request {uuid.uuid4().hex}\n\n"
        p1 = nonce + render(s1t, lab.question[q], ctx["leaf_only@400"])
        p2 = nonce + render(s2t, lab.question[q], ctx["leaf_only@2000"])
        shared = next((i for i, (x, y) in enumerate(zip(p1, p2, strict=False)) if x != y), 0)
        plan.append(
            {
                "question_id": q,
                "stage1_tokens": len(enc.encode(p1)),
                "stage2_tokens": len(enc.encode(p2)),
                "shared_prefix_tokens": len(enc.encode(p1[:shared])),
                "p1": p1,
                "p2": p2,
            }
        )
    total_in = sum(x["stage1_tokens"] + x["stage2_tokens"] for x in plan)
    print(f"{len(plan)} questions, {2 * len(plan)} calls, ~{total_in} input tokens")
    print(f"estimated cost <= ${total_in * 0.30 / 1e6 + 2 * len(plan) * 30 * 1.2 / 1e6:.4f} (peak)")
    for x in plan:
        print(
            f"  {x['question_id'][:12]} stage1 {x['stage1_tokens']} stage2 "
            f"{x['stage2_tokens']} shared prefix {x['shared_prefix_tokens']}"
        )
    if not a.execute:
        print("plan only; pass --execute to call the API")
        return
    if peak_now():
        print("warning: peak hours now (prices x2)")
    load_dotenv()
    from openai import OpenAI

    key, base_url = resolve("deepseek", None, None)
    client = OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=2)

    def call(prompt: str) -> dict:
        r = client.chat.completions.create(
            model=a.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            extra_body=thinking_extra_body("deepseek", False),
        )
        u = r.usage
        return {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "hit": getattr(u, "prompt_cache_hit_tokens", None),
            "miss": getattr(u, "prompt_cache_miss_tokens", None),
            "text": visible_text(r),
        }

    rows = []
    for x in plan:
        r1 = call(x["p1"])
        time.sleep(a.wait)
        r2 = call(x["p2"])
        frac = (r2["hit"] or 0) / max(1, x["shared_prefix_tokens"])
        rows.append(
            {
                **{k: v for k, v in x.items() if k not in ("p1", "p2")},
                "stage1": r1,
                "stage2": r2,
                "retry_hit_fraction_of_shared_prefix": round(frac, 3),
            }
        )
        print(
            f"  {x['question_id'][:12]} stage1 hit {r1['hit']} stage2 hit {r2['hit']} -> {frac:.2f}"
        )
    fr = sorted(r["retry_hit_fraction_of_shared_prefix"] for r in rows)
    out = {
        "measured_at": datetime.now(UTC).isoformat(),
        "model": a.model,
        "median_retry_hit_fraction": fr[len(fr) // 2],
        "rows": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"median retry hit fraction {out['median_retry_hit_fraction']}; wrote {a.out}")


if __name__ == "__main__":
    main()

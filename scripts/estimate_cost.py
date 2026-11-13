"""Estimate the API cost of an experiment config before running it. Offline, no API calls.

    uv run python scripts/estimate_cost.py --config configs/experiments/summary_cap512_dev.yaml
    uv run python scripts/estimate_cost.py --config <yaml> --set dataset.doc_ids_file=<ids>

Token means come from the latest finished label run of the same split (labels_<split>):
answers per config (input/output) and summariser tokens per paper. Index builds are counted
only for papers without a cached index at this config's summary cap. Prices: deepseek-flash
off-peak ($0.15 / 1M input on a cache miss, $0.60 / 1M output); peak hours cost 2x. Provider
prefix-cache hits are ignored, so input cost is an upper bound.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

from adaptive_raptor.config.loader import load_config
from adaptive_raptor.generation import get_template

PRICE_IN, PRICE_OUT = 0.15, 0.60


def latest(pattern: str) -> Path:
    runs = sorted(Path().glob(pattern))
    if not runs:
        raise SystemExit(f"no reference run matches {pattern}")
    return runs[-1]


def cached_indexes(chunk_size: int, cap: int) -> set[str]:
    have = set()
    for meta in Path("indexes/qasper/v0.3").glob(f"f{chunk_size}/seed*/*.json"):
        k = json.loads(meta.read_text())["key"]
        if k["raptor"]["summarization_length"] == cap and k["chunking"]["chunk_size"] == chunk_size:
            have.add(k["doc_id"])
    return have


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    a = p.parse_args()
    cfg = load_config(a.config, a.overrides)
    split = cfg.dataset.split
    docs = set(Path(cfg.dataset.doc_ids_file).read_text().split())
    nq = defaultdict(int)
    with Path(f"data/processed/qasper/v0.3/{split}/questions.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["doc_id"] in docs:
                nq[r["doc_id"]] += 1
    n_q = sum(nq.values())
    cap = cfg.raptor.summarization_length
    chunk = cfg.chunking.variants[0].chunk_size
    docs = {d for d in docs if nq[d]}  # papers without questions are never indexed
    missing = docs - cached_indexes(chunk, cap)

    ref = latest(f"outputs/labels_{split}/*/")
    tok = defaultdict(lambda: [[], []])
    for line in (ref / "labels.jsonl").open():
        r = json.loads(line)
        tok[r["config_id"]][0].append(r.get("input_tokens") or 0)
        tok[r["config_id"]][1].append(r.get("output_tokens") or 0)
    status = json.loads((ref / "status.json").read_text())["metrics"]
    su = status["summarizer_usage"]
    per_paper_in = su["input_tokens"] / status["indexes_built"]
    per_paper_calls = su["calls"] / status["indexes_built"]
    ref_cap = 200
    enc = tiktoken.get_encoding("cl100k_base")
    extra_in = len(enc.encode(get_template(cfg.generation.prompt_template))) - len(
        enc.encode(get_template("qasper_default"))
    )
    extra_out = 8 if "Evidence: SUFFICIENT" in get_template(cfg.generation.prompt_template) else 0

    configs = [c.id for c in cfg.retrieval.catalog()]
    if cfg.retrieval.include_raptor_original:
        configs.append("raptor_original")
    rows, total_lo, total_hi = [], 0.0, 0.0
    for c in configs:
        i, o = np.mean(tok[c][0]) + extra_in, np.mean(tok[c][1]) + extra_out
        usd = n_q * (i * PRICE_IN + o * PRICE_OUT) / 1e6
        rows.append((c, round(i), round(o, 1), round(usd, 3)))
    ans = sum(r[3] for r in rows)
    # leaf answers can come from the response cache only with the reference prompt
    same_prompt = cfg.generation.prompt_template == "qasper_default"
    leaf = sum(r[3] for r in rows if r[0].startswith("leaf_only")) if same_prompt else 0.0
    # index builds: summary count ~unchanged; output per summary between the reference fill
    # and the full cap; higher layers read longer children (input x1.0 .. x1.5)
    out_lo = per_paper_calls * min(cap, su["output_tokens"] / su["calls"] * cap / ref_cap * 0.6)
    out_hi = per_paper_calls * cap
    idx_lo = len(missing) * (per_paper_in * PRICE_IN + out_lo * PRICE_OUT) / 1e6
    idx_hi = len(missing) * (per_paper_in * 1.5 * PRICE_IN + out_hi * PRICE_OUT) / 1e6
    print(f"config {a.config} (split {split}): {len(docs)} papers, {n_q} questions")
    print(f"reference token means: {ref}")
    print(f"index builds needed: {len(missing)} papers at cap {cap}: ${idx_lo:.2f} - ${idx_hi:.2f}")
    print("answers (one call per question per config, no response-cache hits assumed):")
    for c, i, o, usd in rows:
        print(f"  {c:18} in {i:5} out {o:5}  ${usd:.3f}")
    print(
        f"answers total ${ans:.2f}  (excluding leaf_only configs, if served from cache: "
        f"${ans - leaf:.2f})"
    )
    total_lo, total_hi = idx_lo + ans - leaf, idx_hi + ans
    print(f"TOTAL off-peak: ${total_lo:.2f} - ${total_hi:.2f}   (peak hours: x2)")


if __name__ == "__main__":
    main()

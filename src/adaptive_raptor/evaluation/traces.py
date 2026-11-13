"""Execution traces (one per question x configuration) and generic result export."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class ExecutionTrace:
    question_id: str
    doc_id: str
    index_variant: str
    seed: int
    config_id: str
    run_label: str
    question: str
    index_key_hash: str = ""
    index_cache_hit: bool = False
    node_ids: list[int] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)
    contexts: list[str] | None = None
    context_tokens: int = 0
    retrieval_ms: float = 0.0
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    generation_ms: float = 0.0
    generation_cache_hit: bool = False
    model_version: str = ""
    answer_f1: float | None = None
    answer_type: str | None = None
    answer_rouge_l: float | None = None  # max ROUGE-L F1 over references (all datasets)
    error: str | None = None
    sufficiency: str | None = None  # evidence flag from a sufficiency prompt (else None)
    prompt_cache_hit_tokens: int = 0  # input tokens served from the provider prefix cache
    # router (RouterPolicy only; empty for catalog runs)
    router_model: str = ""
    route_fallback_reason: str | None = None
    profile: dict[str, Any] | None = None
    profile_confidence: float | None = None
    router_input_tokens: int = 0
    router_output_tokens: int = 0
    router_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceWriter:
    """Append-only JSONL writer (crash-safe: each line is flushed)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a")

    def write(self, trace: ExecutionTrace) -> None:
        self._f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_traces(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(
    traces: list[dict[str, Any]], price_in: float | None = None, price_out: float | None = None
) -> list[dict[str, Any]]:
    """Per-configuration means. Generic aggregation only (no CIs; see evaluation.bootstrap)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in traces:
        groups[(t["index_variant"], t["config_id"])].append(t)
    rows = []
    for (variant, config_id), ts in groups.items():  # catalog order
        ok = [t for t in ts if t["error"] is None]
        row: dict[str, Any] = {
            "index_variant": variant,
            "config_id": config_id,
            "n": len(ts),
            "errors": len(ts) - len(ok),
            "answer_f1": round(mean(t["answer_f1"] for t in ok), 4) if ok else None,
            "answer_rouge_l": (
                round(mean(t["answer_rouge_l"] for t in ok), 4)
                if ok and all(t.get("answer_rouge_l") is not None for t in ok)
                else None
            ),
            "context_tokens": round(mean(t["context_tokens"] for t in ok), 1) if ok else None,
            "retrieval_ms": round(mean(t["retrieval_ms"] for t in ok), 2) if ok else None,
            "generation_ms": round(mean(t["generation_ms"] for t in ok), 1) if ok else None,
            "input_tokens": sum(t["input_tokens"] for t in ok),
            "output_tokens": sum(t["output_tokens"] for t in ok),
        }
        for kind in ("extractive", "abstractive", "boolean", "none"):
            vals = [t["answer_f1"] for t in ok if t["answer_type"] == kind]
            row[f"f1_{kind}"] = round(mean(vals), 4) if vals else None
        if price_in is not None and price_out is not None:

            def usd(subset: list[dict]) -> float:
                tin = sum(t["input_tokens"] for t in subset)
                tout = sum(t["output_tokens"] for t in subset)
                return round(tin / 1e6 * price_in + tout / 1e6 * price_out, 4)

            # cost_usd: workload cost of this config (as if uncached; the research metric).
            # billed_usd: new spend in this run (cache hits are free).
            row["cost_usd"] = usd(ok)
            row["billed_usd"] = usd([t for t in ok if not t.get("generation_cache_hit")])
        rows.append(row)
    return rows


def write_rows(
    rows: list[dict[str, Any]], run_dir: Path, name: str, formats: list[str]
) -> list[Path]:
    out = []
    if "json" in formats:
        p = run_dir / f"{name}.json"
        p.write_text(json.dumps(rows, indent=2) + "\n")
        out.append(p)
    if "csv" in formats and rows:
        p = run_dir / f"{name}.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        out.append(p)
    if "markdown" in formats and rows:
        p = run_dir / f"{name}.md"
        cols = list(rows[0])
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += [
            "| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows
        ]
        p.write_text("\n".join(lines) + "\n")
        out.append(p)
    return out


def write_official_predictions(traces: list[dict[str, Any]], run_dir: Path) -> list[Path]:
    """Per-config files in the official evaluator's input format (cross-check tool)."""
    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for t in traces:
        if t["error"] is None:
            by_cfg[(t["index_variant"], t["config_id"])].append(t)
    out = []
    for (variant, cfg), ts in by_cfg.items():
        p = run_dir / "predictions" / f"{variant}__{cfg.replace('@', '_')}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for t in ts:
                f.write(
                    json.dumps(
                        {
                            "question_id": t["question_id"],
                            "predicted_answer": t["answer"],
                            "predicted_evidence": [],
                        }
                    )
                    + "\n"
                )
        out.append(p)
    return out

"""Trace-derived router training records and acceptability-label extension point."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LabelRecord:
    question_id: str
    doc_id: str
    index_variant: str
    seed: int
    config_id: str
    run: str
    tokens_used: int
    answer_f1: float
    answer_type: str | None = None
    latency_ms: float | None = None
    model_version: str = ""
    input_tokens: int = 0  # generation prompt tokens (for $ cost); 0 in pre-WP10 tables
    output_tokens: int = 0


def from_traces(traces: Iterable[dict]) -> list[LabelRecord]:
    return [
        LabelRecord(
            question_id=trace["question_id"],
            doc_id=trace["doc_id"],
            index_variant=trace["index_variant"],
            seed=trace["seed"],
            config_id=trace["config_id"],
            run=trace["run_label"],
            tokens_used=trace["context_tokens"],
            answer_f1=trace["answer_f1"],
            answer_type=trace["answer_type"],
            latency_ms=trace["retrieval_ms"] + trace["generation_ms"],
            model_version=trace["model_version"],
            input_tokens=trace.get("input_tokens", 0),
            output_tokens=trace.get("output_tokens", 0),
        )
        for trace in traces
        if trace["error"] is None
    ]


class LabelStore:
    """Append-only JSONL table of label records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, records: Iterable[LabelRecord]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.path.open("a") as output:
            for record in records:
                output.write(json.dumps(asdict(record)) + "\n")
                count += 1
        return count

    def load(self) -> list[LabelRecord]:
        if not self.path.is_file():
            return []
        with self.path.open() as source:
            return [LabelRecord(**json.loads(line)) for line in source if line.strip()]


RecordKey = tuple[str, str, int, str, str]  # question, variant, seed, config, run


def record_key(r: LabelRecord) -> RecordKey:
    return (r.question_id, r.index_variant, r.seed, r.config_id, r.run)


@dataclass
class MergeReport:
    n_input: int = 0
    n_unique: int = 0
    n_duplicates: int = 0
    conflicts: list[RecordKey] = field(default_factory=list)  # same key, different F1


def merge_label_tables(paths: Sequence[Path]) -> tuple[list[LabelRecord], MergeReport]:
    """Merge ``labels.jsonl`` files from several runs, keeping one row per record key.

    Paths are read in the order given (pass run directories sorted by time). The first
    row for a key wins. A later row with the same key but a different F1 means the
    generator was not reproducible for that call; it is counted in ``conflicts`` and
    never silently averaged.
    """
    report = MergeReport()
    merged: dict[RecordKey, LabelRecord] = {}
    for path in paths:
        for record in LabelStore(path).load():
            report.n_input += 1
            key = record_key(record)
            kept = merged.get(key)
            if kept is None:
                merged[key] = record
                continue
            report.n_duplicates += 1
            if kept.answer_f1 != record.answer_f1:
                report.conflicts.append(key)
    report.n_unique = len(merged)
    return list(merged.values()), report


def select_records(
    records: Iterable[LabelRecord],
    *,
    index_variant: str | None = None,
    seed: int | None = None,
    run: str | None = None,
    configs: Collection[str] | None = None,
) -> list[LabelRecord]:
    return [
        r
        for r in records
        if (index_variant is None or r.index_variant == index_variant)
        and (seed is None or r.seed == seed)
        and (run is None or r.run == run)
        and (configs is None or r.config_id in configs)
    ]


def check_single_cell(records: Sequence[LabelRecord]) -> None:
    cells = {(r.index_variant, r.seed, r.run) for r in records}
    if len(cells) > 1:
        raise ValueError(
            f"records mix {len(cells)} (index_variant, seed, run) cells {sorted(cells)[:4]}; "
            "filter with select_records() first"
        )
    keys = [record_key(r) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate record keys; merge with merge_label_tables() first")


@dataclass
class LabelReport:
    epsilon: float
    configs: list[str]
    n_queries: int = 0
    n_labeled: int = 0
    n_all_fail: int = 0  # Q*(q) = 0: excluded from training, kept for evaluation
    incomplete: list[str] = field(default_factory=list)  # missing >= 1 config: excluded
    by_answer_type: dict[str, dict[str, float]] = field(default_factory=dict)
    acceptable_rate: dict[str, float] = field(default_factory=dict)  # per config


def acceptability_labels(
    records: Sequence[LabelRecord],
    epsilon: float,
    configs: Sequence[str] | None = None,
    *,
    report: LabelReport | None = None,
) -> dict[str, dict[str, bool] | None]:
    """Derive epsilon-acceptable labels; exclude all-fail queries from training.

    ``y(q, c) = [F1(q, c) >= max_c' F1(q, c') - epsilon]`` (proposal 5.2). Records must
    come from one (index_variant, seed, run) cell. ``configs`` defaults to every config
    present except ``raptor_original`` (not a catalog config).

    Returns ``question_id -> {config_id: acceptable}``, or ``None`` for an all-fail query
    (best F1 is 0). Queries missing any config are left out and listed in
    ``report.incomplete``; they are never filled in.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    check_single_cell(records)
    if configs is None:
        configs = sorted({r.config_id for r in records} - {"raptor_original"})
    configs = list(configs)
    rep = report if report is not None else LabelReport(epsilon, configs)
    rep.epsilon, rep.configs = epsilon, configs

    by_query: dict[str, dict[str, LabelRecord]] = defaultdict(dict)
    for r in records:
        if r.config_id in configs:
            by_query[r.question_id][r.config_id] = r

    labels: dict[str, dict[str, bool] | None] = {}
    type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, all-fail, ok
    accept_counts = dict.fromkeys(configs, 0)
    for qid in sorted(by_query):
        row = by_query[qid]
        rep.n_queries += 1
        if len(row) < len(configs):
            rep.incomplete.append(qid)
            continue
        answer_type = next(iter(row.values())).answer_type or "unknown"
        type_counts[answer_type][0] += 1
        best = max(r.answer_f1 for r in row.values())
        if best <= 0:
            labels[qid] = None
            rep.n_all_fail += 1
            type_counts[answer_type][1] += 1
            continue
        y = {c: row[c].answer_f1 >= best - epsilon - 1e-12 for c in configs}
        labels[qid] = y
        rep.n_labeled += 1
        type_counts[answer_type][2] += sum(y.values())
        for c, ok in y.items():
            accept_counts[c] += ok
    rep.acceptable_rate = {
        c: round(n / rep.n_labeled, 4) if rep.n_labeled else 0.0 for c, n in accept_counts.items()
    }
    rep.by_answer_type = {
        t: {
            "n": n,
            "all_fail": fail,
            "mean_acceptable_configs": round(ok / (n - fail), 3) if n > fail else 0.0,
        }
        for t, (n, fail, ok) in sorted(type_counts.items())
    }
    return labels


__all__ = [
    "LabelRecord",
    "LabelReport",
    "LabelStore",
    "MergeReport",
    "acceptability_labels",
    "check_single_cell",
    "from_traces",
    "merge_label_tables",
    "record_key",
    "select_records",
]

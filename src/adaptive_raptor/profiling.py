"""Trace-derived router training records and acceptability-label extension point."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
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


def acceptability_labels(
    records: Sequence[LabelRecord], epsilon: float
) -> dict[str, dict[str, bool] | None]:
    """Derive epsilon-acceptable labels; exclude all-fail queries from training."""
    raise NotImplementedError("acceptability labels are research logic (backlog T11)")


__all__ = ["LabelRecord", "LabelStore", "acceptability_labels", "from_traces"]

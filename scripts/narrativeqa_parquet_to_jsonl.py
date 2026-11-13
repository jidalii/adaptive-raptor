"""Convert NarrativeQA parquet shards to raw JSONL. No project imports.

This is the only step that needs pyarrow, which in turn needs numpy>=2 (the project pins
numpy 1.x). So it runs in an isolated environment:

    uv run --isolated --no-project --with pyarrow python scripts/narrativeqa_parquet_to_jsonl.py \
        --split validation --raw-dir data/raw/narrativeqa/2e643e73

``python -m adaptive_raptor.data.download --dataset narrativeqa`` prints this command
when the JSONL is missing. Output (in --raw-dir):
    <split>_documents.jsonl  one row per document (fields as in the parquet `document`)
    <split>_questions.jsonl  one row per question, in parquet row order:
                             {doc_id, index_in_doc, question, answers: [str, str]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--raw-dir", type=Path, required=True)
    a = p.parse_args()
    shards = sorted(a.raw_dir.glob(f"{a.split}-*.parquet"))
    if not shards:
        raise SystemExit(f"no {a.split}-*.parquet in {a.raw_dir}")
    docs: dict[str, dict] = {}
    per_doc: dict[str, int] = {}
    qrows = []
    for shard in shards:
        for r in pq.read_table(shard).to_pylist():
            d = r["document"]
            if d["id"] not in docs:
                docs[d["id"]] = {
                    "id": d["id"],
                    "kind": d["kind"],
                    "url": d["url"],
                    "title": d["summary"]["title"],
                    "word_count": d["word_count"],
                    "start": d["start"],
                    "end": d["end"],
                    "text": d["text"],
                }
            i = per_doc.get(d["id"], 0)
            per_doc[d["id"]] = i + 1
            qrows.append(
                {
                    "doc_id": d["id"],
                    "index_in_doc": i,
                    "question": r["question"]["text"],
                    "answers": [x["text"] for x in r["answers"]],
                }
            )
    for name, rows in (("documents", docs.values()), ("questions", qrows)):
        out = a.raw_dir / f"{a.split}_{name}.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out)
        print(f"wrote {out}")
    print(f"{a.split}: {len(docs)} documents, {len(qrows)} questions")


if __name__ == "__main__":
    main()

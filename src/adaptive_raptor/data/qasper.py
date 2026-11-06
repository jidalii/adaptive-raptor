"""QASPER: parse the official release into the shared schema, and load processed splits.

Official files (v0.3): ``qasper-{train,dev,test}-v0.3.json``. Answer extraction follows the
official ``qasper_evaluator.py`` exactly (extractive spans joined by ", ", free-form,
Yes/No, or "Unanswerable").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from adaptive_raptor.data.records import Document, GoldAnswer, Paragraph, Question

log = logging.getLogger(__name__)

SPLITS = ("train", "dev", "test")
PREPROCESS_VERSION = 1  # bump when parsing/normalization changes
PARAGRAPH_SEP = "\n\n"


def raw_filename(split: str, version: str = "v0.3") -> str:
    return f"qasper-{split}-{version}.json"


def build_document(paper_id: str, paper: dict[str, Any]) -> Document:
    """Title, abstract, then every non-empty section paragraph, separated by blank lines.

    Paragraph offsets point into the returned text, so evidence can be located exactly.
    """
    pieces: list[tuple[str, str]] = []
    if paper.get("abstract", "").strip():
        pieces.append(("Abstract", paper["abstract"].strip()))
    for section in paper.get("full_text") or []:
        name = section.get("section_name") or ""
        for para in section.get("paragraphs") or []:
            if para and para.strip():
                pieces.append((name, para.strip()))
    title = (paper.get("title") or "").strip()
    text = title
    paragraphs = []
    for name, ptext in pieces:
        if text:
            text += PARAGRAPH_SEP
        start = len(text)
        text += ptext
        paragraphs.append(Paragraph(section=name, text=ptext, start=start, end=len(text)))
    return Document(doc_id=paper_id, text=text, title=title, paragraphs=tuple(paragraphs))


def parse_answer(annotation: dict[str, Any]) -> GoldAnswer | None:
    info = annotation["answer"]
    aid = annotation.get("annotation_id", "")
    evidence = tuple(info.get("evidence") or ())
    highlighted = tuple(info.get("highlighted_evidence") or ())
    if info.get("unanswerable"):
        return GoldAnswer("Unanswerable", "none", (), (), aid)
    if info.get("extractive_spans"):
        text, kind = ", ".join(info["extractive_spans"]), "extractive"
    elif info.get("free_form_answer"):
        text, kind = info["free_form_answer"], "abstractive"
    elif info.get("yes_no"):
        text, kind = "Yes", "boolean"
    elif info.get("yes_no") is not None:
        text, kind = "No", "boolean"
    else:
        return None  # the official evaluator raises here; we skip and count it
    return GoldAnswer(text, kind, evidence, highlighted, aid)


def parse_split(raw: dict[str, Any]) -> tuple[list[Document], list[Question], dict[str, int]]:
    documents: list[Document] = []
    questions: list[Question] = []
    stats = {"skipped_annotations": 0, "questions_without_answers": 0}
    for paper_id in sorted(raw):
        paper = raw[paper_id]
        documents.append(build_document(paper_id, paper))
        for qa in paper.get("qas") or []:
            answers = []
            for ann in qa.get("answers") or []:
                parsed = parse_answer(ann)
                if parsed is None:
                    stats["skipped_annotations"] += 1
                else:
                    answers.append(parsed)
            if not answers:
                stats["questions_without_answers"] += 1
            questions.append(Question(qa["question_id"], paper_id, qa["question"], tuple(answers)))
    return documents, questions, stats


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def processed_split_paths(processed: Path, split: str) -> tuple[Path, Path]:
    return processed / split / "documents.jsonl", processed / split / "questions.jsonl"


def load_processed(processed: Path, split: str) -> tuple[list[Document], list[Question]]:
    docs_path, qs_path = processed_split_paths(processed, split)
    if not docs_path.is_file() or not qs_path.is_file():
        raise FileNotFoundError(
            f"processed QASPER split '{split}' not found under {processed}. "
            "Run: ./scripts/download_datasets.sh --dataset qasper"
        )
    docs = [Document.from_dict(d) for d in read_jsonl(docs_path)]
    qs = [Question.from_dict(q) for q in read_jsonl(qs_path)]
    return docs, qs

"""Dataset-independent document and question records (the shared internal schema)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Same answer-type names as the official QASPER evaluator.
AnswerType = Literal["extractive", "abstractive", "boolean", "none"]


@dataclass(frozen=True)
class Paragraph:
    section: str
    text: str
    start: int  # character offsets into Document.text
    end: int


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    paragraphs: tuple[Paragraph, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        paras = tuple(Paragraph(**p) for p in d.get("paragraphs", ()))
        return cls(doc_id=d["doc_id"], text=d["text"], title=d.get("title", ""), paragraphs=paras)


@dataclass(frozen=True)
class GoldAnswer:
    text: str
    answer_type: AnswerType
    evidence: tuple[str, ...] = ()
    highlighted_evidence: tuple[str, ...] = ()
    annotation_id: str = ""


@dataclass(frozen=True)
class Question:
    question_id: str
    doc_id: str
    text: str
    answers: tuple[GoldAnswer, ...] = field(default_factory=tuple)  # one per annotator

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        answers = tuple(
            GoldAnswer(
                text=a["text"],
                answer_type=a["answer_type"],
                evidence=tuple(a.get("evidence", ())),
                highlighted_evidence=tuple(a.get("highlighted_evidence", ())),
                annotation_id=a.get("annotation_id", ""),
            )
            for a in d.get("answers", ())
        )
        return cls(d["question_id"], d["doc_id"], d["text"], answers)


@dataclass
class DatasetSplit:
    name: str
    version: str
    split: str
    documents: list[Document]
    questions: list[Question]
    manifest: dict[str, Any] = field(default_factory=dict)

    def questions_by_doc(self) -> dict[str, list[Question]]:
        out: dict[str, list[Question]] = {d.doc_id: [] for d in self.documents}
        for q in self.questions:
            out.setdefault(q.doc_id, []).append(q)
        return out

"""QASPER answer F1, identical to the official ``qasper_evaluator.py`` (v0.3).

Per question: token F1 against every annotator's reference, take the max. The answer type
reported is the type of the best-matching reference (as in the official script).
"""

from __future__ import annotations

import re
import string
from collections import Counter

from adaptive_raptor.data.records import GoldAnswer


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    num_same = sum((Counter(pred) & Counter(gold)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, golds: tuple[GoldAnswer, ...]) -> tuple[float, str | None]:
    """Return (max F1 over references, answer type of the best reference)."""
    if not golds:
        return 0.0, None
    scored = [(token_f1_score(prediction, g.text), g.answer_type) for g in golds]
    best = sorted(scored, key=lambda x: x[0], reverse=True)[0]
    return best[0], best[1]


def evidence_recall(retrieved_texts: list[str], golds: tuple[GoldAnswer, ...]) -> float:
    """Diagnostic only (proposal 6.3). Research definition pending: a summary is credited
    only for evidence it actually contains."""
    raise NotImplementedError("evidence recall is a research diagnostic (backlog T14)")

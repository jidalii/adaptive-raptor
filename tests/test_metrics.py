"""Answer F1 mirrors the official qasper_evaluator.py (SQuAD normalization, max over refs)."""

import pytest

from adaptive_raptor.data.records import GoldAnswer
from adaptive_raptor.evaluation.metrics import answer_f1, normalize_answer, token_f1_score


def test_normalization():
    assert normalize_answer("The  Cat, sat!") == "cat sat"


@pytest.mark.parametrize(
    ("pred", "gold", "f1"),
    [
        ("the cat sat", "cat sat", 1.0),
        ("cat", "cat sat", 2 / 3),
        ("dog", "cat", 0.0),
        ("Unanswerable", "Unanswerable", 1.0),
        ("", "cat", 0.0),
    ],
)
def test_token_f1(pred, gold, f1):
    assert token_f1_score(pred, gold) == pytest.approx(f1)


def test_answer_f1_takes_best_reference_and_its_type():
    golds = (GoldAnswer("No", "boolean"), GoldAnswer("12 layers", "extractive"))
    assert answer_f1("12 layers", golds) == (1.0, "extractive")
    assert answer_f1("x", ()) == (0.0, None)

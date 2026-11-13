"""Sentence-level extractive compression (offline; no models, no API)."""

from __future__ import annotations

import numpy as np
import tiktoken

from adaptive_raptor.compression import (
    BM25,
    Unit,
    assemble,
    is_dependent,
    pool_units,
    rrf,
    select,
    sentence_units,
)

ENC = tiktoken.get_encoding("cl100k_base").encode


def test_units_follow_upstream_split_text_boundaries():
    # upstream split_text drops '.', so sentences are separated by two spaces
    text = (
        "GlossBERT uses gloss knowledge  We fine-tune BERT on SemCor3 0 training corpus  "
        "Results improve over prior work e g  on Senseval"
    )
    units = sentence_units(7, text)
    assert [u.text for u in units] == [
        "GlossBERT uses gloss knowledge",
        "We fine-tune BERT on SemCor3 0 training corpus",
        "Results improve over prior work e g on Senseval",
    ]
    assert all(u.leaf == 7 for u in units) and [u.pos for u in units] == [0, 1, 2]
    for u in units:  # spans point back into the leaf text
        assert " ".join(text[u.start : u.end].split()) == u.text


def test_decimal_pieces_are_merged():
    units = sentence_units(0, "Accuracy is 92  4% on the test set  The baseline is lower")
    assert [u.text for u in units] == ["Accuracy is 92 4% on the test set", "The baseline is lower"]


def test_pool_units_are_in_document_order():
    units = pool_units([(5, "Later text here  More later"), (2, "Early text here")])
    assert [u.key for u in units] == [(2, 0), (5, 0), (5, 1)]


def test_bm25_prefers_matching_unit():
    corpus = ["the model uses bert", "we evaluate on squad", "bert is fine tuned on squad"]
    s = BM25(corpus).score("which datasets do they evaluate on", corpus)
    assert int(np.argmax(s)) == 1


def test_rrf_combines_ranks():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])
    assert int(np.argmax(rrf(a, b))) == 1


def _units():
    texts = [
        "The corpus has 10k sentences from news",
        "It is annotated by three experts",
        "We train a BiLSTM tagger",
        "The corpus has 10k sentences from news articles",
        "Accuracy reaches 91 percent on the test split",
    ]
    return [Unit(0 if i < 3 else 1, i if i < 3 else i - 3, 0, 0, t) for i, t in enumerate(texts)]


def test_select_respects_budget_and_document_order():
    units = _units()
    scores = np.array([0.9, 0.8, 0.1, 0.85, 0.7])
    for budget in (5, 12, 20, 40, 400):
        picked = select(units, scores, budget, ENC)
        assert len(ENC(assemble(picked))) <= budget
        assert [u.key for u in picked] == sorted(u.key for u in picked)


def test_select_skips_near_duplicates():
    units = _units()
    scores = np.array([0.9, 0.1, 0.1, 0.85, 0.1])
    picked = select(units, scores, 400, ENC, redundancy_jaccard=0.8)
    texts = [u.text for u in picked]
    assert "The corpus has 10k sentences from news" in texts
    assert "The corpus has 10k sentences from news articles" not in texts


def test_neighbors_bring_predecessor_of_dependent_unit():
    units = _units()
    assert is_dependent(units[1].text)
    scores = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    alone = select(units, scores, 9, ENC, neighbors=False)
    assert [u.key for u in alone] == [(0, 1)]
    with_n = select(units, scores, 30, ENC, neighbors=True)
    keys = [u.key for u in with_n]
    assert keys.index((0, 0)) == keys.index((0, 1)) - 1


def test_assemble_joins_contiguous_units_with_space():
    a, b, c = Unit(3, 0, 0, 0, "A"), Unit(3, 1, 0, 0, "B"), Unit(5, 0, 0, 0, "C")
    assert assemble([c, b, a]) == "A B\n\nC"

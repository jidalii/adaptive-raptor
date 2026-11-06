"""Opt-in integration tests: real models, real API, official evaluator.

    uv run pytest -m integration

These download the SBERT model (~420 MB), call the DeepSeek API (a few cents), and execute
the official evaluator script shipped inside the QASPER archive. They are never part of the
default run.
"""

import importlib.util
import json
import os

import pytest
from dotenv import dotenv_values

from adaptive_raptor.data.records import GoldAnswer
from adaptive_raptor.evaluation.metrics import answer_f1
from tests.conftest import REAL_QASPER, ROOT

pytestmark = pytest.mark.integration
EVALUATOR = ROOT / "data/raw/qasper/v0.3/extracted/qasper_evaluator.py"

# pytest does not read .env (only the CLI does). Read it here without exporting it, so
# the default offline suite, which also imports this module, never sees an API key.
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY") or dotenv_values(ROOT / ".env").get(
    "DEEPSEEK_API_KEY"
)


@pytest.mark.skipif(not EVALUATOR.is_file(), reason="QASPER not downloaded")
def test_answer_f1_matches_official_evaluator():
    spec = importlib.util.spec_from_file_location("qasper_evaluator", EVALUATOR)
    qe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qe)
    raw = json.loads((EVALUATOR.parent / "qasper-dev-v0.3.json").read_text())
    gold = qe.get_answers_and_evidence(raw, True)
    from adaptive_raptor.data.qasper import load_processed

    _, qs = load_processed(REAL_QASPER, "dev")
    for name, fn in [
        ("unanswerable", lambda q: "Unanswerable"),
        ("yes", lambda q: "Yes"),
        ("first", lambda q: q.answers[0].text),
    ]:
        preds = {q.question_id: fn(q) for q in qs}
        official = qe.evaluate(gold, {k: {"answer": v, "evidence": []} for k, v in preds.items()})
        ours = sum(answer_f1(preds[q.question_id], q.answers)[0] for q in qs) / len(qs)
        assert ours == pytest.approx(official["Answer F1"], abs=1e-12), name


@pytest.mark.skipif(not DEEPSEEK_KEY, reason="DEEPSEEK_API_KEY not set in env or .env")
def test_deepseek_generation_one_call(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", DEEPSEEK_KEY)  # scoped to this test only
    from adaptive_raptor.config.schema import GenerationSection
    from adaptive_raptor.generation import build_generator

    gen = build_generator(GenerationSection(), cache_dir=tmp_path, use_cache=True)
    r = gen.generate("How many layers does the encoder have?", ["The encoder has 12 layers."])
    assert r.text and r.model_version and r.input_tokens > 0
    assert gen.generate(
        "How many layers does the encoder have?", ["The encoder has 12 layers."]
    ).cache_hit
    assert answer_f1(r.text, (GoldAnswer("12 layers", "extractive"),))[0] >= 0


def test_sbert_embedding_model_loads():
    from adaptive_raptor.config.schema import ModelRef
    from adaptive_raptor.models.embedding import make_embedding_model

    m = make_embedding_model(
        ModelRef(provider="sbert", model="sentence-transformers/multi-qa-mpnet-base-cos-v1")
    )
    assert len(m.create_embedding("hello")) == 768

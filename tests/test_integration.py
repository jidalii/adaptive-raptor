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
    # T03: no hidden reasoning; billed output is close to the visible answer length.
    assert r.reasoning_tokens == 0 and r.output_tokens <= 20, (r.output_tokens, r.text)
    assert gen.generate(
        "How many layers does the encoder have?", ["The encoder has 12 layers."]
    ).cache_hit
    assert answer_f1(r.text, (GoldAnswer("12 layers", "extractive"),))[0] >= 0


@pytest.mark.skipif(not DEEPSEEK_KEY, reason="DEEPSEEK_API_KEY not set in env or .env")
def test_deepseek_summary_is_non_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", DEEPSEEK_KEY)
    from adaptive_raptor.models.summarization import make_openai_compatible_summarizer

    s = make_openai_compatible_summarizer("deepseek", "deepseek-flash", tmp_path, True, 2)
    context = (
        "We train a 12-layer Transformer encoder with 8 attention heads on 1M sentence pairs. "
        "BLEU improves by 2.1 points over the LSTM baseline, mostly on long sentences."
    )
    text = s.summarize(context, max_tokens=100)  # RAPTOR's summarization_length
    assert text.strip()
    assert s.usage["reasoning_tokens"] == 0 and s.usage["output_tokens"] <= 100


@pytest.mark.skipif(not DEEPSEEK_KEY, reason="DEEPSEEK_API_KEY not set in env or .env")
def test_llm_profiler_one_call(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", DEEPSEEK_KEY)
    from adaptive_raptor.config.schema import ProfilerSection
    from adaptive_raptor.data.records import Document
    from adaptive_raptor.optimizer.profiler import LLMProfiler

    p = LLMProfiler(ProfilerSection(), cache_dir=tmp_path)
    doc = Document("d", "", title="Attention Is All You Need")
    r = p.profile("q", "How many attention heads does the model use?", doc)
    assert r.profile is not None, r.error
    assert r.profile.answer_form == "short_span" and r.profile.n_facts == 1
    assert r.confidence is None  # logprobs off by default (post-temperature, no signal at t=0)
    assert r.output_tokens <= 60
    assert p.profile("q", "How many attention heads does the model use?", doc).cache_hit


def test_sbert_embedding_model_loads():
    from adaptive_raptor.config.schema import ModelRef
    from adaptive_raptor.models.embedding import make_embedding_model

    m = make_embedding_model(
        ModelRef(provider="sbert", model="sentence-transformers/multi-qa-mpnet-base-cos-v1")
    )
    assert len(m.create_embedding("hello")) == 768

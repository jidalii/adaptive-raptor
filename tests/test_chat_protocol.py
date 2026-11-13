"""T03: thinking mode is always explicit, empty completions are errors (offline fakes)."""

from types import SimpleNamespace

import pytest

from adaptive_raptor.config.schema import GenerationSection
from adaptive_raptor.models.chat import (
    EmptyCompletionError,
    reasoning_tokens,
    thinking_extra_body,
    visible_text,
)


def _response(content, completion_tokens=5, reasoning=None, finish="stop"):
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=completion_tokens, completion_tokens_details=details
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage, model="deepseek-flash")


class _FakeClient:
    """Records request kwargs; returns queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_thinking_body_is_explicit_for_deepseek_only():
    assert thinking_extra_body("deepseek", False) == {"thinking": {"type": "disabled"}}
    assert thinking_extra_body("deepseek", True) == {"thinking": {"type": "enabled"}}
    assert thinking_extra_body("openai", False) is None


def test_empty_text_raises_and_reports_reasoning():
    with pytest.raises(EmptyCompletionError, match="reasoning_tokens=100"):
        visible_text(_response("  ", completion_tokens=100, reasoning=100, finish="length"))
    assert visible_text(_response(" Yes ")) == "Yes"
    assert reasoning_tokens(_response("x").usage) == 0


def _generator(tmp_path, client):
    from adaptive_raptor.generation import OpenAICompatibleGenerator

    gen = OpenAICompatibleGenerator(GenerationSection(), cache_dir=tmp_path, max_retries=1)
    gen._client = client
    return gen


def test_generator_sends_disabled_thinking_and_retries_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    client = _FakeClient([_response(""), _response("12 layers.", reasoning=0)])
    r = _generator(tmp_path, client).generate("How many layers?", ["12 layers."])
    assert r.text == "12 layers." and r.reasoning_tokens == 0
    assert len(client.calls) == 2  # empty -> retried once
    assert all(c["extra_body"] == {"thinking": {"type": "disabled"}} for c in client.calls)


def test_generator_fails_after_retries_on_persistent_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    client = _FakeClient([_response(""), _response("")])
    with pytest.raises(EmptyCompletionError):
        _generator(tmp_path, client).generate("q", ["c"])


def test_summarizer_sends_disabled_thinking_and_rejects_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-key")
    fake = _FakeClient([_response(""), _response("A summary.")])
    monkeypatch.setattr("openai.OpenAI", lambda **kw: fake)
    from adaptive_raptor.models.summarization import make_openai_compatible_summarizer

    s = make_openai_compatible_summarizer("deepseek", "deepseek-flash", tmp_path, True, 1)
    assert s.summarize("some context", max_tokens=100) == "A summary."
    assert all(c["extra_body"] == {"thinking": {"type": "disabled"}} for c in fake.calls)
    assert s.summarize("some context", max_tokens=100) == "A summary."  # cached
    assert s.usage["calls"] == 1 and s.usage["cache_hits"] == 1


def test_empty_summary_node_fails_the_build(tmp_path):
    from adaptive_raptor.chunking import build_chunker
    from adaptive_raptor.config.schema import (
        DatasetSection,
        FixedChunking,
        ModelRef,
        RaptorSection,
    )
    from adaptive_raptor.data import load_dataset
    from adaptive_raptor.indexing import EmptySummaryError, IndexBuilder
    from adaptive_raptor.models.embedding import make_embedding_model
    from adaptive_raptor.models.summarization import _base

    class Blank(_base()):
        def summarize(self, context, max_tokens=150):
            return ""

    cfg = RaptorSection(
        embedding=ModelRef(provider="mock", model="h"),
        summarization=ModelRef(provider="mock", model="m"),
        index_dir=tmp_path,
    )
    doc = load_dataset(DatasetSection(name="qasper_tiny", split="dev")).documents[0]
    v = FixedChunking(name="f100")
    b = IndexBuilder(
        cfg,
        dataset="t",
        dataset_version="t",
        embedding_model=make_embedding_model(cfg.embedding),
        summarizer_factory=Blank,
    )
    with pytest.raises(EmptySummaryError):
        b.build(doc.doc_id, v, build_chunker(v).chunk(doc), 224)

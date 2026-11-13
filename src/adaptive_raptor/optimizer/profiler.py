"""LLM query profiler (Plan A, docs/OPTIMIZER_PLAN.md WP2).

Following METIS (Ray et al., SOSP 2025, Sec. 4.1), an LLM describes the *question*, not
the answer: a short structured profile that the router maps to a retrieval
configuration. The profiler sees only the question and paper metadata (title, optionally
the abstract), never answers or evidence, so profiling test questions leaks nothing.

Confidence (METIS Sec. 5): the product over the four fields of the probability of the
first token of the chosen value, read from the API's logprobs. Low confidence or an
invalid profile sends the query to the router's fallback configuration.
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adaptive_raptor.cache import stable_hash
from adaptive_raptor.config.schema import ProfilerSection
from adaptive_raptor.data.records import Document

ANSWER_FORMS = ("yes_no", "short_span", "explanation", "unanswerable_likely")
GRANULARITIES = ("specific_detail", "section_level", "whole_paper")
MAX_FACTS = 5


class QueryProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_form: Literal["yes_no", "short_span", "explanation", "unanswerable_likely"]
    granularity: Literal["specific_detail", "section_level", "whole_paper"]
    n_facts: int = Field(ge=1, le=MAX_FACTS)
    cross_section: bool


FIELDS = tuple(QueryProfile.model_fields)

PROMPTS: dict[str, str] = {
    "profile_v1": """You classify a question about a scientific paper so that a retrieval \
system can decide how much context to fetch. Do not answer the question.

{metadata}Question: {question}

Return a JSON object with exactly these keys:
- "answer_form": "yes_no" (answer is yes or no), "short_span" (a name, number, setting or \
short phrase), "explanation" (a sentence or more), or "unanswerable_likely" (the paper \
probably does not say)
- "granularity": "specific_detail" (one number, name or setting in one place), \
"section_level" (one method, experiment or result), or "whole_paper" (main contributions, \
overall findings, limitations)
- "n_facts": integer 1-5, how many separate facts a complete answer needs
- "cross_section": true if those facts are likely in different sections, else false

Examples:
Question: What learning rate do they use? -> {{"answer_form": "short_span", \
"granularity": "specific_detail", "n_facts": 1, "cross_section": false}}
Question: What are the main limitations of the proposed approach? -> {{"answer_form": \
"explanation", "granularity": "whole_paper", "n_facts": 3, "cross_section": true}}

JSON:""",
    # v2 (2026-11-09): ~2.5x shorter. Drops the few-shot examples and "unanswerable_likely"
    # (never predicted by v1 on the 87 pilot questions; still accepted by the schema).
    "profile_v2": """Classify this question about a paper; do not answer it. Reply in JSON:
"answer_form": "yes_no", "short_span" (name, number, setting) or "explanation";
"granularity": "specific_detail" (one value), "section_level" (one method or result) or \
"whole_paper" (overall findings);
"n_facts": 1-5, facts needed;
"cross_section": true if the facts span sections.
{metadata}Question: {question}""",
    # v3 (2026-11-09): v2 trimmed to fit a <=150-token API budget (v2 measured 158.8).
    "profile_v3": """Don't answer; classify this question about a paper as JSON:
"answer_form": "yes_no"/"short_span"/"explanation",
"granularity": "specific_detail" (one value)/"section_level" (one method or result)/\
"whole_paper" (overall findings),
"n_facts": 1-5,
"cross_section": true if facts span sections.
{metadata}Question: {question}""",
}
RETRY_SUFFIX = "\n\nReturn only a valid JSON object with the four keys above."


def get_prompt(name_or_path: str) -> str:
    if name_or_path in PROMPTS:
        return PROMPTS[name_or_path]
    path = Path(name_or_path)
    if path.is_file():
        return path.read_text()
    raise KeyError(f"unknown profiler prompt '{name_or_path}'")


def abstract_of(doc: Document | None) -> str:
    if doc is None:
        return ""
    return " ".join(p.text for p in doc.paragraphs if p.section.strip().lower() == "abstract")


def render_prompt(template: str, inputs: Sequence[str], question: str, doc: Document | None):
    meta = ""
    if "title" in inputs and doc is not None and doc.title:
        meta += f"Paper title: {doc.title}\n"
    if "abstract" in inputs and (abstract := abstract_of(doc)):
        meta += f"Paper abstract: {abstract}\n"
    return template.format(metadata=meta, question=question)


@dataclass
class ProfileResult:
    question_id: str
    prompt_hash: str
    profile: QueryProfile | None  # None => no valid profile (see error)
    confidence: float | None = None  # None => not available
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model_version: str = ""
    cache_hit: bool = False
    attempts: int = 1
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["profile"] = self.profile.model_dump() if self.profile else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ProfileResult:
        d = dict(d)
        d["profile"] = QueryProfile(**d["profile"]) if d.get("profile") else None
        return cls(**d)


# --------------------------------------------------------------------------- confidence


def field_confidence(tokens: Sequence[tuple[str, float]] | None) -> float | None:
    """Product over FIELDS of p(first token of the value). None if not computable."""
    if not tokens:
        return None
    if all(lp == 0.0 for _, lp in tokens):
        # DeepSeek reports post-temperature logprobs: at temperature 0 every chosen token
        # is 0.0 (alternatives -9999). That is no signal, not certainty (NOTES 2026-11-09).
        return None
    text, starts = "", []
    for tok, _ in tokens:
        starts.append(len(text))
        text += tok
    logp = 0.0
    for name in FIELDS:
        m = re.search(rf'"{name}"\s*:\s*"?([^"\s,}}]+)', text)
        if m is None:
            return None
        pos = m.start(1)
        idx = max(i for i, s in enumerate(starts) if s <= pos)
        logp += tokens[idx][1]
    return math.exp(logp)


def parse_profile(text: str) -> QueryProfile:
    """Parse the model's JSON. Raises ValueError on anything but a valid profile."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        m = re.search(r"\{.*\}", text, flags=re.S)  # tolerate prose around the object
        if m is None:
            raise ValueError(f"no JSON object: {exc}") from exc
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("profile is not a JSON object")
    try:
        return QueryProfile(**data)
    except ValidationError as exc:
        raise ValueError(f"invalid profile: {exc.errors()[0]['msg']}") from exc


# --------------------------------------------------------------------------- profilers


class QueryProfiler(ABC):
    prompt_hash: str
    model_name: str

    @abstractmethod
    def profile(self, question_id: str, question: str, doc: Document | None) -> ProfileResult:
        """Profile one question. Never raises for a bad model output (see ``error``)."""


class LLMProfiler(QueryProfiler):
    def __init__(
        self,
        cfg: ProfilerSection,
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> None:
        from adaptive_raptor.models.chat import ChatClient

        self.cfg = cfg
        self.template = get_prompt(cfg.prompt)
        self.prompt_hash = stable_hash([self.template, sorted(cfg.inputs)])[:16]
        self.model_name = f"{cfg.provider}:{cfg.model}"
        self.chat = ChatClient(
            cfg.provider,
            cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
            thinking=False,
            api_key_env=cfg.api_key_env,
            base_url_env=cfg.base_url_env,
            timeout_s=cfg.request_timeout_s,
            cache_dir=cache_dir,
            use_cache=use_cache,
            max_retries=max_retries,
        )

    def profile(self, question_id: str, question: str, doc: Document | None) -> ProfileResult:
        prompt = render_prompt(self.template, self.cfg.inputs, question, doc)
        out = ProfileResult(question_id, self.prompt_hash, None, cache_hit=True, attempts=0)
        for suffix in ("", RETRY_SUFFIX):  # one retry with a stricter instruction
            out.attempts += 1
            try:
                r = self.chat.complete(
                    [{"role": "user", "content": prompt + suffix}],
                    namespace=f"profile/{self.prompt_hash}",
                    json_mode=self.cfg.json_mode,
                    logprobs=self.cfg.use_logprobs,
                )
            except Exception as exc:  # API failure after retries: recorded, routed to fallback
                out.error = f"{type(exc).__name__}: {exc}"
                out.cache_hit = False
                return out
            out.input_tokens += r.input_tokens
            out.output_tokens += r.output_tokens
            out.latency_ms += r.latency_ms
            out.model_version = r.model_version
            out.cache_hit = out.cache_hit and r.cache_hit
            try:
                out.profile = parse_profile(r.text)
            except ValueError as exc:
                out.error = str(exc)
                continue
            out.error = None
            out.confidence = field_confidence(r.logprobs) if self.cfg.use_logprobs else None
            return out
        return out


_YES_NO = re.compile(r"^(is|are|do|does|did|can|could|was|were|has|have|will|should)\b", re.I)


class MockProfiler(QueryProfiler):
    """Deterministic keyword rules. Offline and free; for tests and smoke runs only."""

    def __init__(self, cfg: ProfilerSection | None = None) -> None:
        self.prompt_hash = "mock-v1"
        self.model_name = "mock"

    def profile(self, question_id: str, question: str, doc: Document | None) -> ProfileResult:
        q = question.strip()
        low = q.lower()
        if _YES_NO.match(q):
            form = "yes_no"
        elif re.match(r"^(how many|how much|what is the|which|who|when)\b", low):
            form = "short_span"
        else:
            form = "explanation"
        if re.search(r"\b(main|overall|contribution|limitation|propose)", low):
            gran = "whole_paper"
        elif re.match(r"^(how|why|describe)\b", low):
            gran = "section_level"
        else:
            gran = "specific_detail"
        n = min(MAX_FACTS, 1 + low.count(" and ") + low.count(","))
        profile = QueryProfile(answer_form=form, granularity=gran, n_facts=n, cross_section=n > 1)
        return ProfileResult(
            question_id,
            self.prompt_hash,
            profile,
            confidence=1.0,
            input_tokens=len(q.split()) + 200,
            output_tokens=30,
            model_version="mock",
        )


def build_profiler(
    cfg: ProfilerSection,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    max_retries: int = 3,
) -> QueryProfiler:
    if cfg.provider == "mock":
        return MockProfiler(cfg)
    return LLMProfiler(cfg, cache_dir=cache_dir, use_cache=use_cache, max_retries=max_retries)


# --------------------------------------------------------------------------- store


class ProfileStore:
    """Profiles keyed by (question_id, prompt_hash); JSONL on disk, append-only."""

    def __init__(self, results: Iterable[ProfileResult] = ()) -> None:
        self._by_key: dict[tuple[str, str], ProfileResult] = {}
        for r in results:
            self.add(r)

    def add(self, r: ProfileResult) -> None:
        self._by_key.setdefault((r.question_id, r.prompt_hash), r)  # first wins

    def get(self, question_id: str, prompt_hash: str) -> ProfileResult | None:
        return self._by_key.get((question_id, prompt_hash))

    def __len__(self) -> int:
        return len(self._by_key)

    def results(self, prompt_hash: str | None = None) -> list[ProfileResult]:
        return [r for (_, h), r in self._by_key.items() if prompt_hash in (None, h)]

    @classmethod
    def load(cls, paths: Iterable[Path]) -> ProfileStore:
        store = cls()
        for path in paths:
            with Path(path).open() as f:
                for line in f:
                    if line.strip():
                        store.add(ProfileResult.from_dict(json.loads(line)))
        return store

    @staticmethod
    def append(path: Path, results: Iterable[ProfileResult]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("a") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")
                n += 1
        return n


class CachedProfiler(QueryProfiler):
    """Serve stored profiles first; call the underlying profiler only on a miss."""

    def __init__(self, inner: QueryProfiler, store: ProfileStore) -> None:
        self.inner, self.store = inner, store
        self.prompt_hash, self.model_name = inner.prompt_hash, inner.model_name
        self.misses = 0

    def profile(self, question_id: str, question: str, doc: Document | None) -> ProfileResult:
        hit = self.store.get(question_id, self.prompt_hash)
        if hit is not None:
            return hit
        self.misses += 1
        r = self.inner.profile(question_id, question, doc)
        self.store.add(r)
        return r


__all__ = [
    "ANSWER_FORMS",
    "FIELDS",
    "GRANULARITIES",
    "CachedProfiler",
    "LLMProfiler",
    "MockProfiler",
    "ProfileResult",
    "ProfileStore",
    "QueryProfile",
    "QueryProfiler",
    "build_profiler",
    "field_confidence",
    "parse_profile",
    "render_prompt",
]

"""Provider-neutral answer generation, prompts, caching, and factory."""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from adaptive_raptor.cache import JsonCache, stable_hash
from adaptive_raptor.config.schema import GenerationSection
from adaptive_raptor.models.chat import (
    CHAT_PROTOCOL_VERSION,
    reasoning_tokens,
    thinking_extra_body,
    visible_text,
)
from adaptive_raptor.models.credentials import resolve

log = logging.getLogger(__name__)

TEMPLATES: dict[str, str] = {
    "qasper_default": (
        "Answer the question using only the context below. If the context does not "
        'contain the answer, reply exactly "Unanswerable". For yes/no questions, reply '
        '"Yes" or "No". Be concise.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:'
    ),
    "qasper_paraphrase": (
        "Using nothing but the passages provided, respond to the question. Reply "
        '"Unanswerable" if the passages are insufficient; reply "Yes" or "No" for '
        "yes/no questions. Keep the answer short.\n\nPassages:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    # NarrativeQA: references are short free-form phrases (about 4-5 words).
    "narrativeqa_default": (
        "Answer the question about the story using only the context below. Answer with a "
        "short phrase or one short sentence.\n\nContext:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    "narrativeqa_paraphrase": (
        "Using nothing but the passages from the story provided, respond to the question. "
        "Keep the answer to a brief phrase or a single short sentence.\n\nPassages:\n"
        "{context}\n\nQuestion: {question}\nAnswer:"
    ),
    # Answer + evidence-sufficiency flag in one call (cascade first stage). Everything up to
    # "Question: ..." is identical to qasper_default, so a retry with qasper_default and a
    # larger context (which starts with the same chunks) shares the instructions + first
    # chunks as a prompt prefix (provider prefix caching).
    "qasper_sufficiency_v1": (
        "Answer the question using only the context below. If the context does not "
        'contain the answer, reply exactly "Unanswerable". For yes/no questions, reply '
        '"Yes" or "No". Be concise.\n\nContext:\n{context}\n\nQuestion: {question}\n'
        'Then, on a final new line, write "Evidence: SUFFICIENT", "Evidence: PARTIAL" or '
        '"Evidence: INSUFFICIENT": does the context contain all the evidence needed for a '
        "complete answer?\nAnswer:"
    ),
}
SUFFICIENCY_LEVELS = ("sufficient", "partial", "insufficient")
_SUFFICIENCY_LINE = re.compile(
    r"^\W*evidence\W*:\W*(sufficient|partial|insufficient)\W*$", re.IGNORECASE
)


def wants_sufficiency(template: str) -> bool:
    return "Evidence: SUFFICIENT" in template


def split_sufficiency(text: str) -> tuple[str, str | None]:
    """Split a response into (answer, sufficiency); sufficiency is None if no flag line.

    The last line that is only an evidence flag is removed; the rest is the answer that
    gets scored. A missing flag leaves the text unchanged."""
    lines = text.strip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = _SUFFICIENCY_LINE.match(lines[i].strip())
        if m:
            answer = "\n".join(lines[:i] + lines[i + 1 :]).strip()
            answer = re.sub(r"^\W*answer\W*:\s*", "", answer, flags=re.IGNORECASE)
            return answer, m.group(1).lower()
    return text.strip(), None


def get_template(name_or_path: str) -> str:
    if name_or_path in TEMPLATES:
        return TEMPLATES[name_or_path]
    path = Path(name_or_path)
    if path.is_file():
        return path.read_text()
    raise KeyError(f"unknown prompt template '{name_or_path}'")


def render(template: str, question: str, contexts: list[str]) -> str:
    return template.format(context="\n\n".join(contexts), question=question)


@dataclass
class GenerationResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model_version: str = ""
    cache_hit: bool = False
    reasoning_tokens: int = 0
    sufficiency: str | None = None  # only for sufficiency templates
    raw_text: str = ""  # full response when it was post-processed (sufficiency templates)
    prompt_cache_hit_tokens: int = 0  # provider prefix-cache hits (billed at a lower price)


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[str]) -> GenerationResult:
        """Answer ``question`` using only ``contexts``."""


class MockGenerator(Generator):
    """Deterministic offline generator for tests and smoke runs."""

    def __init__(self, n_words: int = 12) -> None:
        self.n_words = n_words

    def generate(self, question: str, contexts: list[str]) -> GenerationResult:
        text = (
            " ".join(re.findall(r"\S+", contexts[0])[: self.n_words])
            if contexts
            else "Unanswerable"
        )
        input_tokens = sum(len(context.split()) for context in contexts) + len(question.split())
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=len(text.split()),
            model_version="mock",
        )


class OpenAICompatibleGenerator(Generator):
    """Cached DeepSeek/OpenAI chat-completions adapter."""

    def __init__(
        self,
        cfg: GenerationSection,
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.cfg = cfg
        self.template = get_template(cfg.prompt_template)
        self.cache = JsonCache(cache_dir, enabled=use_cache)
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key, base_url = resolve(self.cfg.provider, self.cfg.api_key_env, self.cfg.base_url_env)
            self._client = OpenAI(
                api_key=key,
                base_url=base_url,
                timeout=self.cfg.request_timeout_s,
                max_retries=0,
            )
        return self._client

    def _settings(self) -> dict:
        return {
            "provider": self.cfg.provider,
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "thinking": self.cfg.thinking,
            "protocol": CHAT_PROTOCOL_VERSION,
        }

    def generate(self, question: str, contexts: list[str]) -> GenerationResult:
        prompt = render(self.template, question, contexts)
        key = stable_hash(["generate", self._settings(), prompt])
        hit = self.cache.get(key)
        if hit is not None:
            return GenerationResult(**{**hit, "cache_hit": True})

        from tenacity import retry, stop_after_attempt, wait_random_exponential

        @retry(
            wait=wait_random_exponential(min=1, max=30),
            stop=stop_after_attempt(self.max_retries + 1),
            reraise=True,
        )
        def call():
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_output_tokens,
                extra_body=thinking_extra_body(self.cfg.provider, self.cfg.thinking),
            )
            return response, visible_text(response)  # empty text raises -> retried

        started = time.perf_counter()
        response, text = call()
        usage = response.usage
        raw, sufficiency = "", None
        if wants_sufficiency(self.template):
            raw = text
            text, sufficiency = split_sufficiency(text)
        result = GenerationResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            model_version=response.model or self.cfg.model,
            reasoning_tokens=reasoning_tokens(usage),
            sufficiency=sufficiency,
            raw_text=raw,
            prompt_cache_hit_tokens=getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
        )
        self.cache.put(key, {k: v for k, v in result.__dict__.items() if k != "cache_hit"})
        return result


def build_generator(
    cfg: GenerationSection,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    max_retries: int = 3,
) -> Generator:
    if cfg.provider == "mock":
        return MockGenerator()
    if cfg.provider in ("deepseek", "openai"):
        return OpenAICompatibleGenerator(
            cfg, cache_dir=cache_dir, use_cache=use_cache, max_retries=max_retries
        )
    raise NotImplementedError(f"generation provider '{cfg.provider}' is not implemented")


__all__ = [
    "GenerationResult",
    "Generator",
    "MockGenerator",
    "OpenAICompatibleGenerator",
    "build_generator",
    "get_template",
    "render",
    "split_sufficiency",
    "wants_sufficiency",
]

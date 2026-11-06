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
}


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
        if cfg.thinking:
            log.warning("generation.thinking=true is recorded but not sent to the API yet")

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
            return self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_output_tokens,
            )

        started = time.perf_counter()
        response = call()
        usage = response.usage
        result = GenerationResult(
            text=(response.choices[0].message.content or "").strip(),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            model_version=response.model or self.cfg.model,
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
]

"""Summarization models for RAPTOR, as upstream ``BaseSummarizationModel`` instances.

The OpenAI-compatible summarizer reuses upstream RAPTOR's prompt verbatim
(``GPT3TurboSummarizationModel``), but takes the model name, endpoint and key from
config/env so DeepSeek works. Responses are cached on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

from adaptive_raptor.cache import JsonCache, stable_hash
from adaptive_raptor.models.credentials import resolve

# Upstream raptor/SummarizationModels.py, GPT3TurboSummarizationModel.summarize
UPSTREAM_SYSTEM = "You are a helpful assistant."
UPSTREAM_TEMPLATE = (
    "Write a summary of the following, including as many key details as possible: {context}:"
)


def _base():
    from raptor.SummarizationModels import BaseSummarizationModel

    return BaseSummarizationModel


def make_mock_summarizer():
    class MockSummarizationModel(_base()):
        """First words of the context. Offline and deterministic. Not for results."""

        def summarize(self, context, max_tokens=150):
            words = re.findall(r"\S+", context)
            return " ".join(words[: max(8, max_tokens // 2)])

    return MockSummarizationModel()


def make_openai_compatible_summarizer(
    provider: str, model: str, cache_dir: Path | None, use_cache: bool, max_retries: int
):
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    api_key, base_url = resolve(provider)
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = JsonCache(cache_dir, enabled=use_cache)

    class OpenAICompatibleSummarizationModel(_base()):
        def __init__(self) -> None:
            self.model = model
            self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}

        @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(max_retries + 1))
        def _call(self, context: str, max_tokens: int) -> dict:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": UPSTREAM_SYSTEM},
                    {"role": "user", "content": UPSTREAM_TEMPLATE.format(context=context)},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            usage = resp.usage
            return {
                "text": resp.choices[0].message.content or "",
                "model_version": resp.model,
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }

        def summarize(self, context, max_tokens=150):
            key = stable_hash(["summarize", provider, self.model, max_tokens, context])
            hit = cache.get(key)
            if hit is not None:
                self.usage["cache_hits"] += 1
                return hit["text"]
            out = self._call(context, max_tokens)
            cache.put(key, out)
            self.usage["calls"] += 1
            self.usage["input_tokens"] += out["input_tokens"]
            self.usage["output_tokens"] += out["output_tokens"]
            return out["text"]

    return OpenAICompatibleSummarizationModel()


def make_summarization_model(ref, *, cache_dir: Path | None, use_cache: bool, max_retries: int):
    if ref.provider == "mock":
        return make_mock_summarizer()
    if ref.provider in ("deepseek", "openai"):
        return make_openai_compatible_summarizer(
            ref.provider, ref.model, cache_dir, use_cache, max_retries
        )
    raise NotImplementedError(f"summarization provider '{ref.provider}' is not supported")

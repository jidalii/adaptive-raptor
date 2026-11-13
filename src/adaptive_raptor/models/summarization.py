"""Summarization models for RAPTOR, as upstream ``BaseSummarizationModel`` instances.

The OpenAI-compatible summarizer reuses upstream RAPTOR's prompt verbatim
(``GPT3TurboSummarizationModel``) unless ``raptor.summarization_prompt`` names an
override, but takes the model name, endpoint and key from
config/env so DeepSeek works. Responses are cached on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

from adaptive_raptor.cache import JsonCache, stable_hash
from adaptive_raptor.models.chat import (
    CHAT_PROTOCOL_VERSION,
    reasoning_tokens,
    thinking_extra_body,
    visible_text,
)
from adaptive_raptor.models.credentials import resolve

# Upstream raptor/SummarizationModels.py, GPT3TurboSummarizationModel.summarize
UPSTREAM_SYSTEM = "You are a helpful assistant."
UPSTREAM_TEMPLATE = (
    "Write a summary of the following, including as many key details as possible: {context}:"
)
# Prompt overrides (raptor.summarization_prompt). "upstream" = the verbatim upstream prompt;
# its cache keys and index keys are unchanged. Other prompts change both keys.
SUMMARY_PROMPTS: dict[str, tuple[str, str]] = {
    "upstream": (UPSTREAM_SYSTEM, UPSTREAM_TEMPLATE),
    # 2026-11-10: concise, complete, evidence-preserving (summary-prompt pilot)
    "evidence_dense_v1": (
        UPSTREAM_SYSTEM,
        "Summarize the following text in about 150 words. Keep the concrete facts a reader "
        "would need to answer questions about it: the problem, the methods and model names, "
        "datasets, numbers and results, and how they relate. Do not add general statements "
        "that are not in the text. Finish with a complete sentence.\n\nText:\n{context}",
    ),
}


def summary_prompt(name: str) -> tuple[str, str]:
    if name not in SUMMARY_PROMPTS:
        raise KeyError(f"unknown summarization prompt '{name}'")
    return SUMMARY_PROMPTS[name]


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
    provider: str,
    model: str,
    cache_dir: Path | None,
    use_cache: bool,
    max_retries: int,
    prompt: str = "upstream",
):
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    api_key, base_url = resolve(provider)
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = JsonCache(cache_dir, enabled=use_cache)
    system, template = summary_prompt(prompt)
    # the upstream prompt keeps its historical cache key; overrides add the prompt text
    prompt_key = [] if prompt == "upstream" else [stable_hash([system, template])]

    class OpenAICompatibleSummarizationModel(_base()):
        def __init__(self) -> None:
            self.model = model
            self.usage = {
                "calls": 0,
                "cache_hits": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            }

        @retry(
            wait=wait_random_exponential(min=1, max=20),
            stop=stop_after_attempt(max_retries + 1),
            reraise=True,
        )
        def _call(self, context: str, max_tokens: int) -> dict:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": template.format(context=context)},
                ],
                max_tokens=max_tokens,
                temperature=0,
                extra_body=thinking_extra_body(provider, thinking=False),
            )
            usage = resp.usage
            return {
                "text": visible_text(resp),  # empty summary raises -> retried, then fails
                "model_version": resp.model,
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "reasoning_tokens": reasoning_tokens(usage),
            }

        def summarize(self, context, max_tokens=150):
            key = stable_hash(
                [
                    "summarize",
                    CHAT_PROTOCOL_VERSION,
                    provider,
                    self.model,
                    max_tokens,
                    context,
                    *prompt_key,
                ]
            )
            hit = cache.get(key)
            if hit is not None:
                self.usage["cache_hits"] += 1
                return hit["text"]
            out = self._call(context, max_tokens)
            cache.put(key, out)
            self.usage["calls"] += 1
            self.usage["input_tokens"] += out["input_tokens"]
            self.usage["output_tokens"] += out["output_tokens"]
            self.usage["reasoning_tokens"] += out["reasoning_tokens"]
            return out["text"]

    return OpenAICompatibleSummarizationModel()


def make_summarization_model(
    ref,
    *,
    cache_dir: Path | None,
    use_cache: bool,
    max_retries: int,
    prompt: str = "upstream",
):
    summary_prompt(prompt)  # fail fast on an unknown name
    if ref.provider == "mock":
        return make_mock_summarizer()
    if ref.provider in ("deepseek", "openai"):
        return make_openai_compatible_summarizer(
            ref.provider, ref.model, cache_dir, use_cache, max_retries, prompt
        )
    raise NotImplementedError(f"summarization provider '{ref.provider}' is not supported")

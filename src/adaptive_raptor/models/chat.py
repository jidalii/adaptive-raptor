"""Helpers shared by every OpenAI-compatible chat call (generation, summarization).

DeepSeek V4.x Flash runs with *thinking enabled by default*: hidden reasoning tokens are
billed as output and count against ``max_tokens``. With a small cap (RAPTOR's 100-token
summaries) the reasoning used the whole budget and the visible text came back empty
(backlog T03). Every request therefore states the thinking mode explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_raptor.cache import JsonCache, stable_hash
from adaptive_raptor.models.credentials import resolve

# Bump when the request protocol changes in a way that changes outputs, so response
# caches written under the old protocol are never reused.
CHAT_PROTOCOL_VERSION = 2


class EmptyCompletionError(RuntimeError):
    """The API returned no visible text. Retried, then surfaced as an error."""


def thinking_extra_body(provider: str, thinking: bool) -> dict[str, Any] | None:
    """Request body field that sets DeepSeek's thinking mode; None for other providers.

    Verified live on 2026-11-09 against ``deepseek-flash``: without this field the
    response carries ``reasoning_content`` and ``reasoning_tokens`` > 0; with
    ``{"type": "disabled"}`` both are absent and billed output equals the visible text.
    """
    if provider != "deepseek":
        return None
    return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    return int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0


def visible_text(response: Any) -> str:
    """Return the message text, raising EmptyCompletionError if there is none."""
    text = (response.choices[0].message.content or "").strip()
    if not text:
        finish = getattr(response.choices[0], "finish_reason", None)
        raise EmptyCompletionError(
            f"empty completion (finish_reason={finish}, "
            f"reasoning_tokens={reasoning_tokens(response.usage)})"
        )
    return text


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    model_version: str = ""
    logprobs: list[tuple[str, float]] | None = None  # (token, logprob) of the output
    cache_hit: bool = False


class ChatClient:
    """Cached, retrying OpenAI-compatible chat client with explicit thinking mode.

    Used by the LLM query profiler. Cache keys cover the provider, model, every request
    setting, ``CHAT_PROTOCOL_VERSION`` and the messages, so a changed prompt or setting
    is never served a stale answer.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 200,
        thinking: bool = False,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        timeout_s: float = 60.0,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens, self.thinking = temperature, max_tokens, thinking
        self.api_key_env, self.base_url_env, self.timeout_s = api_key_env, base_url_env, timeout_s
        self.cache = JsonCache(cache_dir, enabled=use_cache)
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key, base_url = resolve(self.provider, self.api_key_env, self.base_url_env)
            self._client = OpenAI(
                api_key=key, base_url=base_url, timeout=self.timeout_s, max_retries=0
            )
        return self._client

    def settings(self, json_mode: bool, logprobs: bool) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": self.thinking,
            "json_mode": json_mode,
            "logprobs": logprobs,
            "protocol": CHAT_PROTOCOL_VERSION,
        }

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        namespace: str,
        json_mode: bool = False,
        logprobs: bool = False,
    ) -> ChatResult:
        key = stable_hash([namespace, self.settings(json_mode, logprobs), messages])
        hit = self.cache.get(key)
        if hit is not None:
            lp = hit.get("logprobs")
            return ChatResult(
                **{**hit, "logprobs": [tuple(x) for x in lp] if lp else lp, "cache_hit": True}
            )

        from tenacity import retry, stop_after_attempt, wait_random_exponential

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": thinking_extra_body(self.provider, self.thinking),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if logprobs:
            kwargs["logprobs"] = True

        @retry(
            wait=wait_random_exponential(min=1, max=30),
            stop=stop_after_attempt(self.max_retries + 1),
            reraise=True,
        )
        def call():
            response = self.client.chat.completions.create(**kwargs)
            return response, visible_text(response)

        started = time.perf_counter()
        response, text = call()
        usage = response.usage
        lp = getattr(response.choices[0], "logprobs", None)
        tokens = getattr(lp, "content", None) if lp else None
        result = ChatResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=reasoning_tokens(usage),
            latency_ms=(time.perf_counter() - started) * 1000,
            model_version=response.model or self.model,
            logprobs=[(t.token, float(t.logprob)) for t in tokens] if tokens else None,
        )
        record = {k: v for k, v in result.__dict__.items() if k != "cache_hit"}
        self.cache.put(key, record)
        return result


__all__ = [
    "CHAT_PROTOCOL_VERSION",
    "ChatClient",
    "ChatResult",
    "EmptyCompletionError",
    "reasoning_tokens",
    "thinking_extra_body",
    "visible_text",
]

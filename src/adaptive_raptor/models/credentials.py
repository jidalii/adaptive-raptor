"""Provider registry and credential checks. Never prints or stores secret values."""

from __future__ import annotations

import os
from dataclasses import dataclass


class MissingCredentialsError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderSpec:
    api_key_env: str | None
    base_url_env: str | None = None
    default_base_url: str | None = None


PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "openai": ProviderSpec("OPENAI_API_KEY", "OPENAI_BASE_URL", None),
    "anthropic": ProviderSpec("ANTHROPIC_API_KEY", None, None),
    "sbert": ProviderSpec(None),
    "mock": ProviderSpec(None),
}


def resolve(provider: str, api_key_env: str | None = None, base_url_env: str | None = None):
    """Return (api_key, base_url) for a provider, raising if the key is missing."""
    spec = PROVIDERS[provider]
    key_env = api_key_env or spec.api_key_env
    if key_env is None:
        return None, None
    key = os.environ.get(key_env)
    if not key:
        raise MissingCredentialsError(
            f"{key_env} is not set (needed for provider '{provider}'). "
            "Copy .env.example to .env and add the key, or export it in your shell."
        )
    url_env = base_url_env or spec.base_url_env
    base_url = (os.environ.get(url_env) if url_env else None) or spec.default_base_url
    return key, base_url


def required_credentials(config) -> dict[str, str]:
    """Env vars an experiment needs, as {env_var: purpose}. Mock/local providers need none."""
    needed: dict[str, str] = {}
    gen = config.generation
    if gen.provider != "mock":
        env = gen.api_key_env or PROVIDERS[gen.provider].api_key_env
        if env:
            needed[env] = f"answer generation ({gen.provider})"
    for purpose, ref in (
        ("RAPTOR summarization", config.raptor.summarization),
        ("RAPTOR embeddings", config.raptor.embedding),
    ):
        env = PROVIDERS[ref.provider].api_key_env
        if env:
            needed.setdefault(env, f"{purpose} ({ref.provider})")
    opt = config.optimizer
    profiling = str(config.experiment.type) == "query_profiling"
    if (opt.enabled or profiling) and opt.uses_profiler and opt.profiler.provider != "mock":
        env = opt.profiler.api_key_env or PROVIDERS[opt.profiler.provider].api_key_env
        if env:
            needed.setdefault(env, f"LLM query profiler ({opt.profiler.provider})")
    return needed


def credential_status(config) -> dict[str, bool]:
    return {env: bool(os.environ.get(env)) for env in required_credentials(config)}

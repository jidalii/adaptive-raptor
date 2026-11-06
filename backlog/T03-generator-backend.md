# T03 · Generator backend (DeepSeek)

Tier: 1 · Owner: _ · Depends: T01

## What

- `generation/__init__.py` (`OpenAICompatibleGenerator`) and `models/summarization.py`: OpenAI-compatible clients for DeepSeek V4.1 Flash (`deepseek-flash`):
  - **non-thinking mode**, temperature 0, output-token cap from config;
  - retries with backoff and `runtime.concurrency`;
  - credentials from `.env`, with a clear error if missing.
- Each call records input/output tokens, latency and the model version returned by the API.
- Disk response cache (`.cache/generations`, `.cache/summaries`), so reruns are free.
- Cost meter from `generation.price_*_per_mtok`.
- Prompts `qasper_default` and `qasper_paraphrase` (run B) in `generation/__init__.py` (`TEMPLATES`).

## Done when

- A live call returns non-empty text, and its billed output tokens are close to the visible answer length (no hidden reasoning).
- A live summary call returns a non-empty summary within `summarization_length`.
- A repeated call is served from the cache.
- Unit tests use the mock generator; no network in the default suite.

## Description

Labels, θ and every H1/H2 number are tied to this generator (proposal §6.4), so its behavior and cost must be pinned and logged.

## Notes

- Nov 6: client, cache, retries, usage logging done. `uv run pytest -m integration` passes, including one live DeepSeek call and a cache hit (integration tests now read `.env` themselves; the default suite never sees the key).
- **BLOCKER (found Nov 6 in the first real baseline run):** `deepseek-flash` is spending hidden reasoning tokens.
  - All 18 RAPTOR summaries came back as **empty strings**, each billed exactly 100 output tokens (= `summarization_length`): the reasoning used the whole cap before any visible text.
  - Answers were non-empty, but short answers such as "Transformer layer." were billed 135–164 output tokens.
  - Fix: find DeepSeek's documented way to disable thinking for this model (request parameter or a non-reasoning model name), send it from both clients, and make `generation.thinking` actually control it.
  - Guards to add: treat an empty completion as an error (retry or fail the run), and log reasoning tokens separately if the API reports them.
- The API returns only the alias `deepseek-flash` as the model version, so silent model updates can't be detected from responses. Record the date of each run, and check whether the API exposes a pinned version.
- Pricing (user-reported, off-peak): $0.15 per 1M input, $0.60 per 1M output. Confirm peak vs. off-peak hours before T08/T12.
- Temperature 0 isn't guaranteed deterministic on hosted APIs; T08 measures run A vs. run B disagreement.

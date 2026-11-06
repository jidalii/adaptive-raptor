# T05 · Retrieval executor (strategy × budget)

Tier: 1 · Owner: _ · Depends: T04

## What

- `retrieval.py` (`RetrievalExecutor.retrieve`): `leaf_only` = L0, `summary_only` = L≥1, `collapsed` = all levels. Each runs upstream `retrieve_information_collapse_tree` on a level-restricted view, with `top_k` = all nodes so the token budget is the only limit.
- Returns node ids, levels, texts, **tokens actually used**, and latency.
- `RetrievalExecutor.retrieve_original`: upstream `TreeRetriever.retrieve` unchanged (config id `raptor_original`).
- The query is embedded once per question and reused by every config (and by the router).

## Done when

- For every catalog config, `tokens_used ≤ budget` and the levels match the strategy.
- `collapsed@B` returns the same nodes as upstream collapsed retrieval at `max_tokens=B`.
- Retrieval latency is recorded on real indexes.

## Description

Each configuration is a different access path over the same index; its cost is the context tokens sent to the generator.

## Notes

- Nov 6: all three checks pass. Budget and level contracts plus equality with upstream are tested offline. On the first real indexes (2 papers), retrieval took 1.1–9.4 ms per query against 1.3–1.9 s for generation, which confirms proposal §1: retrieval latency is negligible.
- `summary_only` hasn't been exercised on real data yet, because the real summaries were empty (T03/T04 blocker).

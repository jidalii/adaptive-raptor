# T04 · RAPTOR index from custom leaves

Tier: 1 · Owner: _ · Depends: T01, T02, T03 (summarizer)

## What

- `indexing/builder.py`: `build_from_chunks` = upstream `build_from_text` without its `split_text` call, so any chunker supplies the leaves; upstream clustering and summarization unchanged.
- Summarizer: upstream's prompt verbatim, DeepSeek via `models/summarization.py`, cached.
- Embeddings: upstream `SBertEmbeddingModel` (`multi-qa-mpnet-base-cos-v1`), memoized.
- Seed control without editing upstream, including `random_state=seed` for UMAP (`raptor.deterministic_umap`).
- `indexing/store.py`: strict cache keys; a mismatched cached index is rejected, never reused.
- Build stats in `indexes.jsonl`: nodes per layer, summary tokens, build time.
- `FixedSizeChunker` (`chunking/__init__.py`): upstream `split_text` with approximate offsets.

## Done when

- F-100 built for 3 pilot papers with SBERT + DeepSeek, with **non-empty** summaries at every layer above 0, cached, and reused on a rerun.
- Same seed → identical tree across processes (tested offline).
- Build cost per paper recorded in `NOTES.md` (Spend log).

## Description

Everything above the leaves depends on the leaves, so leaf construction is pluggable (T09) while clustering and summarization stay exactly RAPTOR's (proposal §4.4).

## Notes

- Nov 6: builder, cache, determinism (measured: unseeded upstream UMAP gave 3 different trees in 4 processes; now identical), and an offline end-to-end smoke run all done.
- **BLOCKED by T03.** The first real build (2 dev papers: 68 → 9 and 42 → 9 nodes, 24 s and 15 s) produced **all-empty summaries**. Empty summary text was never retrieved, so the "real baseline" was effectively leaf-only (every retrieved node was layer 0).
- After the T03 fix:
  1. Add the summarizer settings that change the output (the thinking switch, prompt version) to the index cache key, or bump `INDEX_FORMAT_VERSION`. Otherwise the cache will reuse the broken indexes, since provider and model name are unchanged.
  2. Delete `indexes/` and `.cache/summaries/` from the Nov 6 run.
  3. Add a build-time guard that fails if any summary node is empty.
- **Nov 9: unblocked.**
  - `INDEX_FORMAT_VERSION=2`, and the index key gains `chat_protocol`.
  - The stale `indexes/qasper` and `.cache/summaries` were deleted.
  - `build` raises `EmptySummaryError` if any summary node is empty.
  - F-100 was built for 3 pilot papers: 41/41 summaries non-empty, and 65/330 retrieved nodes come from summary layers. The rerun reused all 3.
  - Cost ≈ $0.0018 per paper (`NOTES.md`). **Done.**
  - **Open for T08:** every summary hits the 100-token cap and ends mid-sentence. Upstream does the same. Decide whether to keep fidelity or raise the cap (see `NOTES.md`).
- Upstream reclustering drops `reduction_dimension` and `threshold` on recursive calls; kept for fidelity and documented.

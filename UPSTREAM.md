# Upstream provenance

`src/raptor/` is a verbatim copy of the RAPTOR package from

- Repository: https://github.com/parthsarthi03/raptor (`git@github.com:parthsarthi03/raptor.git`)
- Commit: `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767`
- License: MIT (see `LICENSE-RAPTOR.txt`); upstream README in `docs/RAPTOR_UPSTREAM_README.md`.

`scripts/check_environment.py` reports whether `src/raptor` still equals the original
checkout when one is present at `../raptor`.

## Rules

- Do not edit `src/raptor/`. Extend it from `src/adaptive_raptor/` (subclass or wrap) so
  `diff -r` against upstream stays empty.
- If an edit to upstream code becomes unavoidable, record it in the first table below.

## Source modifications to `src/raptor/`

| Date | File | Change | Reason |
|---|---|---|---|
| — | — | none | — |

## Behavioral differences introduced by `adaptive_raptor` (upstream files untouched)

| Where | Upstream behavior | Our behavior | Reason |
|---|---|---|---|
| `indexing/builder.py` `build_from_chunks` | `build_from_text` splits text with `split_text` | Takes chunks from any chunker; with the fixed chunker the leaves are byte-identical (tested) | Pluggable chunking (PIC) |
| same | Leaf nodes in thread-completion order | Re-sorted by index | Deterministic clustering input |
| same, `seeded_umap` | `umap.UMAP(...)` without `random_state`: parallel and unseeded, so trees differ between processes (measured: 3 of 4 runs differed) | `random_state=seed` passed during the build when `raptor.deterministic_umap: true` (default); UMAP then runs single-threaded | Reproducible indexes |
| same | One builder could be reused; `construct_tree` mutates `builder.num_layers` | Fresh builder per document | Avoid state leaking between documents |
| `models/summarization.py` | `GPT3TurboSummarizationModel` hard-codes OpenAI and `gpt-3.5-turbo` | Same prompt text verbatim; model/endpoint from config (DeepSeek), `temperature=0`, disk cache | Configurable provider, reproducibility |
| `retrieval.py` | `TreeRetriever.retrieve` with `top_k` cut-off | Strategies run `retrieve_information_collapse_tree` on a layer-restricted view, `top_k` = all nodes (budget-limited); `raptor_original` calls `retrieve` unchanged | Strategy × budget catalog |
| `models/embedding.py` | Query re-embedded on every retrieval | Memoized embedding wrapper | Embed each query once |

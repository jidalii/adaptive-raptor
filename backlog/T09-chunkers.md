# T09 · PIC-capped and chunking controls

Tier: 1 · Owner: _ · Depends: T02, T03 (summarizer). Can run in parallel with T05–T08.

## What

Implement the chunkers in `chunking/__init__.py`: `PICChunker`, `PermutedBoundaryChunker` and `ParagraphChunker` are stubs there; remove `Stub` from each class when it's done (proposal §4.1–4.3). Every chunker keeps offsets and drops no sentence.

- **PIC-capped** (`PICChunker`):
  - document summary (same summarizer as RAPTOR, never sees questions, cached);
  - sentence–summary cosine and a per-document mean threshold τ;
  - same-side runs, split at `max_tokens`, optional merge below `min_tokens`;
  - `summary_source: raptor_root` as the optional zero-cost variant.
- **Permuted boundaries** (`PermutedBoundaryChunker`): PIC-capped's per-document chunk lengths (in sentences), placed at random sentence positions with `permutation_seed`.
- **Paragraph** (`ParagraphChunker`): `Document.paragraphs`, split at `max_tokens`.
- **F-μ**: just the `fixed` method with `chunk_size` = PIC-capped's mean length on dev (record the value).
- One shared sentence splitter for all methods (record which one).

## Done when

- For each method on 5 pilot documents: concatenating the chunks reproduces the document's sentences in order, and no chunk exceeds `max_tokens`.
- The permuted control has exactly PIC-capped's length multiset per document.
- Chunk-length distributions per method are written to `NOTES.md`, along with the F-μ `chunk_size` (update `configs/experiments/h2.yaml`).

## Description

The controls are what make H2 a real claim. F-μ removes average size, permuted removes the length distribution, and paragraph tests whether author structure already suffices.

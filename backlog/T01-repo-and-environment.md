# T01 · Repo, environment and config system

Tier: 1 · Owner: Jida · Depends: none

## What

- Upstream RAPTOR copied verbatim into `src/raptor/`, with provenance and every behavioral difference listed in `UPSTREAM.md`.
- `uv` project: `pyproject.toml`, `uv.lock` (macOS arm64 + Linux CPU torch), `.python-version` (3.11), `.gitignore`, `.env.example`.
- `scripts/setup.sh` (one-command, idempotent setup + offline smoke run) and `scripts/check_environment.py`.
- YAML config system: `config/schema.py`, `config/loader.py` (with `--set` overrides), `config/validation.py`.
- CLI `python -m adaptive_raptor.run --config ... [--validate-only]`. Exit codes: 0 ok, 1 failed, 2 invalid config or environment not ready, 3 not implemented.
- Consolidated layout (see `docs/IMPLEMENTATION.md`): research work lives in `chunking/__init__.py`, `optimizer/__init__.py`, `profiling.py` and `evaluation/research.py`, marked **IMPLEMENT**.
- Example configs: `smoke_mock`, `baseline`, `pilot_catalog`, `h1`, `h2`, `ablation`, `oracle_interaction`.

## Done when

- `./scripts/setup.sh --with-tests` passes on a fresh clone.
- All example configs pass `--validate-only`.
- `diff -r ../raptor/raptor src/raptor` is empty.

## Description

1. **Upstream stays untouched.** New behavior goes in `src/adaptive_raptor/` and wraps or subclasses upstream.
2. **No parameter lives in code.** Everything an experiment varies is in YAML, and the resolved config is saved with every run.
3. **Validation is cheap.** Importing the CLI or config never loads torch or `raptor`.
4. **Placeholders never report success.** An unimplemented component stops the run before any index build or API call (exit code 3).

## Notes

- Verified Nov 6 (current layout): ruff clean; 108 offline tests and 3 integration tests pass; 7/7 configs valid; `src/raptor` identical to upstream `7da1d48`; original `raptor/` repo unchanged.
- `uv` 0.13.4 via Homebrew. Upstream's 2023 pins don't install together; ranges are used and `uv.lock` records torch 2.15, transformers 5.22, sentence-transformers 6.2, openai 3.46, numpy 1.26.
- Git initialized (branch `main`), **no commits yet**.
- Nov 6: `.gitignore` fixed. The unanchored `data/` and `models/` patterns were silently excluding the source packages `src/adaptive_raptor/data/` and `src/adaptive_raptor/models/`, and `.cache/` (API responses) was not ignored. Run-time directories are now root-anchored (`/data/`, `/.cache/`, …); `results/` and `NOTES.md` are meant to be committed.

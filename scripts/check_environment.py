"""Check that the development environment is ready. Never calls APIs or downloads models.

uv run python scripts/check_environment.py          # exit 1 if a required check fails
uv run python scripts/check_environment.py --quiet  # only problems
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "ok", "warn", "FAIL"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


def check_python() -> None:
    want = (ROOT / ".python-version").read_text().strip()
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    record(OK if have == want else FAIL, "python", f"{platform.python_version()} (want {want})")
    record(OK, "platform", f"{platform.system()} {platform.machine()}")


def check_imports() -> None:
    for mod in (
        "adaptive_raptor",
        "adaptive_raptor.run",
        "numpy",
        "pydantic",
        "yaml",
        "tiktoken",
        "openai",
        "sklearn",
        "umap",
        "faiss",
        "torch",
        "sentence_transformers",
        "raptor",
    ):
        try:
            m = importlib.import_module(mod)
            record(OK, f"import {mod}", getattr(m, "__version__", ""))
        except Exception as exc:  # noqa: BLE001 - report every failure
            record(FAIL, f"import {mod}", f"{type(exc).__name__}: {exc}")


def check_upstream_copy() -> None:
    upstream = ROOT.parent / "raptor" / "raptor"
    if not upstream.is_dir():
        record(OK, "src/raptor provenance", "see UPSTREAM.md (original repo not alongside)")
        return
    import filecmp

    cmp = filecmp.dircmp(upstream, ROOT / "src" / "raptor", ignore=["__pycache__"])
    same = not (cmp.diff_files or cmp.left_only or cmp.right_only)
    record(
        OK if same else WARN,
        "src/raptor == upstream",
        "identical" if same else f"differs: {cmp.diff_files + cmp.left_only + cmp.right_only}",
    )


def check_tokenizer() -> None:
    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")  # ~1.7 MB, cached after first use
        record(OK, "tiktoken cl100k_base")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "tiktoken cl100k_base", f"{exc} (needs network once)")


def check_configs() -> None:
    from adaptive_raptor.config import load_config, validate_config

    for path in sorted((ROOT / "configs" / "experiments").glob("*.yaml")):
        try:
            report = validate_config(load_config(path))
            record(OK if report.ok else FAIL, f"config {path.name}", "; ".join(report.errors))
        except Exception as exc:  # noqa: BLE001
            record(FAIL, f"config {path.name}", str(exc).splitlines()[0])


def check_dirs() -> None:
    for d in ("data", "indexes", "outputs", ".cache"):
        p = ROOT / d
        ok = p.is_dir() and os.access(p, os.W_OK)
        record(
            OK if ok else WARN, f"dir {d}/", "writable" if ok else "missing (setup.sh creates it)"
        )


def check_credentials() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if not (ROOT / ".env").is_file():
        record(WARN, ".env", "missing; copy .env.example to .env")
    for env in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        record(
            OK if os.environ.get(env) else WARN,
            env,
            "set" if os.environ.get(env) else "not set (needed only for real runs)",
        )


def check_dataset() -> None:
    from adaptive_raptor.data.download import verify_qasper
    from adaptive_raptor.paths import data_root

    root = data_root()
    if not (root / "processed" / "qasper").exists():
        record(
            WARN,
            "QASPER",
            f"not downloaded under {root} (./scripts/download_datasets.sh --dataset qasper)",
        )
        return
    problems = verify_qasper()
    record(
        OK if not problems else FAIL,
        "QASPER v0.3",
        "verified" if not problems else "; ".join(problems),
    )


def check_sbert_cache() -> None:
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(
            "sentence-transformers/multi-qa-mpnet-base-cos-v1", "config.json"
        )
        cached = isinstance(hit, str)
    except Exception:  # noqa: BLE001
        cached = False
    record(
        OK if cached else WARN,
        "SBERT model cache",
        "cached" if cached else "not downloaded yet (~420 MB on the first real index build)",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    for check in (
        check_python,
        check_imports,
        check_upstream_copy,
        check_tokenizer,
        check_configs,
        check_dirs,
        check_credentials,
        check_dataset,
        check_sbert_cache,
    ):
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            record(FAIL, check.__name__, f"{type(exc).__name__}: {exc}")
    width = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        if args.quiet and status == OK:
            continue
        print(f"[{status:>4}] {name:<{width}}  {detail}")
    failed = [r for r in results if r[0] == FAIL]
    warned = [r for r in results if r[0] == WARN]
    print(
        f"\n{len(results) - len(failed) - len(warned)} ok, {len(warned)} warnings, "
        f"{len(failed)} failures"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

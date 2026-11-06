"""Run-directory creation and reproducibility metadata."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from adaptive_raptor import __version__
from adaptive_raptor.config.loader import dump_yaml
from adaptive_raptor.config.schema import ExperimentConfig
from adaptive_raptor.indexing.store import UPSTREAM_RAPTOR_COMMIT
from adaptive_raptor.paths import REPO_ROOT

TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "transformers",
    "sentence-transformers",
    "umap-learn",
    "scikit-learn",
    "faiss-cpu",
    "tiktoken",
    "openai",
    "pydantic",
)


def make_run_dir(config: ExperimentConfig, now: datetime | None = None) -> Path:
    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    base = config.experiment.output_dir / config.experiment.name / stamp
    run_dir = base
    i = 1
    while run_dir.exists():  # never overwrite an earlier run
        run_dir = base.with_name(f"{stamp}_{i}")
        i += 1
    run_dir.mkdir(parents=True)
    return run_dir


def save_resolved_config(config: ExperimentConfig, run_dir: Path, source: Path | None) -> Path:
    path = run_dir / "resolved_config.yaml"
    header = f"# Resolved configuration (defaults filled in). Source: {source}\n"
    path.write_text(header + dump_yaml(config))
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "note": "not a git repository or no commits"}
    status = git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_metadata(config: ExperimentConfig, source: Path | None, overrides: list[str]) -> dict:
    return {
        "adaptive_raptor_version": __version__,
        "experiment": config.experiment.name,
        "experiment_type": config.experiment.type.value,
        "seed": config.experiment.seed,
        "raptor_seeds": config.raptor.seeds,
        "run_label": config.experiment.run_label,
        "dataset": {
            "name": config.dataset.name,
            "version": config.dataset.version,
            "split": config.dataset.split,
        },
        "generator": config.generation.model_dump(mode="json"),
        "raptor": config.raptor.model_dump(mode="json"),
        "upstream_raptor_commit": UPSTREAM_RAPTOR_COMMIT,
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "config_source": str(source) if source else None,
        "overrides": overrides,
        "started_at": datetime.now(UTC).isoformat(),
    }

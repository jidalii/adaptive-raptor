"""Orchestrates one experiment run:

    validate -> preflight (credentials, data) -> run dir + resolved config + metadata
    -> seed -> dispatch -> status.json

Nothing is written and nothing is called before validation and preflight pass.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from adaptive_raptor.config.loader import ConfigError
from adaptive_raptor.config.schema import ExperimentConfig
from adaptive_raptor.config.validation import ValidationReport, validate_config
from adaptive_raptor.experiments import ExperimentResult, RunStatus, get_experiment_class
from adaptive_raptor.experiments.outputs import (
    make_run_dir,
    run_metadata,
    save_resolved_config,
    write_json,
)

log = logging.getLogger(__name__)


class PreflightError(ConfigError):
    """The config is valid but the environment is not ready (keys, data)."""


def check(config: ExperimentConfig) -> ValidationReport:
    """Semantic validation; raises ConfigError on errors. No side effects."""
    report = validate_config(config)
    if not report.ok:
        raise ConfigError(
            f"configuration is invalid for experiment type '{config.experiment.type}'",
            report.errors,
        )
    return report


def preflight(config: ExperimentConfig) -> None:
    """Environment checks for runners that do real work. No network, no writes."""
    from adaptive_raptor.models.credentials import credential_status, required_credentials

    problems = []
    purposes = required_credentials(config)
    for env, ok in credential_status(config).items():
        if not ok:
            problems.append(f"{env} is not set (needed for {purposes[env]}); add it to .env")
    if config.dataset.name == "qasper":
        from adaptive_raptor.data.qasper import processed_split_paths
        from adaptive_raptor.paths import processed_dir

        pdir = processed_dir("qasper", config.dataset.version, config.dataset.path)
        if not all(p.is_file() for p in processed_split_paths(pdir, config.dataset.split)):
            problems.append(
                f"QASPER {config.dataset.version} is not prepared under {pdir}; run "
                "./scripts/download_datasets.sh --dataset qasper"
            )
    if problems:
        raise PreflightError("environment is not ready for this run", problems)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_experiment(
    config: ExperimentConfig,
    *,
    source: Path | None = None,
    overrides: list[str] | None = None,
) -> tuple[ExperimentResult, Path]:
    overrides = overrides or []
    report = check(config)
    for w in report.warnings:
        log.warning("config warning: %s", w)
    cls = get_experiment_class(config.experiment.type)
    if cls.requires_credentials:
        preflight(config)

    run_dir = make_run_dir(config)
    save_resolved_config(config, run_dir, source)
    meta = run_metadata(config, source, overrides)
    meta["warnings"] = report.warnings
    write_json(run_dir / "run_metadata.json", meta)

    seed_everything(config.experiment.seed)
    experiment = cls(config, run_dir)
    t0 = time.perf_counter()
    try:
        result = experiment.run()
    except NotImplementedError as exc:
        result = ExperimentResult(
            status=RunStatus.NOT_IMPLEMENTED,
            message=str(exc) or "not implemented",
            planned_steps=experiment.planned_steps(),
        )
    except Exception as exc:  # record failures instead of losing them
        log.exception("experiment failed")
        result = ExperimentResult(status=RunStatus.FAILED, message=f"{type(exc).__name__}: {exc}")

    status = asdict(result)
    status["duration_s"] = round(time.perf_counter() - t0, 3)
    status["finished_at"] = datetime.now(UTC).isoformat()
    write_json(run_dir / "status.json", status)
    return result, run_dir

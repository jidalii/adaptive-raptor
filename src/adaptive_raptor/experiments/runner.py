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
    if config.dataset.name in ("qasper", "narrativeqa"):
        from adaptive_raptor.data.qasper import processed_split_paths
        from adaptive_raptor.paths import processed_dir

        name = config.dataset.name
        pdir = processed_dir(name, config.dataset.version, config.dataset.path)
        if not all(p.is_file() for p in processed_split_paths(pdir, config.dataset.split)):
            problems.append(
                f"{name} {config.dataset.version} is not prepared under {pdir}; run "
                f"./scripts/download_datasets.sh --dataset {name}"
            )
    if problems:
        raise PreflightError("environment is not ready for this run", problems)


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%SZ"  # UTC, like run ids


class UTCFormatter(logging.Formatter):
    converter = time.gmtime


def log_dir_for(config: ExperimentConfig) -> Path:
    return config.runtime.log_dir or config.experiment.output_dir.parent / "logs"


def attach_log_file(path: Path) -> logging.Handler:
    """Copy everything the console shows into ``path`` (same format, with timestamps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(UTCFormatter(LOG_FORMAT, LOG_DATEFMT))
    logging.getLogger().addHandler(handler)
    return handler


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
    cls = get_experiment_class(config.experiment.type)
    if cls.requires_credentials:
        preflight(config)

    run_dir = make_run_dir(config)
    log_file = log_dir_for(config) / f"{config.experiment.name}_{run_dir.name}.log"
    handler = attach_log_file(log_file)
    try:
        return _run_in_dir(config, cls, run_dir, log_file, report, source, overrides)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _run_in_dir(
    config: ExperimentConfig,
    cls: type,
    run_dir: Path,
    log_file: Path,
    report: ValidationReport,
    source: Path | None,
    overrides: list[str],
) -> tuple[ExperimentResult, Path]:
    log.info("run %s started: config=%s log=%s", run_dir, source, log_file)
    for w in report.warnings:
        log.warning("config warning: %s", w)
    save_resolved_config(config, run_dir, source)
    meta = run_metadata(config, source, overrides)
    meta["warnings"] = report.warnings
    meta["log_file"] = str(log_file)
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
    log.info(
        "run finished: status=%s (%s) in %.1fs; outputs: %s",
        result.status.value,
        result.message,
        status["duration_s"],
        run_dir,
    )
    return result, run_dir

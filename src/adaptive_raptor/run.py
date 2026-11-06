"""Unified experiment CLI.

Examples::

    uv run python -m adaptive_raptor.run --config configs/experiments/h1.yaml
    uv run python -m adaptive_raptor.run --config configs/experiments/h1.yaml --validate-only
    uv run python -m adaptive_raptor.run --config configs/experiments/h2.yaml \
        --set dataset.max_documents=20 --set runtime.concurrency=2

Exit codes: 0 succeeded / valid, 1 failed, 2 invalid config or usage, 3 not implemented.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from adaptive_raptor.config.loader import ConfigError, dump_yaml, load_config
from adaptive_raptor.experiments import RunStatus
from adaptive_raptor.experiments.runner import check, run_experiment

EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_NOT_IMPLEMENTED = 0, 1, 2, 3


def configure_logging(level: str) -> None:
    """Our logs at ``level``; upstream RAPTOR (root logger, very verbose) at WARNING."""
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.WARNING)
    logging.getLogger().setLevel(logging.DEBUG if level == "DEBUG" else logging.WARNING)
    logging.getLogger("adaptive_raptor").setLevel(level)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adaptive_raptor.run", description="Run an Adaptive-RAPTOR experiment from YAML."
    )
    p.add_argument("--config", "-c", required=True, type=Path, help="experiment YAML file")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override a config value (repeatable), e.g. --set dataset.max_documents=20",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the config and exit; builds nothing, calls no APIs, downloads nothing",
    )
    p.add_argument("--print-config", action="store_true", help="print the resolved config as YAML")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging("INFO")

    try:
        config = load_config(args.config, args.overrides)
        if args.validate_only:
            report = check(config)
            if args.print_config:
                print(dump_yaml(config), end="")
            for w in report.warnings:
                print(f"warning: {w}", file=sys.stderr)
            print(f"OK: {args.config} is a valid '{config.experiment.type}' config")
            return EXIT_OK
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    configure_logging(config.runtime.log_level)
    load_dotenv()  # API keys are read only when an actual run starts
    if args.print_config:
        print(dump_yaml(config), end="")
    try:
        result, run_dir = run_experiment(config, source=args.config, overrides=args.overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    print(f"status: {result.status.value}  ({result.message})")
    for row in result.metrics.get("summary", []):
        print(
            f"  {row['index_variant']:>12} {row['config_id']:>18}  n={row['n']:<4} "
            f"F1={row['answer_f1']}  ctx_tokens={row['context_tokens']}  errors={row['errors']}"
        )
    for step in result.planned_steps:
        print(f"  planned: {step}")
    print(f"outputs: {run_dir}")
    return {
        RunStatus.SUCCEEDED: EXIT_OK,
        RunStatus.FAILED: EXIT_FAILED,
        RunStatus.NOT_IMPLEMENTED: EXIT_NOT_IMPLEMENTED,
    }[result.status]


if __name__ == "__main__":
    sys.exit(main())

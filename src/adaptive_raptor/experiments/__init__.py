"""Experiment contracts, baseline runner, research placeholders, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from adaptive_raptor.config.schema import ExperimentConfig, ExperimentType
from adaptive_raptor.evaluation.traces import (
    read_traces,
    summarize,
    write_official_predictions,
    write_rows,
)
from adaptive_raptor.pipeline import QAPipeline
from adaptive_raptor.profiling import LabelStore, from_traces


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass
class ExperimentResult:
    status: RunStatus
    message: str = ""
    planned_steps: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class Experiment(ABC):
    """One experiment type, given a validated config and a fresh run directory."""

    experiment_type: ClassVar[ExperimentType]
    requires_credentials: ClassVar[bool] = False

    def __init__(self, config: ExperimentConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir

    @abstractmethod
    def planned_steps(self) -> list[str]:
        """Human-readable pipeline steps."""

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Execute the experiment."""


class PlaceholderExperiment(Experiment):
    def run(self) -> ExperimentResult:
        return ExperimentResult(
            status=RunStatus.NOT_IMPLEMENTED,
            message=f"runner for '{self.experiment_type.value}' is not implemented yet",
            planned_steps=self.planned_steps(),
        )


class RaptorBaselineExperiment(Experiment):
    """Run the reusable upstream RAPTOR baseline and emit profiling records."""

    experiment_type = ExperimentType.RAPTOR_BASELINE
    requires_credentials = True

    def planned_steps(self) -> list[str]:
        return [
            "load dataset split",
            "chunk each document with the configured variant",
            "build or reuse cached RAPTOR indexes",
            "run every catalog configuration and optional raptor_original",
            "generate answers and score QASPER answer F1",
            "write traces, summaries, predictions, and label records",
        ]

    def run(self) -> ExperimentResult:
        report = QAPipeline(self.config, self.run_dir).run()
        traces = read_traces(self.run_dir / "traces.jsonl")
        generation = self.config.generation
        rows = summarize(traces, generation.price_input_per_mtok, generation.price_output_per_mtok)
        artifacts = {
            path.name: str(path)
            for path in write_rows(
                rows, self.run_dir, "summary", self.config.evaluation.output_formats
            )
        }
        write_official_predictions(traces, self.run_dir)
        LabelStore(self.run_dir / "labels.jsonl").append(from_traces(traces))
        metrics = {
            "documents": report.n_documents,
            "questions": report.n_questions,
            "traces": report.n_traces,
            "trace_errors": report.n_errors,
            "indexes_built": report.indexes_built,
            "indexes_reused": report.indexes_reused,
            "index_errors": report.index_errors,
            "summarizer_usage": report.summarizer_usage,
            "dataset": report.dataset_manifest,
            "summary": rows,
        }
        failed = report.n_errors + report.index_errors
        if failed:
            return ExperimentResult(
                RunStatus.FAILED,
                f"{failed} errors (see traces/indexes)",
                artifacts=artifacts,
                metrics=metrics,
            )
        return ExperimentResult(
            RunStatus.SUCCEEDED,
            f"{report.n_traces} traces",
            artifacts=artifacts,
            metrics=metrics,
        )


class H1OptimizerExperiment(PlaceholderExperiment):
    experiment_type = ExperimentType.H1_OPTIMIZER

    def planned_steps(self) -> list[str]:
        return [
            "load frozen router checkpoint and theta",
            "extract features and predict acceptability per catalog config",
            "select the cheapest acceptable config or fallback",
            "execute the shared QA pipeline",
            "match the fixed baseline budget and bootstrap delta F1",
        ]


class H2ChunkingExperiment(PlaceholderExperiment):
    experiment_type = ExperimentType.H2_CHUNKING

    def planned_steps(self) -> list[str]:
        return [
            "chunk documents with each configured variant",
            "build RAPTOR indexes per variant, document, and seed",
            "profile the full retrieval catalog on cross-fitted runs",
            "compute matched oracle and deployable fixed frontiers",
            "bootstrap differences and test equivalence",
        ]


class StrategyBudgetAblationExperiment(PlaceholderExperiment):
    experiment_type = ExperimentType.STRATEGY_BUDGET_ABLATION

    def planned_steps(self) -> list[str]:
        return [
            "restrict the catalog per ablation cell",
            "train and calibrate each restricted router",
            "evaluate cells at matched average context tokens",
            "report delta F1 with bootstrap intervals",
        ]


class OracleInteractionExperiment(PlaceholderExperiment):
    experiment_type = ExperimentType.ORACLE_INTERACTION

    def planned_steps(self) -> list[str]:
        return [
            "load label records for each chunking variant",
            "derive acceptability labels",
            "compute oracle quality-cost frontiers",
            "report strategy and budget interactions",
        ]


_REGISTRY: dict[ExperimentType, type[Experiment]] = {}


def register(cls: type[Experiment]) -> type[Experiment]:
    """Register or replace an experiment runner."""
    _REGISTRY[cls.experiment_type] = cls
    return cls


for _runner in (
    RaptorBaselineExperiment,
    H1OptimizerExperiment,
    H2ChunkingExperiment,
    StrategyBudgetAblationExperiment,
    OracleInteractionExperiment,
):
    register(_runner)


def get_experiment_class(experiment_type: ExperimentType) -> type[Experiment]:
    try:
        return _REGISTRY[experiment_type]
    except KeyError as exc:
        raise KeyError(f"no runner registered for experiment type '{experiment_type}'") from exc


def registered_types() -> list[ExperimentType]:
    return list(_REGISTRY)


__all__ = [
    "Experiment",
    "ExperimentResult",
    "RaptorBaselineExperiment",
    "RunStatus",
    "get_experiment_class",
    "register",
    "registered_types",
]

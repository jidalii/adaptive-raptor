"""Retrieval-optimizer contracts and pipeline adapter.

This module is intentionally the single implementation surface for the H1 router.  It
defines the data passed between feature extraction, acceptability prediction, cost
estimation, plan selection, calibration, checkpointing, and online routing.  The research
algorithms are stubs; infrastructure can import these contracts without loading ML models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_raptor.config.schema import ConfigRef, ExperimentConfig, OptimizerSection
from adaptive_raptor.data import guard_split
from adaptive_raptor.data.records import Question
from adaptive_raptor.policy import ConfigPolicy, PlannedConfig
from adaptive_raptor.profiling import LabelRecord
from adaptive_raptor.stub import Stub, ensure_implemented


@dataclass(frozen=True)
class QueryContext:
    question_id: str
    doc_id: str
    text: str
    embedding: Any


@dataclass
class RoutingDecision:
    config: ConfigRef
    probabilities: dict[str, float] = field(default_factory=dict)
    theta: float | None = None
    used_fallback: bool = False


@dataclass
class RouterCheckpoint:
    config_hash: str
    catalog: list[str]
    theta: float
    fallback: str
    artifacts: dict[str, Any] = field(default_factory=dict)


class FeatureExtractor(Stub):
    def __init__(self, cfg: OptimizerSection) -> None:
        self.cfg = cfg

    def fit(self, train: Sequence[QueryContext]) -> FeatureExtractor:
        """Fit training-only PCA and any lexical feature state."""
        raise NotImplementedError

    def transform(self, queries: Sequence[QueryContext]):
        """Return frozen-embedding PCA plus lexical query features."""
        raise NotImplementedError


class AcceptabilityModel(Stub):
    def __init__(self, cfg: OptimizerSection, config_ids: Sequence[str]) -> None:
        self.cfg = cfg
        self.config_ids = list(config_ids)

    def fit(self, features, labels: dict[str, dict[str, bool]], groups) -> AcceptabilityModel:
        """Fit per-config models with document-grouped model selection."""
        raise NotImplementedError

    def predict_proba(self, features) -> list[dict[str, float]]:
        """Return one ``config_id -> P(acceptable)`` mapping per query."""
        raise NotImplementedError


class CostModel(Stub):
    def fit(self, dev_records: Sequence[LabelRecord]) -> CostModel:
        """Fit expected context-token cost from development records."""
        raise NotImplementedError

    def cost(self, config: ConfigRef) -> float:
        raise NotImplementedError


class PlanSelector(Stub):
    def __init__(
        self, catalog: Sequence[ConfigRef], cost_model: CostModel, fallback: ConfigRef
    ) -> None:
        self.catalog = list(catalog)
        self.cost_model = cost_model
        self.fallback = fallback

    def select(self, probabilities: Mapping[str, float], theta: float) -> RoutingDecision:
        """Select the cheapest acceptable config, or the frozen fallback."""
        raise NotImplementedError


def calibrate_theta(
    split: str,
    dev_probabilities: Sequence[dict[str, float]],
    selector: PlanSelector,
    target_mean_tokens: float,
    grid: Sequence[float] | None = None,
) -> float:
    """Calibrate on dev to the closest mean cost not above the target."""
    guard_split(split, "calibrate")
    raise NotImplementedError("theta calibration is research logic (backlog T11)")


def theta_sweep(
    split: str, dev_probabilities, selector: PlanSelector, dev_f1_lookup, grid
) -> list[tuple[float, float, float]]:
    """Return ``(theta, mean tokens, mean F1)`` points on dev."""
    guard_split(split, "calibrate")
    raise NotImplementedError


def train_router(config: ExperimentConfig, label_store_path: Path, split: str) -> RouterCheckpoint:
    """Train and calibrate a frozen router checkpoint; never use test data."""
    guard_split(split, "train")
    raise NotImplementedError("router training is research logic (backlog T11)")


def save_checkpoint(ckpt: RouterCheckpoint, path: Path) -> None:
    raise NotImplementedError


def load_checkpoint(path: Path) -> RouterCheckpoint:
    raise NotImplementedError


class Router(Stub):
    """Compose features, acceptability, cost, and selection with a frozen theta."""

    def __init__(self, cfg: OptimizerSection, catalog: Sequence[ConfigRef]) -> None:
        self.cfg = cfg
        self.catalog = list(catalog)

    @classmethod
    def from_checkpoint(cls, path: Path, cfg: OptimizerSection, catalog) -> Router:
        raise NotImplementedError

    def route(self, query: QueryContext) -> RoutingDecision:
        raise NotImplementedError


class RouterPolicy(ConfigPolicy):
    """Pipeline adapter that runs only the router's selected configuration."""

    def __init__(self, router: Router) -> None:
        ensure_implemented(router)
        self.router = router
        self.decisions: dict[str, RoutingDecision] = {}

    def configs_for(self, question: Question, query_embedding) -> list[PlannedConfig]:
        context = QueryContext(
            question.question_id, question.doc_id, question.text, query_embedding
        )
        decision = self.router.route(context)
        self.decisions[question.question_id] = decision
        return [decision.config]


__all__ = [
    "AcceptabilityModel",
    "CostModel",
    "FeatureExtractor",
    "PlanSelector",
    "QueryContext",
    "Router",
    "RouterCheckpoint",
    "RouterPolicy",
    "RoutingDecision",
    "calibrate_theta",
    "load_checkpoint",
    "save_checkpoint",
    "theta_sweep",
    "train_router",
]

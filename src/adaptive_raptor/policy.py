"""Which retrieval configurations to run for a question.

The pipeline asks a :class:`ConfigPolicy` for each question's configurations, so the
experiment runner never changes when a new decision rule arrives:

- :class:`CatalogPolicy`  runs every catalog config (baseline, profiling, label collection);
- ``optimizer.router.RouterPolicy`` runs the router's single choice (H1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from adaptive_raptor.config.schema import ConfigRef, RetrievalSection
from adaptive_raptor.data.records import Question

RaptorOriginal = Literal["raptor_original"]
PlannedConfig = ConfigRef | RaptorOriginal


class ConfigPolicy(ABC):
    @abstractmethod
    def configs_for(self, question: Question, query_embedding) -> list[PlannedConfig]:
        """Configurations to execute for this question, in order."""


class CatalogPolicy(ConfigPolicy):
    def __init__(self, retrieval: RetrievalSection) -> None:
        self.plan: list[PlannedConfig] = list(retrieval.catalog())
        if retrieval.include_raptor_original:
            self.plan.append("raptor_original")

    def configs_for(self, question: Question, query_embedding) -> list[PlannedConfig]:
        return list(self.plan)

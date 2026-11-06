"""Unimplemented statistical analysis contracts for the research experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from adaptive_raptor.profiling import LabelRecord


def paired_document_bootstrap(
    per_query_delta: Mapping[str, float],
    query_to_doc: Mapping[str, str],
    n_samples: int,
    confidence: float,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample documents with replacement and return ``(mean, lo, hi)``."""
    raise NotImplementedError


def within_equivalence_margin(ci: tuple[float, float], margin: float) -> bool:
    lo, hi = ci
    return -margin <= lo and hi <= margin


def fixed_config_frontier(records: Sequence[LabelRecord]) -> list[tuple[str, float, float]]:
    """Return the upper envelope of config id, mean tokens, and mean F1."""
    raise NotImplementedError


def oracle_frontier(
    records_a: Sequence[LabelRecord], records_b: Sequence[LabelRecord] | None = None
) -> list[tuple[float, float]]:
    """Choose on run A and, when supplied, cross-fit scores on run B."""
    raise NotImplementedError


def match_fixed_baseline(
    router_mean_tokens: float,
    dev_frontier: list[tuple[str, float, float]],
    rule: str = "conservative_at_least",
) -> str:
    """Select the dev-fixed comparator under the configured budget rule."""
    raise NotImplementedError


__all__ = [
    "fixed_config_frontier",
    "match_fixed_baseline",
    "oracle_frontier",
    "paired_document_bootstrap",
    "within_equivalence_margin",
]

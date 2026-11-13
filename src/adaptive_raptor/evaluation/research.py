"""Decision rules fixed before test data is seen (backlog T07, proposal 5.4 and 6.2).

All functions work on label records (one row per question x config) and never call an
API. Conventions:

- F1 is on the 0-1 scale; tokens are the context tokens actually used.
- Comparisons use the **common query set**: questions that have every compared config.
- Ties go to the cheaper config, then to the config id (deterministic).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from adaptive_raptor.profiling import LabelRecord, check_single_cell

ORIGINAL = "raptor_original"


# --------------------------------------------------------------------------- tables


def per_query_table(
    records: Sequence[LabelRecord], configs: Sequence[str] | None = None
) -> dict[str, dict[str, LabelRecord]]:
    """``question_id -> config_id -> record`` over the common query set.

    Records must come from one (index_variant, seed, run) cell. ``configs`` defaults to
    every config present except ``raptor_original``.
    """
    check_single_cell(records)
    if configs is None:
        configs = sorted({r.config_id for r in records} - {ORIGINAL})
    wanted = set(configs)
    table: dict[str, dict[str, LabelRecord]] = defaultdict(dict)
    for r in records:
        if r.config_id in wanted:
            table[r.question_id][r.config_id] = r
    return {q: row for q, row in sorted(table.items()) if len(row) == len(wanted)}


@dataclass(frozen=True)
class ConfigPoint:
    config_id: str
    mean_tokens: float
    mean_f1: float
    n: int


def fixed_config_points(
    records: Sequence[LabelRecord], configs: Sequence[str] | None = None
) -> list[ConfigPoint]:
    """Mean tokens and mean F1 of every fixed config, sorted by tokens."""
    table = per_query_table(records, configs)
    if not table:
        return []
    ids = sorted(next(iter(table.values())))
    points = [
        ConfigPoint(
            c,
            float(np.mean([row[c].tokens_used for row in table.values()])),
            float(np.mean([row[c].answer_f1 for row in table.values()])),
            len(table),
        )
        for c in ids
    ]
    return sorted(points, key=lambda p: (p.mean_tokens, -p.mean_f1, p.config_id))


def upper_envelope(points: Sequence[ConfigPoint]) -> list[ConfigPoint]:
    """Pareto frontier: keep a point only if it beats every cheaper point's F1."""
    frontier: list[ConfigPoint] = []
    for p in sorted(points, key=lambda p: (p.mean_tokens, -p.mean_f1, p.config_id)):
        if not frontier or p.mean_f1 > frontier[-1].mean_f1:
            frontier.append(p)
    return frontier


def fixed_config_frontier(records: Sequence[LabelRecord]) -> list[tuple[str, float, float]]:
    """Return the upper envelope of config id, mean tokens, and mean F1."""
    return [
        (p.config_id, p.mean_tokens, p.mean_f1)
        for p in upper_envelope(fixed_config_points(records))
    ]


# --------------------------------------------------------------------------- oracle


@dataclass
class OraclePoint:
    lam: float  # F1 traded per context token
    mean_tokens: float
    mean_f1: float  # scored on run B when cross-fitted, else on run A
    choice: dict[str, str] = field(repr=False, default_factory=dict)  # question -> config
    per_query_f1: dict[str, float] = field(repr=False, default_factory=dict)


DEFAULT_LAMBDAS = np.concatenate([[0.0], np.geomspace(1e-6, 1e-1, 300)])


def _choose(row: Mapping[str, LabelRecord], lam: float) -> str:
    # argmax F1 - lam * tokens; ties -> fewer tokens -> config id
    return min(
        row, key=lambda c: (-(row[c].answer_f1 - lam * row[c].tokens_used), row[c].tokens_used, c)
    )


def oracle_points(
    records_a: Sequence[LabelRecord],
    records_b: Sequence[LabelRecord] | None = None,
    configs: Sequence[str] | None = None,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> list[OraclePoint]:
    """Per-query oracle along a Lagrangian sweep of the quality-cost trade-off.

    For each lambda, every query takes ``argmax_c F1_A(q, c) - lambda * tokens(q, c)``.
    With ``records_b`` (cross-fitting, proposal 6.2) the choice is made on run A and
    scored on run B, removing the winner's-curse bias of max-over-noisy-scores. Only
    queries present in both runs are used. Duplicate (tokens, F1) points are dropped.
    """
    table_a = per_query_table(records_a, configs)
    table_b = per_query_table(records_b, configs) if records_b is not None else table_a
    queries = [q for q in table_a if q in table_b]
    if not queries:
        return []
    out: list[OraclePoint] = []
    seen: set[tuple[float, float]] = set()
    for lam in lambdas:
        choice = {q: _choose(table_a[q], float(lam)) for q in queries}
        f1 = {q: table_b[q][c].answer_f1 for q, c in choice.items()}
        tokens = float(np.mean([table_b[q][c].tokens_used for q, c in choice.items()]))
        point = OraclePoint(float(lam), tokens, float(np.mean(list(f1.values()))), choice, f1)
        sig = (round(point.mean_tokens, 6), round(point.mean_f1, 9))
        if sig not in seen:
            seen.add(sig)
            out.append(point)
    return sorted(out, key=lambda p: (p.mean_tokens, p.lam))


def oracle_frontier(
    records_a: Sequence[LabelRecord], records_b: Sequence[LabelRecord] | None = None
) -> list[tuple[float, float]]:
    """Choose on run A and, when supplied, cross-fit scores on run B."""
    return [(p.mean_tokens, p.mean_f1) for p in oracle_points(records_a, records_b)]


def oracle_at_budget(
    records_a: Sequence[LabelRecord],
    records_b: Sequence[LabelRecord] | None,
    budget: float,
    configs: Sequence[str] | None = None,
) -> OraclePoint:
    """The swept oracle point with the highest F1 whose mean tokens are <= ``budget``.

    Used for H2 (oracle F1 at the pre-set matched budget) and the E0 oracle gap.
    Raises if even the cheapest choice exceeds the budget.
    """
    feasible = [
        p for p in oracle_points(records_a, records_b, configs) if p.mean_tokens <= budget + 1e-9
    ]
    if not feasible:
        raise ValueError(f"no oracle point within budget {budget}")
    return max(feasible, key=lambda p: (p.mean_f1, -p.mean_tokens))


# --------------------------------------------------------------------------- matching


@dataclass(frozen=True)
class BaselineMatch:
    """The fixed comparator for a router (proposal 5.4).

    ``conservative_at_least``: one config, ``weights == {config: 1.0}``.
    ``interpolate``: the two frontier neighbours of the router's tokens, with linear
    weights; a query's baseline score is the weighted sum of its two F1 values.
    """

    rule: str
    weights: dict[str, float]
    mean_tokens: float
    mean_f1: float
    satisfied: bool  # False only if no fixed config uses at least the router's tokens

    @property
    def config_id(self) -> str:
        return max(self.weights, key=lambda c: (self.weights[c], c))


def match_baseline(
    router_mean_tokens: float,
    dev_frontier: Sequence[tuple[str, float, float]],
    rule: str = "conservative_at_least",
    max_fixed_tokens: float | None = None,
) -> BaselineMatch:
    """Select the dev-fixed comparator under the configured budget rule.

    ``dev_frontier`` is :func:`fixed_config_frontier` on **dev** records.
    ``conservative_at_least``: the baseline gets a budget of at least the router's dev
    mean tokens and uses the best fixed config within it, i.e. the cheapest frontier
    config with tokens >= the router's. If the frontier ends below the router's tokens
    (more context stopped helping), its last point is the best fixed config within any
    larger budget, so it is the comparator. A dominated config that merely uses more
    tokens would be a *weaker* baseline, the opposite of conservative.

    ``satisfied`` is False only when no fixed config reaches the router's tokens
    (``max_fixed_tokens``: the largest dev mean tokens of any fixed config; defaults to
    the frontier's last point), i.e. the router uses more context than any baseline can.
    """
    if not dev_frontier:
        raise ValueError("empty dev frontier")
    pts = sorted(dev_frontier, key=lambda p: p[1])
    reach = pts[-1][1] if max_fixed_tokens is None else max(max_fixed_tokens, pts[-1][1])
    available = router_mean_tokens <= reach + 1e-9
    if rule == "conservative_at_least":
        for cid, tok, f1 in pts:
            if tok >= router_mean_tokens - 1e-9:
                return BaselineMatch(rule, {cid: 1.0}, tok, f1, True)
        cid, tok, f1 = pts[-1]
        return BaselineMatch(rule, {cid: 1.0}, tok, f1, available)
    if rule == "interpolate":
        if router_mean_tokens <= pts[0][1]:
            cid, tok, f1 = pts[0]
            return BaselineMatch(rule, {cid: 1.0}, tok, f1, True)
        if router_mean_tokens >= pts[-1][1]:
            cid, tok, f1 = pts[-1]  # flat beyond the frontier's end
            return BaselineMatch(rule, {cid: 1.0}, tok, f1, available)
        for (c0, t0, f0), (c1, t1, f1) in zip(pts, pts[1:], strict=False):
            if t0 <= router_mean_tokens <= t1:
                w = (router_mean_tokens - t0) / (t1 - t0)
                weights = {c0: 1 - w, c1: w} if c0 != c1 else {c0: 1.0}
                return BaselineMatch(rule, weights, router_mean_tokens, f0 + w * (f1 - f0), True)
    raise ValueError(f"unknown budget-matching rule '{rule}'")


def match_fixed_baseline(
    router_mean_tokens: float,
    dev_frontier: list[tuple[str, float, float]],
    rule: str = "conservative_at_least",
) -> str:
    """Config id of the comparator (the larger-weight neighbour when interpolating)."""
    return match_baseline(router_mean_tokens, dev_frontier, rule).config_id


def baseline_scores(match: BaselineMatch, records: Sequence[LabelRecord]) -> dict[str, float]:
    """Per-query F1 of the matched baseline on a split (e.g. test), for paired deltas."""
    table = per_query_table(records, list(match.weights))
    return {
        q: float(sum(w * row[c].answer_f1 for c, w in match.weights.items()))
        for q, row in table.items()
    }


# --------------------------------------------------------------------------- replay scoring


def execution_tokens(r: LabelRecord) -> int:
    """Generation input (prompt incl. retrieved context) + output tokens of one call.

    Falls back to context tokens for label tables written before generation tokens were
    recorded (input_tokens == 0).
    """
    return r.input_tokens + r.output_tokens if r.input_tokens else r.tokens_used


@dataclass
class SystemScore:
    """Per-query replay of one system (router or fixed comparator) on a label table.

    Costs are kept separate: retrieved context tokens, generation input/output tokens
    (input includes the context), profiler input+output tokens, and their total
    ``generation input + generation output + profiler`` (the matching cost). USD uses the
    given per-1M-token prices for generation and profiler alike (None without prices).
    """

    name: str
    per_query_f1: dict[str, float]
    context_tokens: dict[str, float]
    gen_input: dict[str, float]
    gen_output: dict[str, float]
    profiler_tokens: dict[str, float] = field(default_factory=dict)
    usd: dict[str, float] | None = None
    fallback: dict[str, bool] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.per_query_f1)

    def total(self) -> dict[str, float]:
        return {
            q: self.gen_input[q] + self.gen_output[q] + self.profiler_tokens.get(q, 0.0)
            for q in self.per_query_f1
        }

    def row(self) -> dict[str, float | str | int | None]:
        def m(d):
            return float(np.mean(list(d.values()))) if d else 0.0

        return {
            "system": self.name,
            "n": self.n,
            "answer_f1": round(m(self.per_query_f1), 4),
            "context_tokens": round(m(self.context_tokens), 1),
            "generation_input_tokens": round(m(self.gen_input), 1),
            "generation_output_tokens": round(m(self.gen_output), 1),
            "profiler_tokens": round(m(self.profiler_tokens), 1),
            "total_tokens": round(m(self.total()), 1),
            "usd_per_query": round(m(self.usd), 7) if self.usd is not None else None,
            "usd_total": round(sum(self.usd.values()), 5) if self.usd is not None else None,
            "fallback_rate": round(m({q: float(v) for q, v in self.fallback.items()}), 4)
            if self.fallback
            else None,
        }


def _usd(tokens_in: float, tokens_out: float, prices: tuple[float, float] | None):
    if prices is None:
        return None
    return tokens_in / 1e6 * prices[0] + tokens_out / 1e6 * prices[1]


def _gen_tokens(r: LabelRecord) -> tuple[float, float]:
    if r.input_tokens:
        return float(r.input_tokens), float(r.output_tokens)
    return float(r.tokens_used), 0.0  # pre-WP10 tables: context only


def score_fixed(
    name: str,
    weights: Mapping[str, float],
    table: Mapping[str, Mapping[str, LabelRecord]],
    prices: tuple[float, float] | None = None,
) -> SystemScore:
    """A fixed (or interpolated) comparator: weighted F1, tokens and cost per query."""
    f1, ctx, gin, gout, usd = {}, {}, {}, {}, {}
    for q, row in table.items():
        f1[q] = float(sum(w * row[c].answer_f1 for c, w in weights.items()))
        ctx[q] = float(sum(w * row[c].tokens_used for c, w in weights.items()))
        gin[q] = float(sum(w * _gen_tokens(row[c])[0] for c, w in weights.items()))
        gout[q] = float(sum(w * _gen_tokens(row[c])[1] for c, w in weights.items()))
        if prices is not None:
            usd[q] = _usd(gin[q], gout[q], prices)
    return SystemScore(name, f1, ctx, gin, gout, {}, usd if prices is not None else None)


def score_routed(
    name: str,
    choices: Mapping[str, str],
    table: Mapping[str, Mapping[str, LabelRecord]],
    profiler_tokens: Mapping[str, tuple[int, int]],
    fallback: Mapping[str, bool],
    prices: tuple[float, float] | None = None,
) -> SystemScore:
    """A router replayed from labels: the chosen config's recorded answer and tokens,
    plus the question's profiler (input, output) tokens."""
    f1, ctx, gin, gout, prof, usd = {}, {}, {}, {}, {}, {}
    for q, c in choices.items():
        r = table[q][c]
        pin, pout = profiler_tokens.get(q, (0, 0))
        f1[q], ctx[q], prof[q] = r.answer_f1, float(r.tokens_used), float(pin + pout)
        gin[q], gout[q] = _gen_tokens(r)
        if prices is not None:
            usd[q] = _usd(gin[q] + pin, gout[q] + pout, prices)
    return SystemScore(
        name,
        f1,
        ctx,
        gin,
        gout,
        prof,
        usd if prices is not None else None,
        {q: bool(fallback.get(q)) for q in choices},
    )


def fixed_cost_points(
    table: Mapping[str, Mapping[str, LabelRecord]], config_ids: Sequence[str]
) -> list[ConfigPoint]:
    """Every fixed config's mean total execution tokens and mean F1, sorted by cost."""
    points = [
        ConfigPoint(
            c,
            float(np.mean([execution_tokens(row[c]) for row in table.values()])),
            float(np.mean([row[c].answer_f1 for row in table.values()])),
            len(table),
        )
        for c in config_ids
    ]
    return sorted(points, key=lambda p: (p.mean_tokens, -p.mean_f1, p.config_id))


MATCHING_RULES = ("best_within_budget", "nearest_higher")


def select_comparator(
    router_cost: float, points: Sequence[ConfigPoint], rule: str = "best_within_budget"
) -> BaselineMatch:
    """Freeze the fixed comparator for a router from **dev** cost points.

    ``best_within_budget`` (primary): the highest-dev-F1 fixed config whose dev mean
    total cost is <= the router's dev mean total cost (best fixed config affordable at the
    router's budget). ``nearest_higher`` (sensitivity): the cheapest fixed config whose
    cost is >= the router's, regardless of F1. ``satisfied`` is False when no config
    qualifies; the closest one is returned so the report can flag it.
    """
    if not points:
        raise ValueError("no fixed configs")
    pts = sorted(points, key=lambda p: (p.mean_tokens, p.config_id))
    if rule == "best_within_budget":
        within = [p for p in pts if p.mean_tokens <= router_cost + 1e-9]
        if not within:
            p = pts[0]
            return BaselineMatch(rule, {p.config_id: 1.0}, p.mean_tokens, p.mean_f1, False)
        p = max(within, key=lambda p: (p.mean_f1, -p.mean_tokens))
        return BaselineMatch(rule, {p.config_id: 1.0}, p.mean_tokens, p.mean_f1, True)
    if rule == "nearest_higher":
        higher = [p for p in pts if p.mean_tokens >= router_cost - 1e-9]
        p, ok = (higher[0], True) if higher else (pts[-1], False)
        return BaselineMatch(rule, {p.config_id: 1.0}, p.mean_tokens, p.mean_f1, ok)
    raise ValueError(f"unknown matching rule '{rule}'")


# --------------------------------------------------------------------------- bootstrap


def paired_document_bootstrap(
    per_query_delta: Mapping[str, float],
    query_to_doc: Mapping[str, str],
    n_samples: int,
    confidence: float,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample documents with replacement and return ``(mean, lo, hi)``.

    The statistic is the mean per-query delta over all questions of the resampled
    documents (questions stay with their paper, so within-paper correlation is kept).
    ``lo``/``hi`` are percentile bounds; ``mean`` is the observed mean over queries.
    """
    if not per_query_delta:
        raise ValueError("no queries")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    missing = [q for q in per_query_delta if q not in query_to_doc]
    if missing:
        raise KeyError(f"{len(missing)} queries have no document, e.g. {missing[0]}")
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for q, d in per_query_delta.items():
        doc = query_to_doc[q]
        sums[doc] += float(d)
        counts[doc] += 1
    docs = sorted(sums)
    s = np.array([sums[d] for d in docs])
    n = np.array([counts[d] for d in docs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(docs), size=(n_samples, len(docs)))
    stats = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(stats, [alpha, 1 - alpha])
    return float(s.sum() / n.sum()), float(lo), float(hi)


def within_equivalence_margin(ci: tuple[float, float], margin: float) -> bool:
    lo, hi = ci
    return -margin <= lo and hi <= margin


def verdict(ci: tuple[float, float], margin: float | None) -> str:
    """Pre-registered reading of a paired-difference CI (proposal 2, H1/H2 criteria).

    ``supported``: CI entirely above 0. ``worse``: entirely below 0.
    ``equivalent``: CI inside +/- margin. Otherwise ``inconclusive``.
    """
    lo, hi = ci
    if lo > 0:
        return "supported"
    if hi < 0:
        return "worse"
    if margin is not None and within_equivalence_margin(ci, margin):
        return "equivalent"
    return "inconclusive"


__all__ = [
    "BaselineMatch",
    "ConfigPoint",
    "OraclePoint",
    "SystemScore",
    "MATCHING_RULES",
    "baseline_scores",
    "fixed_config_frontier",
    "fixed_config_points",
    "match_baseline",
    "match_fixed_baseline",
    "oracle_at_budget",
    "oracle_frontier",
    "oracle_points",
    "paired_document_bootstrap",
    "execution_tokens",
    "fixed_cost_points",
    "per_query_table",
    "select_comparator",
    "score_fixed",
    "score_routed",
    "upper_envelope",
    "verdict",
    "within_equivalence_margin",
]

"""Query-time optimizer: LLM-profile router (Plan A) and its shared selection core.

Design (docs/OPTIMIZER_PLAN.md):

1. An LLM profiles each question (``optimizer.profiler``; METIS-style).
2. An acceptability model maps a profile to ``P(acceptable | profile, c)`` for every
   catalog config c:
   - ``llm_profile_table`` (A1, primary): acceptance rates per profile cell, learned
     from train labels with hierarchical Beta smoothing;
   - ``llm_profile_rules`` (A2, ablation): a hand-written mapping (``RULES_V1``) fixed
     before any labels were collected.
3. ``PlanSelector`` takes the cheapest config with P >= theta, else the fallback.
4. Theta is calibrated on dev so that mean *total* cost (context tokens + profiler
   input tokens) meets ``target_mean_tokens``.

Plan B (``FeatureExtractor`` / ``AcceptabilityModel``: embeddings + logistic regression)
stays a stub. It is built only if the LLM router shows no improvement on dev.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_raptor.cache import stable_hash
from adaptive_raptor.config.schema import ConfigRef, ExperimentConfig, OptimizerSection
from adaptive_raptor.data import guard_split
from adaptive_raptor.data.records import Document, Question
from adaptive_raptor.optimizer.profiler import (
    ProfileResult,
    ProfileStore,
    QueryProfile,
    QueryProfiler,
)
from adaptive_raptor.policy import ConfigPolicy, PlannedConfig
from adaptive_raptor.profiling import (
    LabelRecord,
    LabelReport,
    acceptability_labels,
    select_records,
)
from adaptive_raptor.stub import Stub, ensure_implemented

CHECKPOINT_FORMAT = 1
DEFAULT_THETA_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.01))


@dataclass(frozen=True)
class QueryContext:
    question_id: str
    doc_id: str
    text: str
    embedding: Any = None
    document: Document | None = None


@dataclass
class RoutingDecision:
    config: ConfigRef
    probabilities: dict[str, float] = field(default_factory=dict)
    theta: float | None = None
    used_fallback: bool = False
    fallback_reason: str | None = None
    profile: dict[str, Any] | None = None
    profile_confidence: float | None = None
    router_input_tokens: int = 0
    router_output_tokens: int = 0
    router_ms: float = 0.0
    profile_cache_hit: bool = False

    def trace_fields(self, router_model: str) -> dict[str, Any]:
        return {
            "router_model": router_model,
            "route_fallback_reason": self.fallback_reason,
            "profile": self.profile,
            "profile_confidence": self.profile_confidence,
            "router_input_tokens": self.router_input_tokens,
            "router_output_tokens": self.router_output_tokens,
            "router_ms": round(self.router_ms, 3),
        }


@dataclass
class RouterCheckpoint:
    config_hash: str
    catalog: list[str]
    theta: float
    fallback: str
    artifacts: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- Plan B stubs


class FeatureExtractor(Stub):
    """Plan B only (not built unless the LLM router fails on dev)."""

    def __init__(self, cfg: OptimizerSection) -> None:
        self.cfg = cfg

    def fit(self, train: Sequence[QueryContext]) -> FeatureExtractor:
        """Fit training-only PCA and any lexical feature state."""
        raise NotImplementedError

    def transform(self, queries: Sequence[QueryContext]):
        """Return frozen-embedding PCA plus lexical query features."""
        raise NotImplementedError


class AcceptabilityModel(Stub):
    """Plan B only: per-config logistic regression on embedding + lexical features."""

    def __init__(self, cfg: OptimizerSection, config_ids: Sequence[str]) -> None:
        self.cfg = cfg
        self.config_ids = list(config_ids)

    def fit(self, features, labels: dict[str, dict[str, bool]], groups) -> AcceptabilityModel:
        raise NotImplementedError

    def predict_proba(self, features) -> list[dict[str, float]]:
        raise NotImplementedError


# --------------------------------------------------------------------------- cost


class CostModel:
    """Expected execution tokens per config (dev means of generation input + output, the
    input including the retrieved context) plus a per-query routing overhead (profiler
    input + output tokens). ``total`` is the cost used for selection and calibration."""

    def __init__(self, mean_tokens: Mapping[str, float] | None = None, overhead: float = 0.0):
        self.mean_tokens = dict(mean_tokens or {})
        self.overhead = float(overhead)

    def fit(self, dev_records: Sequence[LabelRecord], overhead: float = 0.0) -> CostModel:
        from adaptive_raptor.evaluation.research import execution_tokens

        sums: dict[str, list[int]] = defaultdict(list)
        for r in dev_records:
            sums[r.config_id].append(execution_tokens(r))
        self.mean_tokens = {c: float(np.mean(v)) for c, v in sums.items()}
        self.overhead = float(overhead)
        return self

    def cost(self, config: ConfigRef | str) -> float:
        cid = config.id if isinstance(config, ConfigRef) else config
        if cid in self.mean_tokens:
            return self.mean_tokens[cid]
        if isinstance(config, ConfigRef):
            return float(config.budget)  # unseen config: its budget is an upper bound
        raise KeyError(f"no cost for config '{cid}'")

    def total(self, config: ConfigRef | str) -> float:
        return self.cost(config) + self.overhead

    def to_dict(self) -> dict[str, Any]:
        return {"mean_tokens": self.mean_tokens, "overhead": self.overhead}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CostModel:
        return cls(d["mean_tokens"], d.get("overhead", 0.0))


# --------------------------------------------------------------------------- selection


class PlanSelector:
    def __init__(
        self, catalog: Sequence[ConfigRef], cost_model: CostModel, fallback: ConfigRef
    ) -> None:
        self.catalog = list(catalog)
        self.cost_model = cost_model
        self.fallback = fallback
        # cheapest first; ties broken by catalog order (deterministic)
        self._order = sorted(
            range(len(self.catalog)), key=lambda i: (cost_model.cost(self.catalog[i]), i)
        )

    def select(self, probabilities: Mapping[str, float] | None, theta: float) -> RoutingDecision:
        """Select the cheapest acceptable config, or the frozen fallback."""
        if probabilities is None:
            return RoutingDecision(self.fallback, {}, theta, True, "no_profile")
        for i in self._order:
            c = self.catalog[i]
            if probabilities.get(c.id, 0.0) >= theta:
                return RoutingDecision(c, dict(probabilities), theta)
        return RoutingDecision(self.fallback, dict(probabilities), theta, True, "none_acceptable")


def _mean_cost(
    dev_probabilities: Mapping[str, Mapping[str, float] | None],
    selector: PlanSelector,
    theta: float,
) -> float:
    costs = [
        selector.cost_model.total(selector.select(p, theta).config)
        for p in dev_probabilities.values()
    ]
    return float(np.mean(costs))


def calibrate_theta(
    split: str,
    dev_probabilities: Mapping[str, Mapping[str, float] | None],
    selector: PlanSelector,
    target_mean_tokens: float,
    grid: Sequence[float] | None = None,
) -> float:
    """Largest theta whose dev mean total cost (context + overhead) is <= the target.

    Mean cost is non-decreasing in theta, so this is the most quality-seeking threshold
    that still meets the budget. Raises if even theta=min(grid) exceeds the target.
    """
    guard_split(split, "calibrate")
    if not dev_probabilities:
        raise ValueError("no dev queries to calibrate on")
    feasible = [
        t
        for t in (grid or DEFAULT_THETA_GRID)
        if _mean_cost(dev_probabilities, selector, t) <= target_mean_tokens + 1e-9
    ]
    if not feasible:
        raise ValueError(
            f"no theta meets target_mean_tokens={target_mean_tokens} "
            f"(cheapest routing costs {_mean_cost(dev_probabilities, selector, 0.0):.0f})"
        )
    return float(max(feasible))


def theta_sweep(
    split: str,
    dev_probabilities: Mapping[str, Mapping[str, float] | None],
    selector: PlanSelector,
    dev_lookup: Mapping[str, Mapping[str, LabelRecord]],
    grid: Sequence[float] | None = None,
) -> list[tuple[float, float, float]]:
    """``(theta, mean total tokens, mean F1)`` on dev, replayed from dev label records.

    Tokens are the recorded execution tokens (generation input + output) of the chosen
    config plus the routing overhead; F1 is the recorded F1 of that config. No API calls.
    """
    from adaptive_raptor.evaluation.research import execution_tokens

    guard_split(split, "calibrate")
    out = []
    for t in grid or DEFAULT_THETA_GRID:
        toks, f1s = [], []
        for q, p in dev_probabilities.items():
            rec = dev_lookup[q][selector.select(p, t).config.id]
            toks.append(execution_tokens(rec) + selector.cost_model.overhead)
            f1s.append(rec.answer_f1)
        out.append((float(t), float(np.mean(toks)), float(np.mean(f1s))))
    return out


# --------------------------------------------------------------------------- Plan A models


def _n_bucket(n: int) -> int:
    return min(n, 3)  # 1, 2, 3+


def profile_cells(p: QueryProfile) -> list[str]:
    """Cell keys from coarse to fine; the last one is the full cell."""
    return [
        "*",
        p.answer_form,
        f"{p.answer_form}|{p.granularity}",
        f"{p.answer_form}|{p.granularity}|{_n_bucket(p.n_facts)}|{int(p.cross_section)}",
    ]


class ProfileTableModel:
    """A1: P(acceptable | profile cell, c) with hierarchical Beta smoothing.

    The global rate (Laplace-smoothed) is the prior for each answer form, which is the
    prior for (form, granularity), which is the prior for the full cell:
    ``p_child = (k + alpha * p_parent) / (n + alpha)``. Sparse cells therefore fall back
    smoothly toward their parent; unseen cells get the parent's rate exactly.
    """

    kind = "llm_profile_table"

    def __init__(self, config_ids: Sequence[str], alpha: float = 5.0) -> None:
        self.config_ids = list(config_ids)
        self.alpha = float(alpha)
        self.counts: dict[str, dict[str, list[int]]] = {}  # cell -> config -> [k, n]

    def fit(
        self, profiles: Sequence[QueryProfile], labels: Sequence[Mapping[str, bool]]
    ) -> ProfileTableModel:
        if len(profiles) != len(labels):
            raise ValueError("profiles and labels differ in length")
        if not profiles:
            raise ValueError("no labeled, profiled training queries")
        counts: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {c: [0, 0] for c in self.config_ids}
        )
        for p, y in zip(profiles, labels, strict=True):
            for cell in profile_cells(p):
                for c in self.config_ids:
                    counts[cell][c][0] += int(bool(y[c]))
                    counts[cell][c][1] += 1
        self.counts = {cell: dict(v) for cell, v in counts.items()}
        return self

    def predict(self, p: QueryProfile) -> dict[str, float]:
        out = {}
        for c in self.config_ids:
            k, n = self.counts["*"][c]
            prob = (k + 1) / (n + 2)
            for cell in profile_cells(p)[1:]:
                if cell in self.counts:
                    k, n = self.counts[cell][c]
                    prob = (k + self.alpha * prob) / (n + self.alpha)
            out[c] = prob
        return out

    def cell_sizes(self) -> dict[str, int]:
        first = self.config_ids[0]
        return {cell: v[first][1] for cell, v in self.counts.items() if cell.count("|") == 3}

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "config_ids": self.config_ids,
            "alpha": self.alpha,
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ProfileTableModel:
        m = cls(d["config_ids"], d["alpha"])
        m.counts = {cell: {c: list(kn) for c, kn in v.items()} for cell, v in d["counts"].items()}
        return m


# Written on 2026-11-09, before any router labels existed (METIS Alg. 1 analogue).
RULES_V1: dict[str, Any] = {
    "strategies": {
        "specific_detail": ["leaf_only", "collapsed"],
        "section_level": ["leaf_only", "collapsed"],
        "whole_paper": ["summary_only", "collapsed"],
    },
    # budget level index into the sorted catalog budgets: 0 = smallest
    "budget_level": {
        "single_fact": 0,  # yes_no / short_span, 1 fact, one section
        "few_facts": 1,  # <= 2 facts, one section
        "otherwise": 2,
    },
    "unanswerable_likely": {"strategies": ["collapsed"], "budget_level": 1},
}


class RuleMapModel:
    """A2: hand-written profile -> allowed configs (P = 1 allowed, 0 otherwise)."""

    kind = "llm_profile_rules"

    def __init__(self, catalog: Sequence[ConfigRef], rules: Mapping[str, Any] = RULES_V1):
        self.catalog = list(catalog)
        self.rules = dict(rules)
        self.budgets = sorted({c.budget for c in self.catalog})

    def fit(self, profiles=(), labels=()) -> RuleMapModel:  # nothing to learn
        return self

    def _allowed(self, p: QueryProfile) -> tuple[set[str], int]:
        if p.answer_form == "unanswerable_likely":
            r = self.rules["unanswerable_likely"]
            return set(r["strategies"]), r["budget_level"]
        levels = self.rules["budget_level"]
        if p.answer_form in ("yes_no", "short_span") and p.n_facts == 1 and not p.cross_section:
            level = levels["single_fact"]
        elif p.n_facts <= 2 and not p.cross_section:
            level = levels["few_facts"]
        else:
            level = levels["otherwise"]
        return set(self.rules["strategies"][p.granularity]), level

    def predict(self, p: QueryProfile) -> dict[str, float]:
        strategies, level = self._allowed(p)
        min_budget = self.budgets[min(level, len(self.budgets) - 1)]
        return {
            c.id: float(c.strategy.value in strategies and c.budget >= min_budget)
            for c in self.catalog
        }

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "rules": self.rules}


def _model_from_dict(d: Mapping[str, Any], catalog: Sequence[ConfigRef]):
    if d["kind"] == ProfileTableModel.kind:
        return ProfileTableModel.from_dict(d)
    if d["kind"] == RuleMapModel.kind:
        return RuleMapModel(catalog, d["rules"])
    raise ValueError(f"unknown model kind '{d['kind']}'")


# --------------------------------------------------------------------------- router


class Router:
    """Profile -> P(acceptable) -> cheapest acceptable config, with a frozen theta."""

    def __init__(
        self,
        cfg: OptimizerSection,
        catalog: Sequence[ConfigRef],
        *,
        model: ProfileTableModel | RuleMapModel,
        cost_model: CostModel,
        theta: float,
        profiler: QueryProfiler,
    ) -> None:
        self.cfg = cfg
        self.catalog = list(catalog)
        self.model = model
        self.selector = PlanSelector(self.catalog, cost_model, cfg.fallback)
        self.theta = float(theta)
        self.profiler = profiler
        self.threshold = cfg.profiler.confidence_threshold

    @property
    def name(self) -> str:
        return f"{self.model.kind}/{self.profiler.model_name}/{self.profiler.prompt_hash}"

    def probabilities(self, r: ProfileResult) -> tuple[dict[str, float] | None, str | None]:
        """Model probabilities for one profile, or (None, reason) to force the fallback."""
        if r.profile is None:
            return None, "invalid_profile" if r.error else "no_profile"
        if (
            self.threshold is not None
            and r.confidence is not None
            and r.confidence < self.threshold
        ):
            return None, "low_confidence"
        return self.model.predict(r.profile), None

    def decide(self, r: ProfileResult) -> RoutingDecision:
        probs, reason = self.probabilities(r)
        d = self.selector.select(probs, self.theta)
        if reason is not None:
            d.fallback_reason = reason
        d.profile = r.profile.model_dump() if r.profile else None
        d.profile_confidence = r.confidence
        d.router_input_tokens, d.router_output_tokens = r.input_tokens, r.output_tokens
        d.router_ms, d.profile_cache_hit = r.latency_ms, r.cache_hit
        return d

    def route(self, query: QueryContext) -> RoutingDecision:
        r = self.profiler.profile(query.question_id, query.text, query.document)
        return self.decide(r)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        cfg: OptimizerSection,
        catalog: Sequence[ConfigRef],
        profiler: QueryProfiler,
    ) -> Router:
        ckpt = load_checkpoint(path)
        art = ckpt.artifacts
        if [c.id for c in catalog] != ckpt.catalog:
            raise ValueError(f"catalog differs from checkpoint: {ckpt.catalog}")
        if art.get("prompt_hash") != profiler.prompt_hash:
            raise ValueError(
                f"profiler prompt hash {profiler.prompt_hash} != checkpoint "
                f"{art.get('prompt_hash')}; the router was trained on other profiles"
            )
        return cls(
            cfg,
            catalog,
            model=_model_from_dict(art["model"], catalog),
            cost_model=CostModel.from_dict(art["cost_model"]),
            theta=ckpt.theta,
            profiler=profiler,
        )


class RouterPolicy(ConfigPolicy):
    """Pipeline adapter that runs only the router's selected configuration."""

    def __init__(self, router: Router) -> None:
        ensure_implemented(router)
        self.router = router
        self.decisions: dict[str, RoutingDecision] = {}

    def configs_for(
        self, question: Question, query_embedding, document: Document | None = None
    ) -> list[PlannedConfig]:
        context = QueryContext(
            question.question_id, question.doc_id, question.text, query_embedding, document
        )
        decision = self.router.route(context)
        self.decisions[question.question_id] = decision
        return [decision.config]

    def trace_fields(self, question_id: str) -> dict[str, Any]:
        d = self.decisions.get(question_id)
        return d.trace_fields(self.router.name) if d else {}


# --------------------------------------------------------------------------- checkpoints


def save_checkpoint(ckpt: RouterCheckpoint, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload = {"format": CHECKPOINT_FORMAT, **asdict(ckpt)}
    (path / "checkpoint.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_checkpoint(path: Path) -> RouterCheckpoint:
    path = Path(path)
    file = path / "checkpoint.json" if path.is_dir() else path
    d = json.loads(file.read_text())
    if d.pop("format", None) != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported router checkpoint format in {file}")
    return RouterCheckpoint(**d)


# --------------------------------------------------------------------------- training


def optimizer_config_hash(cfg: OptimizerSection, catalog: Sequence[ConfigRef]) -> str:
    relevant = cfg.model_dump(
        mode="json", exclude={"enabled", "checkpoint", "train_labels", "dev_labels"}
    )
    relevant["profiler"].pop("profiles_path", None)
    return stable_hash([relevant, [c.id for c in catalog]])


@dataclass
class TrainResult:
    checkpoint: RouterCheckpoint
    train_report: LabelReport
    dev_probabilities: dict[str, dict[str, float] | None]
    dev_decisions: dict[str, RoutingDecision]
    sweep: list[tuple[float, float, float]]
    missing_train_profiles: list[str]
    missing_dev_profiles: list[str]
    model_cell_sizes: dict[str, int] = field(default_factory=dict)  # A1 full cells (train)


def _table(records: Sequence[LabelRecord], config_ids: Sequence[str]):
    table: dict[str, dict[str, LabelRecord]] = defaultdict(dict)
    for r in records:
        if r.config_id in config_ids:
            table[r.question_id][r.config_id] = r
    return {q: row for q, row in table.items() if len(row) == len(config_ids)}


def train_router(
    config: ExperimentConfig,
    *,
    train_records: Sequence[LabelRecord],
    dev_records: Sequence[LabelRecord],
    profiles: ProfileStore,
    prompt_hash: str,
    profiler_model: str,
    split_train: str = "train",
    split_dev: str = "dev",
) -> TrainResult:
    """Fit the Plan A model on train labels, calibrate theta on dev; never touches test.

    Uses the labels of the first chunking variant, the first RAPTOR seed and
    ``optimizer.label_run``. Queries without a stored profile are skipped in training and
    routed to the fallback on dev (counted in the result, never hidden).
    """
    guard_split(split_train, "train")
    guard_split(split_dev, "calibrate")
    opt = config.optimizer
    catalog = config.retrieval.catalog()
    ids = [c.id for c in catalog]
    cell = dict(
        index_variant=config.chunking.variants[0].name,
        seed=config.raptor.seeds[0],
        run=opt.label_run,
        configs=set(ids),
    )
    train = select_records(train_records, **cell)
    dev = select_records(dev_records, **cell)

    report = LabelReport(opt.label_epsilon, ids)
    labels = acceptability_labels(train, opt.label_epsilon, ids, report=report)
    fit_profiles, fit_labels, missing_train = [], [], []
    for q, y in labels.items():
        if y is None:
            continue  # all-fail: excluded from training (proposal 5.2)
        r = profiles.get(q, prompt_hash)
        if r is None or r.profile is None:
            missing_train.append(q)
            continue
        fit_profiles.append(r.profile)
        fit_labels.append(y)

    if opt.model_type == "llm_profile_table":
        model: ProfileTableModel | RuleMapModel = ProfileTableModel(ids, opt.table_alpha).fit(
            fit_profiles, fit_labels
        )
    elif opt.model_type == "llm_profile_rules":
        model = RuleMapModel(catalog)
    else:
        raise NotImplementedError(
            f"optimizer.model_type={opt.model_type} is Plan B (built only if Plan A fails)"
        )

    dev_table = _table(dev, ids)
    dev_profiles = {q: profiles.get(q, prompt_hash) for q in dev_table}
    missing_dev = sorted(q for q, r in dev_profiles.items() if r is None)
    overhead = 0.0
    if opt.cost_includes_profiler:
        seen = [r.input_tokens + r.output_tokens for r in dev_profiles.values() if r is not None]
        overhead = float(np.mean(seen)) if seen else 0.0
    cost_model = CostModel().fit([r for row in dev_table.values() for r in row.values()], overhead)

    # A router with theta=0 just to reuse the fallback logic for probabilities
    tmp = Router(
        opt,
        catalog,
        model=model,
        cost_model=cost_model,
        theta=0.0,
        profiler=_NullProfiler(prompt_hash, profiler_model),
    )
    dev_probs: dict[str, dict[str, float] | None] = {}
    for q, r in dev_profiles.items():
        dev_probs[q] = tmp.probabilities(r)[0] if r is not None else None

    if opt.theta is not None:
        theta = opt.theta
    elif opt.model_type == "llm_profile_rules":
        theta = 0.5  # probabilities are 0/1: theta does not change decisions
    else:
        target = opt.target_mean_tokens
        if opt.target_config is not None:  # the dev total cost of a fixed config
            target = cost_model.cost(opt.target_config)
        if target is None:
            raise ValueError("set optimizer.theta, target_mean_tokens or target_config")
        theta = calibrate_theta(split_dev, dev_probs, tmp.selector, target)
    sweep = theta_sweep(split_dev, dev_probs, tmp.selector, dev_table)

    tmp.theta = theta
    decisions = {}
    for q, r in dev_profiles.items():
        decisions[q] = (
            tmp.decide(r)
            if r is not None
            else RoutingDecision(opt.fallback, {}, theta, True, "no_profile")
        )

    ckpt = RouterCheckpoint(
        config_hash=optimizer_config_hash(opt, catalog),
        catalog=ids,
        theta=float(theta),
        fallback=opt.fallback.id,
        artifacts={
            "model_type": opt.model_type,
            "model": model.to_dict(),
            "cost_model": cost_model.to_dict(),
            "prompt_hash": prompt_hash,
            "profiler_model": profiler_model,
            "label_epsilon": opt.label_epsilon,
            "label_cell": {k: v for k, v in cell.items() if k != "configs"},
            "confidence_threshold": opt.profiler.confidence_threshold,
            "target_mean_tokens": opt.target_mean_tokens,
            "target_config": opt.target_config,
            "n_train_queries": len(fit_labels),
            "n_dev_queries": len(dev_table),
            # paper ids, so H1 can refuse to evaluate on papers used here
            "train_docs": sorted({r.doc_id for r in train}),
            "dev_docs": sorted({r.doc_id for r in dev}),
        },
    )
    sizes = model.cell_sizes() if isinstance(model, ProfileTableModel) else {}
    return TrainResult(ckpt, report, dev_probs, decisions, sweep, missing_train, missing_dev, sizes)


class _NullProfiler(QueryProfiler):
    def __init__(self, prompt_hash: str, model_name: str) -> None:
        self.prompt_hash, self.model_name = prompt_hash, model_name

    def profile(self, question_id, question, doc):  # pragma: no cover - never called
        raise RuntimeError("training replays stored profiles only")


__all__ = [
    "RULES_V1",
    "AcceptabilityModel",
    "CostModel",
    "FeatureExtractor",
    "PlanSelector",
    "ProfileTableModel",
    "QueryContext",
    "Router",
    "RouterCheckpoint",
    "RouterPolicy",
    "RoutingDecision",
    "RuleMapModel",
    "TrainResult",
    "calibrate_theta",
    "load_checkpoint",
    "optimizer_config_hash",
    "profile_cells",
    "save_checkpoint",
    "theta_sweep",
    "train_router",
]

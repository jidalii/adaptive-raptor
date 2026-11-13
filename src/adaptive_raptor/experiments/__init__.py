"""Experiment contracts, baseline runner, research placeholders, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
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
from adaptive_raptor.experiments.outputs import write_json
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


class QueryProfilingExperiment(Experiment):
    """Profile every question of a split once with the LLM profiler (Plan A, WP3).

    Paid unless ``optimizer.profiler.provider`` is ``mock``. Responses are cached in
    ``<cache_dir>/profiles``, so a rerun is free. Output: ``profiles.jsonl``.
    """

    experiment_type = ExperimentType.QUERY_PROFILING
    requires_credentials = True

    def planned_steps(self) -> list[str]:
        return [
            "load dataset split and paper metadata",
            "profile each question (question + configured metadata only; no answers)",
            "write profiles.jsonl and profile statistics",
        ]

    def run(self) -> ExperimentResult:
        from concurrent.futures import ThreadPoolExecutor

        from adaptive_raptor.data import load_dataset
        from adaptive_raptor.optimizer.profiler import ProfileStore, build_profiler

        cfg, rt = self.config, self.config.runtime
        dataset = load_dataset(cfg.dataset)
        docs = {d.doc_id: d for d in dataset.documents}
        profiler = build_profiler(
            cfg.optimizer.profiler,
            cache_dir=rt.cache_dir / "profiles",
            use_cache=rt.use_cache,
            max_retries=rt.max_retries,
        )
        with ThreadPoolExecutor(max_workers=rt.concurrency) as pool:
            results = list(
                pool.map(
                    lambda q: profiler.profile(q.question_id, q.text, docs.get(q.doc_id)),
                    dataset.questions,
                )
            )
        out = self.run_dir / "profiles.jsonl"
        ProfileStore.append(out, results)
        metrics = profile_statistics(results, dataset.questions, cfg)
        metrics["prompt_hash"] = profiler.prompt_hash
        metrics["dataset"] = dataset.manifest
        invalid = metrics["invalid"]
        status = RunStatus.SUCCEEDED if invalid < len(results) else RunStatus.FAILED
        return ExperimentResult(
            status,
            f"{len(results)} profiles ({invalid} invalid)",
            artifacts={"profiles": str(out)},
            metrics=metrics,
        )


def profile_statistics(results, questions, cfg: ExperimentConfig) -> dict[str, Any]:
    import numpy as np

    by_q = {q.question_id: q for q in questions}
    valid = [r for r in results if r.profile is not None]
    conf = [r.confidence for r in valid if r.confidence is not None]
    threshold = cfg.optimizer.profiler.confidence_threshold
    fresh = [r for r in results if not r.cache_hit]
    tin = sum(r.input_tokens for r in results)
    tout = sum(r.output_tokens for r in results)
    gen = cfg.generation

    def usd(i, o):
        if gen.price_input_per_mtok is None or gen.price_output_per_mtok is None:
            return None
        return round(i / 1e6 * gen.price_input_per_mtok + o / 1e6 * gen.price_output_per_mtok, 4)

    by_type: dict[str, Counter] = {}
    for r in valid:
        golds = by_q[r.question_id].answers if r.question_id in by_q else ()
        gold_type = golds[0].answer_type if golds else "unknown"
        by_type.setdefault(gold_type, Counter())[r.profile.answer_form] += 1
    return {
        "questions": len(results),
        "invalid": len(results) - len(valid),
        "errors": dict(Counter((r.error or "").split(":")[0] for r in results if r.error)),
        "retried": sum(r.attempts > 1 for r in results),
        "confidence_available": len(conf),
        "confidence_quantiles": (
            dict(
                zip(
                    ("p10", "p50", "p90"),
                    np.quantile(conf, [0.1, 0.5, 0.9]).round(4).tolist(),
                    strict=True,
                )
            )
            if conf
            else None
        ),
        "below_threshold": sum(c < threshold for c in conf) if threshold is not None else None,
        "answer_form": dict(Counter(r.profile.answer_form for r in valid)),
        "granularity": dict(Counter(r.profile.granularity for r in valid)),
        "n_facts": dict(Counter(r.profile.n_facts for r in valid)),
        "cross_section": dict(Counter(str(r.profile.cross_section).lower() for r in valid)),
        "answer_form_by_gold_type": {t: dict(c) for t, c in sorted(by_type.items())},
        "mean_input_tokens": round(tin / len(results), 1) if results else 0,
        "mean_output_tokens": round(tout / len(results), 1) if results else 0,
        "cost_usd": usd(tin, tout),
        "billed_usd": usd(sum(r.input_tokens for r in fresh), sum(r.output_tokens for r in fresh)),
    }


def _prices(cfg: ExperimentConfig) -> tuple[float, float] | None:
    g = cfg.generation
    if g.price_input_per_mtok is None or g.price_output_per_mtok is None:
        return None
    return (g.price_input_per_mtok, g.price_output_per_mtok)


def _label_table(records, cell: dict[str, Any], ids: list[str]) -> dict[str, dict]:
    """question -> config -> record for one (variant, seed, run) cell, complete rows only."""
    table: dict[str, dict] = {}
    for r in records:
        if (
            r.index_variant == cell["index_variant"]
            and r.seed == cell["seed"]
            and r.run == cell["run"]
            and r.config_id in ids
        ):
            table.setdefault(r.question_id, {})[r.config_id] = r
    return {q: row for q, row in sorted(table.items()) if len(row) == len(ids)}


def _replay(router, table: dict[str, dict], store) -> tuple[dict, dict, dict, list[str]]:
    """Route every question of ``table`` from stored profiles only (never calls an API)."""
    choices, prof_tokens, fallback, missing = {}, {}, {}, []
    for q in table:
        r = store.get(q, router.profiler.prompt_hash)
        if r is None:
            missing.append(q)
            continue
        d = router.decide(r)
        choices[q] = d.config.id
        prof_tokens[q] = (r.input_tokens, r.output_tokens)
        fallback[q] = d.used_fallback
    return choices, prof_tokens, fallback, missing


def _delta(a, b, q2d: dict[str, str], cfg: ExperimentConfig) -> dict[str, Any]:
    from adaptive_raptor.evaluation.research import paired_document_bootstrap, verdict

    qs = sorted(set(a.per_query_f1) & set(b.per_query_f1))
    delta = {q: a.per_query_f1[q] - b.per_query_f1[q] for q in qs}
    mean, lo, hi = paired_document_bootstrap(
        delta,
        q2d,
        cfg.evaluation.bootstrap_samples,
        cfg.evaluation.confidence_level,
        seed=cfg.experiment.seed,
    )
    return {
        "system": a.name,
        "baseline": b.name,
        "n": len(qs),
        "delta_f1": round(mean, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "verdict": verdict((lo, hi), cfg.evaluation.equivalence_margin),
    }


class RouterTrainExperiment(Experiment):
    """Train + calibrate the Plan A router offline and replay it on dev (WP9).

    No API calls: labels come from catalog runs (``optimizer.train_labels`` /
    ``dev_labels``) and profiles from query_profiling runs. Everything H1 needs from dev
    is frozen into the checkpoint here: theta, the cost model, the router's dev mean
    total tokens, and the dev-selected fixed comparator under both matching rules.
    Dev numbers are optimistic (theta is calibrated on the same dev queries).
    """

    experiment_type = ExperimentType.ROUTER_TRAIN
    requires_credentials = False

    def planned_steps(self) -> list[str]:
        return [
            "merge train/dev label tables and load stored profiles",
            "derive epsilon-acceptability labels on train",
            "fit the Plan A acceptability model",
            "fit the cost model and calibrate theta on dev (total cost)",
            "replay on dev; select the matched fixed comparator on dev and freeze it",
            "write the frozen checkpoint, theta sweep and dev report",
        ]

    def run(self) -> ExperimentResult:
        from adaptive_raptor.evaluation.research import (
            MATCHING_RULES,
            fixed_cost_points,
            oracle_at_budget,
            score_fixed,
            score_routed,
            select_comparator,
            upper_envelope,
        )
        from adaptive_raptor.optimizer import (
            PlanSelector,
            calibrate_theta,
            save_checkpoint,
            train_router,
        )
        from adaptive_raptor.optimizer.profiler import ProfileStore, build_profiler
        from adaptive_raptor.profiling import merge_label_tables

        cfg = self.config
        opt = cfg.optimizer
        prices = _prices(cfg)
        train, train_merge = merge_label_tables(opt.train_labels)
        dev, dev_merge = merge_label_tables(opt.dev_labels)
        profiler = build_profiler(opt.profiler)  # only for its prompt hash; never called
        store = ProfileStore.load(opt.profiler.profiles_path)
        res = train_router(
            cfg,
            train_records=train,
            dev_records=dev,
            profiles=store,
            prompt_hash=profiler.prompt_hash,
            profiler_model=profiler.model_name,
        )
        ckpt = res.checkpoint
        art = ckpt.artifacts
        ids = ckpt.catalog
        table = _label_table(dev, art["label_cell"], ids)
        q2d = {q: next(iter(row.values())).doc_id for q, row in table.items()}
        prof_tokens = {
            q: (d.router_input_tokens, d.router_output_tokens) for q, d in res.dev_decisions.items()
        }

        def replay(theta: float) -> tuple[Any, dict]:
            sel = PlanSelector(cfg.retrieval.catalog(), _cost_model(art), opt.fallback)
            dec = {q: sel.select(res.dev_probabilities.get(q), theta) for q in table}
            choices = {q: d.config.id for q, d in dec.items()}
            fb = {q: d.used_fallback for q, d in dec.items()}
            return score_routed(opt.model_type, choices, table, prof_tokens, fb, prices), dec

        router, _ = replay(ckpt.theta)
        router_row = router.row()
        points = fixed_cost_points(table, ids)
        frontier = upper_envelope(points)

        # comparators are selected on dev by the router's dev mean TOTAL cost, then frozen
        comparators = {}
        for rule in MATCHING_RULES:
            m = select_comparator(router_row["total_tokens"], points, rule)
            comparators[rule] = {
                "weights": m.weights,
                "dev_total_tokens": round(m.mean_tokens, 1),
                "dev_f1": round(m.mean_f1, 4),
                "rule_satisfied": m.satisfied,
            }
        primary = cfg.evaluation.budget_matching
        art["comparators"] = comparators
        art["primary_matching"] = primary
        art["dev_router"] = router_row
        art["dev_fixed_points"] = [
            {
                "config": p.config_id,
                "total_tokens": round(p.mean_tokens, 1),
                "f1": round(p.mean_f1, 4),
            }
            for p in points
        ]
        router_dir = self.run_dir / "router"
        save_checkpoint(ckpt, router_dir)

        deltas = []
        for rule, comp in comparators.items():
            base = score_fixed(f"fixed[{rule}]", comp["weights"], table, prices)
            deltas.append(
                {
                    **_delta(router, base, q2d, cfg),
                    "matching": rule,
                    "primary": rule == primary,
                    "comparator": comp,
                }
            )

        # Pareto check: at each fixed frontier config's cost, the best this router can do
        sel = PlanSelector(cfg.retrieval.catalog(), _cost_model(art), opt.fallback)
        # targets: every fixed frontier config, plus original RAPTOR (the baseline) if the
        # dev labels include it; the router is calibrated to each target's dev total cost
        targets = [
            (
                p.config_id,
                p.mean_tokens,
                score_fixed(p.config_id, {p.config_id: 1.0}, table, prices),
            )
            for p in frontier
        ]
        orig_table = _label_table(dev, art["label_cell"], ["raptor_original"])
        original = None
        if orig_table:
            original = score_fixed("raptor_original", {"raptor_original": 1.0}, orig_table, prices)
            targets.append(("raptor_original", original.row()["total_tokens"], original))
        operating = []
        for name, cost, base in targets:
            row = {
                "fixed": name,
                "fixed_total_tokens": round(cost, 1),
                "fixed_f1": base.row()["answer_f1"],
                **dict.fromkeys(
                    (
                        "theta",
                        "router_total_tokens",
                        "router_f1",
                        "router_fallback_rate",
                        "delta_f1",
                        "ci",
                        "verdict",
                    )
                ),
            }
            try:
                theta = calibrate_theta("dev", res.dev_probabilities, sel, cost)
            except ValueError:
                operating.append({**row, "verdict": "router cannot reach this cost"})
                continue
            r_score, _ = replay(theta)
            d = _delta(r_score, base, q2d, cfg)
            rr = r_score.row()
            operating.append(
                {
                    **row,
                    "theta": theta,
                    "router_total_tokens": rr["total_tokens"],
                    "router_f1": rr["answer_f1"],
                    "router_fallback_rate": rr["fallback_rate"],
                    "delta_f1": d["delta_f1"],
                    "ci": [d["ci_lo"], d["ci_hi"]],
                    "verdict": d["verdict"],
                }
            )

        # router's own frontier from the theta sweep, and its Pareto standing
        sweep_pts = sorted({(round(t, 1), round(f, 4)) for _, t, f in res.sweep})
        dominated_fixed = [
            p.config_id
            for p in frontier
            if any(t <= p.mean_tokens + 1e-9 and f > p.mean_f1 for t, f in sweep_pts)
        ]
        try:
            all_dev = [r for row in table.values() for r in row.values()]
            oracle_f1 = round(
                oracle_at_budget(all_dev, None, router_row["context_tokens"], ids).mean_f1, 4
            )
        except ValueError:
            oracle_f1 = None

        dec_all = res.dev_decisions
        reasons = Counter(d.fallback_reason for d in dec_all.values() if d.used_fallback)
        chosen = Counter(d.config.id for q, d in dec_all.items() if q in table)
        dev_prof = [store.get(q, profiler.prompt_hash) for q in table]
        dev_prof = [r.profile for r in dev_prof if r is not None and r.profile is not None]
        from adaptive_raptor.optimizer import profile_cells

        sizes = res.model_cell_sizes
        small = sum(n for n in sizes.values() if n < 10)
        dev_report = {
            "theta": ckpt.theta,
            "calibration_target": art.get("target_config") or art.get("target_mean_tokens"),
            "n_dev": router.n,
            "router": {
                **router_row,
                "fallback_reasons": dict(reasons),
                "config_frequency": dict(chosen.most_common()),
            },
            "deltas_vs_matched_fixed": deltas,
            "raptor_original": original.row() if original else None,
            "operating_points": operating,
            "fixed_frontier_points_dominated_by_router_sweep": dominated_fixed,
            "fixed_points": art["dev_fixed_points"],
            "fixed_frontier": [p.config_id for p in frontier],
            "in_sample_oracle_f1_at_router_context": oracle_f1,
            "profiles_dev": {
                f: dict(Counter(str(getattr(p, f)) for p in dev_prof))
                for f in ("answer_form", "granularity", "n_facts", "cross_section")
            },
            "profile_cells_dev": dict(Counter(profile_cells(p)[-1] for p in dev_prof)),
            "profile_cells_train": {
                "n_full_cells": len(sizes),
                "singletons": sum(n == 1 for n in sizes.values()),
                "share_of_train_in_cells_below_10": round(small / max(1, sum(sizes.values())), 3),
                "sizes": dict(sorted(sizes.items(), key=lambda kv: -kv[1])),
            },
            "train": {
                "labeled": res.train_report.n_labeled,
                "all_fail": res.train_report.n_all_fail,
                "incomplete": len(res.train_report.incomplete),
                "missing_profiles": len(res.missing_train_profiles),
                "acceptable_rate": res.train_report.acceptable_rate,
                "by_answer_type": res.train_report.by_answer_type,
                "merge": {
                    "rows": train_merge.n_input,
                    "unique": train_merge.n_unique,
                    "conflicts": len(train_merge.conflicts),
                },
            },
            "dev_missing_profiles": len(res.missing_dev_profiles),
            "dev_merge_conflicts": len(dev_merge.conflicts),
            "dev_execution_tokens_recorded": all(
                r.input_tokens > 0 for row in table.values() for r in row.values()
            ),
            "caveat": "dev numbers are optimistic: theta and comparators are chosen on these "
            "dev queries",
        }
        write_json(self.run_dir / "dev_report.json", dev_report)
        sweep_rows = [
            {"theta": t, "mean_total_tokens": round(tok, 1), "mean_f1": round(f1, 4)}
            for t, tok, f1 in res.sweep
        ]
        fmt = cfg.evaluation.output_formats
        artifacts = {"checkpoint": str(router_dir / "checkpoint.json")}
        for name, rows in (("theta_sweep", sweep_rows), ("operating_points", operating)):
            for p in write_rows(rows, self.run_dir, name, fmt):
                artifacts[p.name] = str(p)
        prim = next(d for d in deltas if d["primary"])
        return ExperimentResult(
            RunStatus.SUCCEEDED,
            f"theta={ckpt.theta}; dev delta F1 {prim['delta_f1']:+.4f} "
            f"[{prim['ci_lo']:+.4f}, {prim['ci_hi']:+.4f}] vs {primary} comparator",
            artifacts=artifacts,
            metrics=dev_report,
        )


def _cost_model(art: dict):
    from adaptive_raptor.optimizer import CostModel

    return CostModel.from_dict(art["cost_model"])


class H1OptimizerExperiment(Experiment):
    """H1 / E1 (WP10): frozen routers vs. the dev-selected fixed config, by replay.

    Every catalog config of the evaluated split is already labeled (T12 test labeling),
    so a router's answer for a question is the recorded answer of the config it picks:
    the same retrieval and the same cached generation an online run would produce. No API
    calls; nothing is fitted or selected here. Theta, the cost model and the matched
    comparator all come from the checkpoints (selected on dev in router_train).
    """

    experiment_type = ExperimentType.H1_OPTIMIZER
    requires_credentials = False

    def planned_steps(self) -> list[str]:
        return [
            "load frozen router checkpoints (primary + ablations) and stored profiles",
            "route every evaluated question from its stored profile",
            "score routers and their frozen dev-selected comparators from the label table",
            "report F1, context/profiler/total tokens and USD separately",
            "paired document bootstrap of delta F1 vs. each matched comparator",
        ]

    def run(self) -> ExperimentResult:
        from adaptive_raptor.evaluation.research import score_fixed, score_routed
        from adaptive_raptor.optimizer import Router, load_checkpoint
        from adaptive_raptor.optimizer.profiler import ProfileStore, build_profiler
        from adaptive_raptor.profiling import merge_label_tables

        cfg, opt = self.config, self.config.optimizer
        prices = _prices(cfg)
        catalog = cfg.retrieval.catalog()
        records, merge = merge_label_tables(opt.eval_labels)
        store = ProfileStore.load(opt.profiler.profiles_path)
        paths = [opt.checkpoint, *opt.ablation_checkpoints]
        ckpts = [load_checkpoint(p) for p in paths]
        cell = ckpts[0].artifacts["label_cell"]
        ids = [c.id for c in catalog]
        table = _label_table(records, cell, ids)
        eval_docs = {next(iter(row.values())).doc_id for row in table.values()}
        problems = []
        for path, ck in zip(paths, ckpts, strict=True):
            if ck.artifacts["label_cell"] != cell:
                problems.append(f"{path}: label cell differs from the primary checkpoint")
            seen = set(ck.artifacts.get("train_docs", [])) | set(ck.artifacts.get("dev_docs", []))
            if eval_docs & seen:
                problems.append(
                    f"{path}: {len(eval_docs & seen)} evaluated papers were used to train or "
                    "calibrate this router"
                )
        if problems:
            raise ValueError("; ".join(problems))

        q2d = {q: next(iter(row.values())).doc_id for q, row in table.items()}
        rows, deltas, missing_all = [], [], {}
        fixed_done: dict[str, Any] = {}
        for path, ck in zip(paths, ckpts, strict=True):
            art = ck.artifacts
            replay_profiler = build_profiler(opt.profiler)  # prompt hash only; never called
            router = Router.from_checkpoint(path, opt, catalog, replay_profiler)
            choices, prof_tokens, fallback, missing = _replay(router, table, store)
            name = f"router:{art['model_type']}"
            missing_all[name] = len(missing)
            routed = score_routed(name, choices, table, prof_tokens, fallback, prices)
            rows.append(
                {
                    **routed.row(),
                    "theta": ck.theta,
                    "dev_total_tokens": art["dev_router"]["total_tokens"],
                }
            )
            for rule, comp in art["comparators"].items():
                label = "+".join(f"{c}x{w:.2f}" for c, w in comp["weights"].items())
                fname = f"fixed[{rule}]:{label}"
                if fname not in fixed_done:
                    fixed_done[fname] = score_fixed(fname, comp["weights"], table, prices)
                    rows.append(
                        {
                            **fixed_done[fname].row(),
                            "theta": None,
                            "dev_total_tokens": comp["dev_total_tokens"],
                        }
                    )
                d = _delta(routed, fixed_done[fname], q2d, cfg)
                d["matching"] = rule
                d["primary"] = rule == art.get("primary_matching")
                deltas.append(d)
        original = _label_table(records, cell, ["raptor_original"])
        if original:
            orig = score_fixed("raptor_original", {"raptor_original": 1.0}, original, prices)
            rows.append({**orig.row(), "theta": None, "dev_total_tokens": None})

        fmt = cfg.evaluation.output_formats
        artifacts = {p.name: str(p) for p in write_rows(rows, self.run_dir, "h1_systems", fmt)}
        artifacts.update(
            {p.name: str(p) for p in write_rows(deltas, self.run_dir, "h1_deltas", fmt)}
        )
        metrics = {
            "split": cfg.dataset.split,
            "n_questions": len(table),
            "systems": rows,
            "deltas": deltas,
            "missing_profiles": missing_all,
            "label_merge_conflicts": len(merge.conflicts),
            "prices_usd_per_mtok": prices,
        }
        primary = [d for d in deltas if d["primary"]]
        summary = "; ".join(
            f"{d['system']} {d['delta_f1']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] "
            f"{d['verdict']}"
            for d in primary
        )
        if any(missing_all.values()):
            return ExperimentResult(
                RunStatus.FAILED,
                f"questions without a stored profile: {missing_all} (profile the split first)",
                artifacts=artifacts,
                metrics=metrics,
            )
        return ExperimentResult(RunStatus.SUCCEEDED, summary, artifacts=artifacts, metrics=metrics)


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
    QueryProfilingExperiment,
    RouterTrainExperiment,
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

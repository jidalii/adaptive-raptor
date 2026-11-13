"""Experiment-type-specific (cross-section) validation.

Schema validation (types, ranges, unknown keys) happens in the pydantic models. This
module checks that a valid config is also *meaningful* for its experiment type, before
any expensive work starts. It never touches the network or the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_raptor.config.schema import (
    ExperimentConfig,
    ExperimentType,
    PICChunking,
)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_optimizer_common(cfg: ExperimentConfig, r: ValidationReport) -> None:
    opt = cfg.optimizer
    catalog_ids = {c.id for c in cfg.retrieval.catalog()}
    if opt.fallback.id not in catalog_ids:
        r.errors.append(
            f"optimizer.fallback {opt.fallback.id} is not in the retrieval catalog "
            f"{sorted(catalog_ids)}"
        )
    if len(catalog_ids) < 2:
        r.errors.append("optimizer requires at least 2 catalog configurations")
    targets = (opt.theta, opt.target_mean_tokens, opt.target_config, opt.checkpoint)
    if all(t is None for t in targets):
        r.errors.append(
            "optimizer needs one of: theta (fixed), target_mean_tokens or target_config "
            "(calibrate on dev), or checkpoint (pre-trained and calibrated)"
        )
    if opt.target_config is not None and opt.target_config not in catalog_ids:
        r.errors.append(f"optimizer.target_config {opt.target_config} is not in the catalog")


def _check_raptor_baseline(cfg: ExperimentConfig, r: ValidationReport) -> None:
    if len(cfg.chunking.variants) != 1:
        r.errors.append("raptor_baseline expects exactly one chunking variant")
    if cfg.optimizer.enabled:
        r.errors.append("raptor_baseline must have optimizer.enabled: false")


def _check_h1(cfg: ExperimentConfig, r: ValidationReport) -> None:
    opt = cfg.optimizer
    if not opt.enabled:
        r.errors.append("h1_optimizer requires optimizer.enabled: true")
    else:
        _check_optimizer_common(cfg, r)
    # H1 is a replay over frozen routers: nothing is fitted or selected on this split
    if opt.checkpoint is None:
        r.errors.append("h1_optimizer needs optimizer.checkpoint (a frozen router_train output)")
    if not opt.eval_labels:
        r.errors.append("h1_optimizer needs optimizer.eval_labels (catalog labels of the split)")
    if opt.uses_profiler and not opt.profiler.profiles_path:
        r.errors.append("h1_optimizer needs optimizer.profiler.profiles_path for the split")
    if cfg.dataset.split == "train":
        r.errors.append("h1_optimizer never evaluates on the train split")
    if "best_fixed" not in cfg.evaluation.baselines:
        r.errors.append("h1_optimizer requires 'best_fixed' in evaluation.baselines (iso-budget)")
    if "answer_f1" not in cfg.evaluation.metrics:
        r.errors.append("h1_optimizer requires 'answer_f1' in evaluation.metrics")
    if len(cfg.chunking.variants) != 1:
        r.warnings.append("h1_optimizer uses only the first chunking variant")


def _check_h2(cfg: ExperimentConfig, r: ValidationReport) -> None:
    variants = cfg.chunking.variants
    pic = [v for v in variants if isinstance(v, PICChunking)]
    non_pic = [v for v in variants if not isinstance(v, PICChunking)]
    if not pic:
        r.errors.append("h2_chunking requires at least one PIC variant")
    if not non_pic:
        r.errors.append("h2_chunking requires at least one non-PIC control variant")
    if not cfg.evaluation.oracle.enabled:
        r.errors.append("h2_chunking requires evaluation.oracle.enabled: true")
    if cfg.evaluation.matched_budget is None:
        r.errors.append("h2_chunking requires evaluation.matched_budget (pre-set before test)")
    elif cfg.evaluation.matched_budget not in cfg.retrieval.budgets:
        r.errors.append(
            f"evaluation.matched_budget={cfg.evaluation.matched_budget} is not one of "
            f"retrieval.budgets {cfg.retrieval.budgets}"
        )
    if cfg.evaluation.equivalence_margin is None:
        r.warnings.append("no equivalence_margin: ties can only be reported as inconclusive")
    if cfg.dataset.split == "test" and cfg.chunking.primary_comparator is None:
        r.errors.append(
            "on the test split, chunking.primary_comparator must be fixed (selected on dev)"
        )


def _check_ablation(cfg: ExperimentConfig, r: ValidationReport) -> None:
    if not cfg.optimizer.enabled:
        r.errors.append("strategy_budget_ablation requires optimizer.enabled: true")
    else:
        _check_optimizer_common(cfg, r)
    if not cfg.evaluation.ablation.enabled:
        r.errors.append("strategy_budget_ablation requires evaluation.ablation.enabled: true")
    if len(cfg.retrieval.strategies) < 2 or len(cfg.retrieval.budgets) < 2:
        r.errors.append("the 2x2 ablation needs >= 2 strategies and >= 2 budgets")


def _check_oracle_interaction(cfg: ExperimentConfig, r: ValidationReport) -> None:
    if len(cfg.chunking.variants) < 2:
        r.errors.append("oracle_interaction compares >= 2 chunking variants")
    if not cfg.evaluation.oracle.enabled:
        r.errors.append("oracle_interaction requires evaluation.oracle.enabled: true")
    if cfg.optimizer.enabled:
        r.warnings.append("oracle_interaction does not use the router; optimizer is ignored")


def _check_query_profiling(cfg: ExperimentConfig, r: ValidationReport) -> None:
    if not cfg.optimizer.uses_profiler:
        r.errors.append("query_profiling requires 'llm_profile' in optimizer.features")
    prof = cfg.optimizer.profiler
    if prof.temperature != 0:
        r.warnings.append("profiler temperature != 0: profiles are samples, not reproducible")
    if prof.use_logprobs and prof.temperature == 0:
        r.warnings.append(
            "profiler.use_logprobs at temperature 0: DeepSeek logprobs are post-temperature "
            "(all 0.0), so confidence is unavailable and the confidence fallback is inactive"
        )


def _check_router_train(cfg: ExperimentConfig, r: ValidationReport) -> None:
    opt = cfg.optimizer
    _check_optimizer_common(cfg, r)
    if opt.model_type not in ("llm_profile_table", "llm_profile_rules"):
        r.errors.append(
            f"router_train supports the Plan A models only (got {opt.model_type}); "
            "Plan B is built only if Plan A shows no improvement on dev"
        )
    if not opt.train_labels and opt.model_type == "llm_profile_table":
        r.errors.append("router_train needs optimizer.train_labels (labels.jsonl on train)")
    if not opt.dev_labels:
        r.errors.append("router_train needs optimizer.dev_labels (labels.jsonl on dev)")
    if not opt.profiler.profiles_path:
        r.errors.append("router_train needs optimizer.profiler.profiles_path")
    if cfg.dataset.split == "test":
        r.errors.append("router_train never uses the test split")
    if cfg.evaluation.budget_matching not in ("best_within_budget", "nearest_higher"):
        r.errors.append(
            "router_train freezes comparators by total cost: evaluation.budget_matching must "
            "be best_within_budget (primary) or nearest_higher"
        )


_TYPE_CHECKS = {
    ExperimentType.RAPTOR_BASELINE: _check_raptor_baseline,
    ExperimentType.H1_OPTIMIZER: _check_h1,
    ExperimentType.H2_CHUNKING: _check_h2,
    ExperimentType.STRATEGY_BUDGET_ABLATION: _check_ablation,
    ExperimentType.ORACLE_INTERACTION: _check_oracle_interaction,
    ExperimentType.QUERY_PROFILING: _check_query_profiling,
    ExperimentType.ROUTER_TRAIN: _check_router_train,
}


def validate_config(cfg: ExperimentConfig) -> ValidationReport:
    report = ValidationReport()
    _TYPE_CHECKS[cfg.experiment.type](cfg, report)

    # Generic checks that apply to every experiment type.
    gen = cfg.generation
    if gen.provider not in ("deepseek", "openai", "mock"):
        report.errors.append(f"generation.provider '{gen.provider}' is not implemented")
    if cfg.raptor.embedding.provider not in ("sbert", "openai", "mock"):
        report.errors.append(
            f"raptor.embedding.provider '{cfg.raptor.embedding.provider}' is not supported"
        )
    if cfg.raptor.summarization.provider not in ("deepseek", "openai", "mock"):
        report.errors.append(
            f"raptor.summarization.provider '{cfg.raptor.summarization.provider}' is not supported"
        )
    if gen.temperature != 0:
        report.warnings.append("generation.temperature != 0: labels will be less reproducible")
    if gen.thinking:
        report.warnings.append(
            "generation.thinking=true: hidden reasoning tokens are billed and count against "
            "max_output_tokens; labels and costs are not comparable with non-thinking runs"
        )
    mocks = [
        name
        for name, provider in (
            ("generation", gen.provider),
            ("raptor.embedding", cfg.raptor.embedding.provider),
            ("raptor.summarization", cfg.raptor.summarization.provider),
        )
        if provider == "mock"
    ]
    if mocks or cfg.dataset.name == "qasper_tiny":
        report.warnings.append(
            f"offline smoke configuration (mock: {', '.join(mocks) or 'none'}; "
            f"dataset: {cfg.dataset.name}): results are not meaningful"
        )
    if cfg.dataset.name == "narrativeqa" and not cfg.generation.prompt_template.startswith(
        "narrativeqa"
    ):
        report.warnings.append(
            f"dataset narrativeqa with prompt template '{cfg.generation.prompt_template}' "
            "(QASPER prompts ask for 'Unanswerable'/'Yes'/'No' answers)"
        )
    if cfg.evaluation.oracle.enabled and not cfg.evaluation.oracle.cross_fit:
        report.warnings.append("oracle without cross-fitting is biased upward (winner's curse)")
    if cfg.dataset.split == "test" and cfg.optimizer.enabled and cfg.optimizer.checkpoint is None:
        report.errors.append(
            "on the test split the router must be frozen: set optimizer.checkpoint"
        )
    return report

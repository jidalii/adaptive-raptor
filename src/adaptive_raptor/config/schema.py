"""Typed configuration schema for all experiments.

Field-level constraints live here. Cross-section rules that depend on the experiment
type (e.g. "H2 needs a PIC variant") live in :mod:`adaptive_raptor.config.validation`.

Every section forbids unknown keys so that typos in YAML fail loudly.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- experiment


class ExperimentType(StrEnum):
    RAPTOR_BASELINE = "raptor_baseline"
    H1_OPTIMIZER = "h1_optimizer"
    H2_CHUNKING = "h2_chunking"
    STRATEGY_BUDGET_ABLATION = "strategy_budget_ablation"
    ORACLE_INTERACTION = "oracle_interaction"


class ExperimentSection(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.\-]+$", description="Used in output paths.")
    type: ExperimentType
    seed: int = 0
    output_dir: Path = Path("outputs")
    run_label: str = Field(
        default="A", pattern=r"^[A-Za-z0-9_\-]+$", description="Cross-fitting run id (A/B)."
    )
    description: str = ""


# --------------------------------------------------------------------------- dataset


class DatasetSection(StrictModel):
    name: Literal["qasper", "qasper_tiny", "quality"] = Field(
        default="qasper",
        description="'qasper_tiny' is a bundled offline fixture for smoke tests (not for results).",
    )
    version: str = "v0.3"
    path: Path | None = Field(
        default=None,
        description="Data root (contains raw/ and processed/). "
        "None = $ADAPTIVE_RAPTOR_DATA_DIR or ./data.",
    )
    split: Literal["train", "dev", "test"] = "dev"
    max_documents: PositiveInt | None = None
    max_questions: PositiveInt | None = None
    sample_seed: int = Field(default=0, description="Seed for deterministic document subsets.")
    doc_ids_file: Path | None = Field(
        default=None, description="Keep only these paper ids (one per line)."
    )
    exclude_doc_ids_file: Path | None = Field(
        default=None, description="Drop these paper ids (e.g. the pilot set)."
    )


# --------------------------------------------------------------------------- chunking


class _ChunkingBase(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.\-]+$", description="Index-variant identifier.")


class FixedChunking(_ChunkingBase):
    """Original RAPTOR fixed-size, sentence-preserving chunking (raptor.utils.split_text)."""

    method: Literal["fixed"] = "fixed"
    chunk_size: PositiveInt = 100
    overlap: NonNegativeInt = 0

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> FixedChunking:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class PICChunking(_ChunkingBase):
    """PIC-capped: summary-guided chunking with a per-document mean threshold."""

    method: Literal["pic"] = "pic"
    max_tokens: PositiveInt = 300
    min_tokens: PositiveInt | None = Field(
        default=None, description="Merge runs shorter than this (ablated). None = no merging."
    )
    threshold: Literal["mean"] = "mean"
    summary_source: Literal["llm", "raptor_root"] = "llm"
    summary_max_tokens: PositiveInt = 300

    @model_validator(mode="after")
    def _min_lt_max(self) -> PICChunking:
        if self.min_tokens is not None and self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")
        return self


class PermutedBoundaryChunking(_ChunkingBase):
    """Control: PIC chunk lengths per document, boundaries at random sentence positions."""

    method: Literal["permuted_boundaries"] = "permuted_boundaries"
    source_variant: str = Field(description="Name of the PIC variant whose lengths are reused.")
    max_tokens: PositiveInt = 300
    permutation_seed: int = 0


class ParagraphChunking(_ChunkingBase):
    """Control: author paragraph boundaries, split at max_tokens."""

    method: Literal["paragraph"] = "paragraph"
    max_tokens: PositiveInt = 300


ChunkingVariant = Annotated[
    FixedChunking | PICChunking | PermutedBoundaryChunking | ParagraphChunking,
    Field(discriminator="method"),
]


class ChunkingSection(StrictModel):
    variants: list[ChunkingVariant] = Field(
        default_factory=lambda: [FixedChunking(name="f100")], min_length=1
    )
    primary_comparator: str | None = Field(
        default=None,
        description="H2: the non-PIC control selected on dev and frozen before test. "
        "None = not yet selected (allowed on dev).",
    )

    @model_validator(mode="after")
    def _check_names_and_refs(self) -> ChunkingSection:
        names = [v.name for v in self.variants]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate chunking variant names: {sorted(dupes)}")
        by_name = {v.name: v for v in self.variants}
        for v in self.variants:
            if isinstance(v, PermutedBoundaryChunking):
                src = by_name.get(v.source_variant)
                if src is None or not isinstance(src, PICChunking):
                    raise ValueError(
                        f"variant '{v.name}': source_variant '{v.source_variant}' "
                        "must name a PIC variant in this config"
                    )
        if self.primary_comparator is not None:
            comp = by_name.get(self.primary_comparator)
            if comp is None:
                raise ValueError(f"primary_comparator '{self.primary_comparator}' is not a variant")
            if isinstance(comp, PICChunking):
                raise ValueError("primary_comparator must be a non-PIC control")
        return self

    def get(self, name: str) -> ChunkingVariant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(name)


# --------------------------------------------------------------------------- RAPTOR


class ModelRef(StrictModel):
    provider: Literal["openai", "deepseek", "anthropic", "sbert", "mock"]
    model: str


class ClusteringParams(StrictModel):
    """Maps onto raptor.ClusterTreeConfig / RAPTOR_Clustering arguments."""

    reduction_dimension: PositiveInt = 10
    threshold: float = Field(default=0.1, gt=0, le=1)
    max_length_in_cluster: PositiveInt = 3500


class RaptorSection(StrictModel):
    embedding: ModelRef = Field(
        default_factory=lambda: ModelRef(
            provider="sbert", model="sentence-transformers/multi-qa-mpnet-base-cos-v1"
        )
    )
    summarization: ModelRef = Field(
        default_factory=lambda: ModelRef(provider="deepseek", model="deepseek-flash")
    )
    summarization_length: PositiveInt = 100
    num_layers: PositiveInt = 5
    tokenizer: Literal["cl100k_base"] = "cl100k_base"
    deterministic_umap: bool = Field(
        default=True,
        description="Pass random_state=seed to upstream UMAP during builds. Upstream leaves it "
        "unset, so trees differ between processes. True = reproducible, single-threaded UMAP.",
    )
    clustering: ClusteringParams = Field(default_factory=ClusteringParams)
    seeds: list[int] = Field(default_factory=lambda: [224], min_length=1)
    index_dir: Path = Path("indexes")
    cache_indexes: bool = True

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("seeds must be unique")
        return v


# --------------------------------------------------------------------------- retrieval


class Strategy(StrEnum):
    LEAF_ONLY = "leaf_only"
    SUMMARY_ONLY = "summary_only"
    COLLAPSED = "collapsed"


class ConfigRef(StrictModel):
    """A single (strategy, budget) configuration from the catalog."""

    strategy: Strategy
    budget: PositiveInt

    @property
    def id(self) -> str:
        return f"{self.strategy.value}@{self.budget}"


class RaptorOriginalParams(StrictModel):
    """Arguments for upstream ``TreeRetriever.retrieve`` (the untouched RAPTOR retrieval)."""

    collapse_tree: bool = True
    top_k: PositiveInt = 10
    max_tokens: PositiveInt = 2000


class RetrievalSection(StrictModel):
    include_raptor_original: bool = Field(
        default=False,
        description="Also run upstream TreeRetriever.retrieve as config id 'raptor_original'.",
    )
    raptor_original: RaptorOriginalParams = Field(default_factory=RaptorOriginalParams)
    strategies: list[Strategy] = Field(default_factory=lambda: list(Strategy), min_length=1)
    budgets: list[PositiveInt] = Field(default_factory=lambda: [400, 1000, 2000], min_length=1)
    similarity: Literal["cosine"] = "cosine"
    backend: Literal["numpy", "faiss"] = "numpy"
    fill_rule: Literal["greedy"] = "greedy"
    deduplicate: bool = True

    @field_validator("strategies")
    @classmethod
    def _unique_strategies(cls, v: list[Strategy]) -> list[Strategy]:
        if len(set(v)) != len(v):
            raise ValueError("strategies must be unique")
        return v

    @field_validator("budgets")
    @classmethod
    def _sorted_unique_budgets(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("budgets must be unique")
        return sorted(v)

    def catalog(self) -> list[ConfigRef]:
        return [ConfigRef(strategy=s, budget=b) for s in self.strategies for b in self.budgets]


# --------------------------------------------------------------------------- optimizer


class OptimizerSection(StrictModel):
    enabled: bool = False
    model_type: Literal["logistic_regression", "ordinal_min_budget"] = "logistic_regression"
    features: list[Literal["query_embedding", "lexical"]] = Field(
        default_factory=lambda: ["query_embedding", "lexical"], min_length=1
    )
    pca_dim: PositiveInt | None = 32
    regularization_c: float = Field(default=1.0, gt=0)
    label_epsilon: float = Field(
        default=0.05, ge=0, le=1, description="Acceptable if F1 >= per-query best F1 - epsilon."
    )
    theta: float | None = Field(
        default=None, ge=0, le=1, description="Fixed acceptability threshold; None = calibrate."
    )
    target_mean_tokens: PositiveInt | None = Field(
        default=None, description="Average-token target used to calibrate theta on dev."
    )
    fallback: ConfigRef = Field(
        default_factory=lambda: ConfigRef(strategy=Strategy.COLLAPSED, budget=2000)
    )
    checkpoint: Path | None = None

    @model_validator(mode="after")
    def _pca_needs_embedding(self) -> OptimizerSection:
        if self.pca_dim is not None and "query_embedding" not in self.features:
            raise ValueError("pca_dim is set but 'query_embedding' is not in features")
        return self


# --------------------------------------------------------------------------- generation


class GenerationSection(StrictModel):
    provider: Literal["deepseek", "openai", "anthropic", "mock"] = "deepseek"
    model: str = "deepseek-flash"
    thinking: bool = False
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: PositiveInt = 200
    prompt_template: str = Field(
        default="qasper_default",
        description="Name of a built-in template or a path to a template file.",
    )
    api_key_env: str | None = Field(
        default=None, description="Override the provider's default key env var."
    )
    base_url_env: str | None = Field(
        default=None, description="Override the provider's default base-URL env var."
    )
    request_timeout_s: float = Field(default=60.0, gt=0)
    price_input_per_mtok: float | None = Field(
        default=None, ge=0, description="USD per 1M input tokens (cost meter only)."
    )
    price_output_per_mtok: float | None = Field(
        default=None, ge=0, description="USD per 1M output tokens (cost meter only)."
    )


# --------------------------------------------------------------------------- evaluation


class OracleSettings(StrictModel):
    enabled: bool = False
    cross_fit: bool = True
    second_run_prompt: str = Field(
        default="qasper_paraphrase", description="Prompt template for cross-fitting run B."
    )


class AblationSettings(StrictModel):
    enabled: bool = False
    cells: list[Literal["fixed_fixed", "adaptive_strategy", "adaptive_budget", "joint"]] = Field(
        default_factory=lambda: ["fixed_fixed", "adaptive_strategy", "adaptive_budget", "joint"]
    )


class EvaluationSection(StrictModel):
    metrics: list[Literal["answer_f1", "evidence_recall", "context_tokens", "latency"]] = Field(
        default_factory=lambda: ["answer_f1", "context_tokens", "latency"], min_length=1
    )
    baselines: list[Literal["best_fixed", "raptor_default", "full_collapsed_max_budget"]] = Field(
        default_factory=lambda: ["best_fixed", "raptor_default"]
    )
    budget_matching: Literal["conservative_at_least", "interpolate"] = "conservative_at_least"
    matched_budget: PositiveInt | None = Field(
        default=None, description="H2: pre-set budget at which oracle F1 is compared."
    )
    oracle: OracleSettings = Field(default_factory=OracleSettings)
    ablation: AblationSettings = Field(default_factory=AblationSettings)
    bootstrap_samples: PositiveInt = 2000
    bootstrap_unit: Literal["document"] = "document"
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    equivalence_margin: float | None = Field(
        default=0.02, gt=0, description="On the 0-1 F1 scale (0.02 = 2 F1 points)."
    )
    save_contexts: bool = Field(default=True, description="Store retrieved context text in traces.")
    output_formats: list[Literal["csv", "json", "markdown"]] = Field(
        default_factory=lambda: ["csv", "json"], min_length=1
    )


# --------------------------------------------------------------------------- runtime


class RuntimeSection(StrictModel):
    concurrency: PositiveInt = 4
    batch_size: PositiveInt = 16
    max_retries: NonNegativeInt = 3
    use_cache: bool = True
    cache_dir: Path = Path(".cache")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# --------------------------------------------------------------------------- root


class ExperimentConfig(StrictModel):
    schema_version: Literal[1] = 1
    experiment: ExperimentSection
    dataset: DatasetSection = Field(default_factory=DatasetSection)
    chunking: ChunkingSection = Field(default_factory=ChunkingSection)
    raptor: RaptorSection = Field(default_factory=RaptorSection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    optimizer: OptimizerSection = Field(default_factory=OptimizerSection)
    generation: GenerationSection = Field(default_factory=GenerationSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    runtime: RuntimeSection = Field(default_factory=RuntimeSection)

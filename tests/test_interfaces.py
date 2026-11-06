"""Research interfaces exist, placeholders are marked, and stubs fail loudly."""

import pytest

from adaptive_raptor.chunking import (
    Chunker,
    FixedSizeChunker,
    ParagraphChunker,
    PermutedBoundaryChunker,
    PICChunker,
    build_chunker,
)
from adaptive_raptor.config.schema import (
    ConfigRef,
    ExperimentType,
    GenerationSection,
    OptimizerSection,
    ParagraphChunking,
    PermutedBoundaryChunking,
    PICChunking,
    Strategy,
)
from adaptive_raptor.data import Document, TestSplitLeakageError, guard_split
from adaptive_raptor.evaluation.research import within_equivalence_margin
from adaptive_raptor.experiments import RunStatus, get_experiment_class, registered_types
from adaptive_raptor.generation import MockGenerator, build_generator, get_template, render
from adaptive_raptor.models.credentials import MissingCredentialsError
from adaptive_raptor.optimizer import (
    AcceptabilityModel,
    CostModel,
    FeatureExtractor,
    PlanSelector,
    Router,
    RouterPolicy,
    calibrate_theta,
    train_router,
)
from adaptive_raptor.profiling import acceptability_labels
from adaptive_raptor.stub import ensure_implemented, is_stub

DOC = Document(doc_id="d1", text="One. Two.")
CATALOG = [ConfigRef(strategy=Strategy.LEAF_ONLY, budget=400)]


@pytest.mark.parametrize(
    ("cfg", "cls"),
    [
        (PICChunking(name="p"), PICChunker),
        (PermutedBoundaryChunking(name="x", source_variant="p"), PermutedBoundaryChunker),
        (ParagraphChunking(name="g"), ParagraphChunker),
    ],
)
def test_placeholder_chunkers(cfg, cls):
    chunker = build_chunker(cfg)
    assert isinstance(chunker, cls) and isinstance(chunker, Chunker)
    assert is_stub(chunker)
    with pytest.raises(NotImplementedError):
        ensure_implemented(chunker)
    with pytest.raises(NotImplementedError):
        chunker.chunk(DOC)


def test_fixed_chunker_is_not_a_stub():
    assert not is_stub(FixedSizeChunker)


def test_optimizer_components_are_stubs():
    opt = OptimizerSection()
    for obj in (
        FeatureExtractor(opt),
        AcceptabilityModel(opt, ["leaf_only@400"]),
        CostModel(),
        PlanSelector(CATALOG, CostModel(), CATALOG[0]),
        Router(opt, CATALOG),
    ):
        assert is_stub(obj)
    with pytest.raises(NotImplementedError):
        RouterPolicy(Router(opt, CATALOG))  # rejected before any work
    with pytest.raises(NotImplementedError):
        acceptability_labels([], 0.05)


def test_fitting_on_test_split_is_refused():
    with pytest.raises(TestSplitLeakageError):
        guard_split("test", "train")
    guard_split("test", "evaluate")
    guard_split("dev", "calibrate")
    with pytest.raises(TestSplitLeakageError):
        calibrate_theta("test", [], None, 1000)
    with pytest.raises(TestSplitLeakageError):
        train_router(None, None, "test")


def test_research_runners_report_not_implemented():
    research = set(ExperimentType) - {ExperimentType.RAPTOR_BASELINE}
    assert set(registered_types()) == set(ExperimentType)
    for t in research:
        result = get_experiment_class(t)(config=None, run_dir=None).run()
        assert result.status is RunStatus.NOT_IMPLEMENTED
        assert result.planned_steps


def test_generation_factory_and_prompts(no_api_keys):
    assert isinstance(build_generator(GenerationSection(provider="mock")), MockGenerator)
    prompt = render(get_template("qasper_default"), "Q?", ["ctx"])
    assert "Q?" in prompt and "ctx" in prompt
    real = build_generator(GenerationSection(), use_cache=False)
    with pytest.raises(MissingCredentialsError, match="DEEPSEEK_API_KEY"):
        real.generate("Q?", ["ctx"])  # fails before any network call


def test_mock_generator_unanswerable_without_context():
    assert MockGenerator().generate("Q?", []).text == "Unanswerable"


def test_equivalence_margin_helper():
    assert within_equivalence_margin((-0.01, 0.015), 0.02)
    assert not within_equivalence_margin((-0.03, 0.01), 0.02)

"""Every module imports, and the config/CLI layer stays free of heavy dependencies."""

import importlib
import subprocess
import sys

import pytest

MODULES = [
    "adaptive_raptor",
    "adaptive_raptor.run",
    "adaptive_raptor.paths",
    "adaptive_raptor.stub",
    "adaptive_raptor.cache",
    "adaptive_raptor.policy",
    "adaptive_raptor.pipeline",
    "adaptive_raptor.config",
    "adaptive_raptor.experiments",
    "adaptive_raptor.experiments.runner",
    "adaptive_raptor.data",
    "adaptive_raptor.data.qasper",
    "adaptive_raptor.data.download",
    "adaptive_raptor.data.subsets",
    "adaptive_raptor.chunking",
    "adaptive_raptor.indexing",
    "adaptive_raptor.retrieval",
    "adaptive_raptor.generation",
    "adaptive_raptor.models",
    "adaptive_raptor.models.embedding",
    "adaptive_raptor.models.summarization",
    "adaptive_raptor.evaluation.metrics",
    "adaptive_raptor.evaluation.traces",
    "adaptive_raptor.evaluation.research",
    "adaptive_raptor.profiling",
    "adaptive_raptor.optimizer",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_validation_path_does_not_import_heavy_deps():
    code = (
        "import sys, adaptive_raptor.run, adaptive_raptor.pipeline, adaptive_raptor.data.download;"
        "from adaptive_raptor.config import load_config, validate_config;"
        "validate_config(load_config('configs/experiments/h2.yaml'));"
        "bad=[m for m in ('torch','sentence_transformers','raptor','umap') if m in sys.modules];"
        "print(','.join(bad))"
    )
    from tests.conftest import ROOT

    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert out.stdout.strip() == ""

"""End-to-end CLI runs of the baseline pipeline (offline, mock providers)."""

import json

import pytest
import yaml

from adaptive_raptor.run import EXIT_INVALID, EXIT_NOT_IMPLEMENTED, EXIT_OK, main
from tests.conftest import ROOT, SMOKE, smoke_overrides

pytestmark = pytest.mark.heavy

TRACE_FIELDS = {
    "question_id",
    "doc_id",
    "index_variant",
    "seed",
    "config_id",
    "run_label",
    "question",
    "index_key_hash",
    "index_cache_hit",
    "node_ids",
    "levels",
    "contexts",
    "sufficiency",
    "prompt_cache_hit_tokens",
    "context_tokens",
    "retrieval_ms",
    "answer",
    "answer_rouge_l",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "router_model",
    "route_fallback_reason",
    "profile",
    "profile_confidence",
    "router_input_tokens",
    "router_output_tokens",
    "router_ms",
    "generation_ms",
    "generation_cache_hit",
    "model_version",
    "answer_f1",
    "answer_type",
    "error",
}


def _args(tmp_path, *extra):
    out = []
    for o in [*smoke_overrides(tmp_path), *extra]:
        out += ["--set", o]
    return ["--config", str(SMOKE), *out]


def _run_dirs(tmp_path):
    return sorted((tmp_path / "out" / "smoke_mock").iterdir())


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    assert main(_args(tmp)) == EXIT_OK
    return tmp, _run_dirs(tmp)[0]


def test_artifacts_and_status(first_run):
    _, run = first_run
    for name in (
        "resolved_config.yaml",
        "run_metadata.json",
        "status.json",
        "traces.jsonl",
        "indexes.jsonl",
        "labels.jsonl",
        "summary.csv",
        "summary.json",
    ):
        assert (run / name).is_file(), name
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "succeeded" and status["metrics"]["trace_errors"] == 0
    meta = json.loads((run / "run_metadata.json").read_text())
    for key in (
        "seed",
        "git",
        "dataset",
        "raptor",
        "generator",
        "packages",
        "upstream_raptor_commit",
    ):
        assert key in meta
    resolved = yaml.safe_load((run / "resolved_config.yaml").read_text())
    assert resolved["dataset"]["name"] == "qasper_tiny"


def test_trace_schema_and_counts(first_run):
    _, run = first_run
    traces = [json.loads(line) for line in (run / "traces.jsonl").read_text().splitlines()]
    assert traces and all(set(t) == TRACE_FIELDS for t in traces)
    configs = {t["config_id"] for t in traces}
    assert "raptor_original" in configs and len(configs) == 10  # 3x3 catalog + original
    assert len(traces) == 6 * 10  # 6 dev fixture questions
    assert all(t["error"] is None and 0 <= t["answer_f1"] <= 1 for t in traces)
    summary = json.loads((run / "summary.json").read_text())
    assert {r["config_id"] for r in summary} == configs


def test_second_run_reuses_indexes(first_run):
    tmp, _ = first_run
    assert main(_args(tmp)) == EXIT_OK
    latest = _run_dirs(tmp)[-1]
    idx = [json.loads(line) for line in (latest / "indexes.jsonl").read_text().splitlines()]
    assert idx and all(r["cache_hit"] for r in idx)


def test_pic_variant_stops_before_any_work(tmp_path):
    args = _args(tmp_path, "chunking.variants.0={name: pic, method: pic}")
    assert main(args) == EXIT_NOT_IMPLEMENTED
    (run,) = _run_dirs(tmp_path)
    assert json.loads((run / "status.json").read_text())["status"] == "not_implemented"
    assert not (tmp_path / "idx").exists()  # no index was built


def test_real_baseline_without_key_fails_preflight(tmp_path, no_api_keys, capsys):
    cfg = ROOT / "configs/experiments/baseline.yaml"
    args = [
        "--config",
        str(cfg),
        "--set",
        f"experiment.output_dir={tmp_path / 'o'}",
        "--set",
        f"dataset.path={tmp_path / 'nodata'}",
    ]
    assert main(args) == EXIT_INVALID
    err = capsys.readouterr().err
    assert "DEEPSEEK_API_KEY" in err and "download_datasets" in err
    assert not (tmp_path / "o").exists()  # nothing written before preflight passes

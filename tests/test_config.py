import pytest

from adaptive_raptor.config import (
    ConfigError,
    ExperimentType,
    load_config,
    resolved_config_dict,
    validate_config,
)
from adaptive_raptor.config.loader import apply_overrides, config_from_dict
from tests.conftest import EXAMPLE_CONFIGS


def test_example_configs_exist():
    names = {p.stem for p in EXAMPLE_CONFIGS}
    assert {"baseline", "h1", "h2", "ablation", "oracle_interaction", "smoke_mock"} <= names


# Templates whose input paths are filled in by the user before running.
TEMPLATE_FILL = {
    "h1": [
        "optimizer.checkpoint=ckpt/router",
        "optimizer.eval_labels=[t/labels.jsonl]",
        "optimizer.profiler.profiles_path=[t/profiles.jsonl]",
    ],
    "router_train": [
        "optimizer.train_labels=[a/labels.jsonl]",
        "optimizer.dev_labels=[b/labels.jsonl]",
        "optimizer.profiler.profiles_path=[c/profiles.jsonl]",
    ],
}


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.stem)
def test_example_configs_validate(path):
    cfg = load_config(path, TEMPLATE_FILL.get(path.stem, []))
    report = validate_config(cfg)
    assert report.ok, report.errors


def test_router_train_template_requires_inputs():
    from tests.conftest import ROOT

    errors = validate_config(load_config(ROOT / "configs/experiments/router_train.yaml")).errors
    assert {"train_labels", "dev_labels", "profiles_path"} <= {
        k for k in ("train_labels", "dev_labels", "profiles_path") for e in errors if k in e
    }


def test_plan_b_model_is_rejected_by_router_train(minimal_config_dict):
    minimal_config_dict["experiment"]["type"] = "router_train"
    minimal_config_dict["optimizer"] = {
        "enabled": True,
        "model_type": "logistic_regression",
        "features": ["query_embedding"],
        "theta": 0.5,
        "train_labels": ["a"],
        "dev_labels": ["b"],
        "profiler": {"profiles_path": ["c"]},
    }
    errors = validate_config(config_from_dict(minimal_config_dict)).errors
    assert any("Plan A" in e for e in errors)


def test_defaults_fill_in(minimal_config_dict):
    cfg = config_from_dict(minimal_config_dict)
    assert cfg.experiment.type is ExperimentType.RAPTOR_BASELINE
    assert cfg.generation.temperature == 0.0
    assert cfg.generation.model == "deepseek-flash"
    assert [c.id for c in cfg.retrieval.catalog()][:2] == ["leaf_only@400", "leaf_only@1000"]
    assert len(cfg.retrieval.catalog()) == 9
    assert cfg.chunking.variants[0].method == "fixed"


def test_unknown_key_rejected(minimal_config_dict):
    minimal_config_dict["retrieval"] = {"budgts": [400]}
    with pytest.raises(ConfigError, match="budgts"):
        config_from_dict(minimal_config_dict)


def test_bad_enum_rejected(minimal_config_dict):
    minimal_config_dict["retrieval"] = {"strategies": ["tree_traversal"]}
    with pytest.raises(ConfigError):
        config_from_dict(minimal_config_dict)


def test_permuted_requires_pic_source(minimal_config_dict):
    minimal_config_dict["chunking"] = {
        "variants": [
            {"name": "f100", "method": "fixed"},
            {"name": "perm", "method": "permuted_boundaries", "source_variant": "f100"},
        ]
    }
    with pytest.raises(ConfigError, match="PIC variant"):
        config_from_dict(minimal_config_dict)


def test_overlap_must_be_smaller(minimal_config_dict):
    minimal_config_dict["chunking"] = {
        "variants": [{"name": "f", "method": "fixed", "chunk_size": 10, "overlap": 10}]
    }
    with pytest.raises(ConfigError, match="overlap"):
        config_from_dict(minimal_config_dict)


def test_budgets_sorted(minimal_config_dict):
    minimal_config_dict["retrieval"] = {"budgets": [2000, 400]}
    assert config_from_dict(minimal_config_dict).retrieval.budgets == [400, 2000]


def test_h1_requires_enabled_optimizer(minimal_config_dict):
    minimal_config_dict["experiment"]["type"] = "h1_optimizer"
    report = validate_config(config_from_dict(minimal_config_dict))
    assert not report.ok
    assert any("optimizer.enabled" in e for e in report.errors)


def test_fallback_must_be_in_catalog(minimal_config_dict):
    minimal_config_dict["experiment"]["type"] = "h1_optimizer"
    minimal_config_dict["optimizer"] = {
        "enabled": True,
        "theta": 0.5,
        "fallback": {"strategy": "collapsed", "budget": 9999},
    }
    report = validate_config(config_from_dict(minimal_config_dict))
    assert any("fallback" in e for e in report.errors)


def test_h2_test_split_needs_frozen_comparator():
    from tests.conftest import ROOT

    cfg = load_config(ROOT / "configs/experiments/h2.yaml", ["dataset.split=test"])
    report = validate_config(cfg)
    assert any("primary_comparator" in e for e in report.errors)


def test_test_split_router_must_be_frozen():
    from tests.conftest import ROOT

    cfg = load_config(ROOT / "configs/experiments/h1.yaml", ["dataset.split=test"])
    assert any("checkpoint" in e for e in validate_config(cfg).errors)
    fill = [x for x in TEMPLATE_FILL["h1"] if "checkpoint" not in x]
    cfg = load_config(ROOT / "configs/experiments/h1.yaml", fill)
    assert any("checkpoint" in e for e in validate_config(cfg).errors)
    train = load_config(
        ROOT / "configs/experiments/h1.yaml", [*TEMPLATE_FILL["h1"], "dataset.split=train"]
    )
    assert any("train split" in e for e in validate_config(train).errors)


def test_overrides_nested_and_indexed():
    data = {"a": {"b": 1}, "l": [{"x": 1}]}
    apply_overrides(data, ["a.b=2", "a.c=[1, 2]", "l.0.x=true", "new.key=hello"])
    assert data == {"a": {"b": 2, "c": [1, 2]}, "l": [{"x": True}], "new": {"key": "hello"}}


def test_bad_override_rejected():
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equals_sign"])


def test_resolved_dict_roundtrip(minimal_config_dict):
    cfg = config_from_dict(minimal_config_dict)
    again = config_from_dict(resolved_config_dict(cfg))
    assert again == cfg


def test_unsupported_provider_rejected(minimal_config_dict):
    minimal_config_dict["generation"] = {"provider": "anthropic"}
    report = validate_config(config_from_dict(minimal_config_dict))
    assert any("not implemented" in e for e in report.errors)


def test_smoke_config_warns_not_meaningful():
    from tests.conftest import SMOKE

    report = validate_config(load_config(SMOKE))
    assert report.ok and any("not meaningful" in w for w in report.warnings)

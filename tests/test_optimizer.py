"""Plan A router: profiler, models, selection, calibration, checkpoints, experiments.

All offline: the LLM profiler is exercised through a fake OpenAI client.
"""

import json
import math
from types import SimpleNamespace

import pytest

from adaptive_raptor.config.schema import ConfigRef, OptimizerSection, ProfilerSection, Strategy
from adaptive_raptor.data import TestSplitLeakageError
from adaptive_raptor.data.records import Document, Paragraph
from adaptive_raptor.optimizer import (
    CostModel,
    PlanSelector,
    ProfileTableModel,
    Router,
    RouterCheckpoint,
    RuleMapModel,
    calibrate_theta,
    load_checkpoint,
    save_checkpoint,
    theta_sweep,
)
from adaptive_raptor.optimizer.profiler import (
    CachedProfiler,
    LLMProfiler,
    MockProfiler,
    ProfileResult,
    ProfileStore,
    QueryProfile,
    field_confidence,
    parse_profile,
    render_prompt,
)
from adaptive_raptor.profiling import LabelRecord
from tests.conftest import ROOT as _ROOT  # noqa: E402

ROOT_CFG = _ROOT / "configs/experiments"
L, S, C = Strategy.LEAF_ONLY, Strategy.SUMMARY_ONLY, Strategy.COLLAPSED
CATALOG = [ConfigRef(strategy=s, budget=b) for s in (L, S, C) for b in (400, 1000, 2000)]
IDS = [c.id for c in CATALOG]
FALLBACK = ConfigRef(strategy=C, budget=2000)


def prof(form="short_span", gran="specific_detail", n=1, cross=False):
    return QueryProfile(answer_form=form, granularity=gran, n_facts=n, cross_section=cross)


# --------------------------------------------------------------------------- parsing


def test_parse_profile_accepts_valid_and_rejects_bad():
    ok = (
        '{"answer_form": "yes_no", "granularity": "whole_paper", "n_facts": 2, '
        '"cross_section": true}'
    )
    assert parse_profile(ok) == prof("yes_no", "whole_paper", 2, True)
    assert parse_profile("Sure! " + ok + " done").answer_form == "yes_no"
    for bad in (
        "not json",
        '{"answer_form": "maybe", "granularity": "whole_paper", "n_facts": 1, '
        '"cross_section": false}',
        ok[:-1] + ', "extra": 1}',
        ok.replace('"n_facts": 2', '"n_facts": 9'),
        "[1, 2]",
    ):
        with pytest.raises(ValueError):
            parse_profile(bad)


def test_field_confidence_reads_first_value_token():
    toks = [
        ('{"', 0.0),
        ("answer", 0.0),
        ('_form": "', 0.0),
        ("yes", math.log(0.8)),
        ("_no", 0.0),
        ('", "granularity": "', 0.0),
        ("specific", math.log(0.5)),
        ('_detail", ', 0.0),
        ('"n_facts": ', 0.0),
        ("1", math.log(0.9)),
        (', "cross_section": ', 0.0),
        ("false", 0.0),
        ("}", 0.0),
    ]
    assert field_confidence(toks) == pytest.approx(0.8 * 0.5 * 0.9)
    assert field_confidence(toks[:6]) is None  # a field is missing
    assert field_confidence(None) is None
    assert field_confidence([(t, 0.0) for t, _ in toks]) is None  # t=0: no signal


def test_prompt_uses_only_question_and_metadata():
    doc = Document(
        "d",
        "T\n\nA",
        title="Tiny Title",
        paragraphs=(Paragraph("Abstract", "We study X.", 0, 1), Paragraph("Intro", "Y", 2, 3)),
    )
    p = render_prompt("{metadata}Q: {question}", ["question", "title"], "Why X?", doc)
    assert p == "Paper title: Tiny Title\nQ: Why X?"
    p2 = render_prompt("{metadata}Q: {question}", ["question", "abstract"], "Why X?", doc)
    assert "We study X." in p2 and "Tiny Title" not in p2 and "Y" not in p2.split("Q:")[0]


# --------------------------------------------------------------------------- LLM profiler


def _resp(text, logprobs=None, tokens=(120, 30)):
    usage = SimpleNamespace(
        prompt_tokens=tokens[0], completion_tokens=tokens[1], completion_tokens_details=None
    )
    lp = (
        SimpleNamespace(content=[SimpleNamespace(token=t, logprob=v) for t, v in logprobs])
        if logprobs
        else None
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason="stop", logprobs=lp
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="deepseek-flash")


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


GOOD = (
    '{"answer_form": "short_span", "granularity": "specific_detail", "n_facts": 1, '
    '"cross_section": false}'
)


def _llm(tmp_path, responses, **cfg):
    p = LLMProfiler(ProfilerSection(**cfg), cache_dir=tmp_path, max_retries=0)
    p.chat._client = FakeClient(responses)
    return p


def test_llm_profiler_request_and_cache(tmp_path):
    p = _llm(tmp_path, [_resp(GOOD, [(GOOD, math.log(0.7))])], use_logprobs=True)
    doc = Document("d", "", title="Paper T")
    r = p.profile("q1", "What learning rate?", doc)
    assert r.profile == prof() and r.error is None and r.attempts == 1
    assert r.input_tokens == 120 and not r.cache_hit
    call = p.chat._client.calls[0]
    assert call["response_format"] == {"type": "json_object"} and call["logprobs"] is True
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "Paper T" in call["messages"][0]["content"]
    again = p.profile("q1", "What learning rate?", doc)  # served from the disk cache
    assert again.cache_hit and again.profile == r.profile and len(p.chat._client.calls) == 1


def test_llm_profiler_retries_invalid_json_then_gives_up(tmp_path):
    p = _llm(tmp_path, [_resp("oops"), _resp(GOOD)], use_logprobs=False)
    r = p.profile("q", "Q?", None)
    assert r.profile == prof() and r.attempts == 2 and r.confidence is None
    assert r.input_tokens == 240  # both attempts are paid for and counted
    p2 = _llm(tmp_path / "b", [_resp("oops"), _resp("still bad")])
    r2 = p2.profile("q", "Q?", None)
    assert r2.profile is None and "JSON" in r2.error


def test_prompt_hash_depends_on_inputs():
    a = LLMProfiler(ProfilerSection(inputs=["question", "title"])).prompt_hash
    b = LLMProfiler(ProfilerSection(inputs=["question", "title", "abstract"])).prompt_hash
    assert a != b


def test_store_roundtrip_and_cached_profiler(tmp_path):
    mock = MockProfiler()
    r = mock.profile("q1", "Does it work?", None)
    assert r.profile.answer_form == "yes_no" and mock.profile("q1", "Does it work?", None) == r
    path = tmp_path / "profiles.jsonl"
    ProfileStore.append(path, [r, ProfileResult("q2", "mock-v1", None, error="bad")])
    store = ProfileStore.load([path])
    assert store.get("q1", "mock-v1") == r and store.get("q2", "mock-v1").profile is None
    cached = CachedProfiler(mock, store)
    cached.profile("q1", "Does it work?", None)
    cached.profile("q3", "Why?", None)
    assert cached.misses == 1


# --------------------------------------------------------------------------- models


def _labels(fast: bool) -> dict[str, bool]:
    # "fast" queries are fine at the cheapest leaf config; others need collapsed@2000
    return {c: (c == "collapsed@2000") or (fast and c.startswith("leaf_only")) for c in IDS}


def test_profile_table_learns_and_backs_off():
    profiles = [prof("short_span")] * 20 + [prof("explanation", "whole_paper", 3, True)] * 20
    labels = [_labels(True)] * 20 + [_labels(False)] * 20
    m = ProfileTableModel(IDS, alpha=5).fit(profiles, labels)
    fast, slow = (
        m.predict(prof("short_span")),
        m.predict(prof("explanation", "whole_paper", 3, True)),
    )
    assert fast["leaf_only@400"] > 0.8 and slow["leaf_only@400"] < 0.2
    assert slow["collapsed@2000"] > 0.9
    # unseen full cell under a seen answer form -> that form's rate
    unseen = m.predict(prof("short_span", "section_level", 2, False))
    assert unseen["leaf_only@400"] == pytest.approx(
        m.predict(prof("short_span"))["leaf_only@400"], abs=0.15
    )
    # unseen answer form -> global rate (about 0.5 for leaf configs here)
    assert 0.3 < m.predict(prof("yes_no"))["leaf_only@400"] < 0.7
    assert (
        ProfileTableModel.from_dict(json.loads(json.dumps(m.to_dict()))).predict(prof("short_span"))
        == fast
    )


def test_rule_map_v1():
    m = RuleMapModel(CATALOG)
    p = m.predict(prof("yes_no", "specific_detail", 1, False))
    assert p["leaf_only@400"] == 1 and p["collapsed@400"] == 1 and p["summary_only@2000"] == 0
    p = m.predict(prof("explanation", "whole_paper", 3, True))
    assert p["summary_only@2000"] == 1 and p["collapsed@2000"] == 1 and p["collapsed@1000"] == 0
    assert m.predict(prof("unanswerable_likely"))["collapsed@1000"] == 1


# --------------------------------------------------------------------------- selection


COSTS = {
    f"{s}@{b}": b * f
    for s, f in (("leaf_only", 0.95), ("summary_only", 0.6), ("collapsed", 0.97))
    for b in (400, 1000, 2000)
}


def _selector(overhead=0.0):
    return PlanSelector(CATALOG, CostModel(COSTS, overhead), FALLBACK)


def test_selector_cheapest_acceptable_and_fallbacks():
    sel = _selector()
    d = sel.select({"leaf_only@1000": 0.9, "collapsed@400": 0.6, "summary_only@400": 0.1}, 0.5)
    assert d.config.id == "collapsed@400" and not d.used_fallback  # 388 < 950 tokens
    assert sel.select({"leaf_only@1000": 0.4}, 0.5).fallback_reason == "none_acceptable"
    assert sel.select(None, 0.5).config == FALLBACK
    assert sel.select({c: 1.0 for c in IDS}, 0.0).config.id == "summary_only@400"


def test_calibration_meets_target_with_overhead():
    probs = {
        f"q{i}": {c: (1.0 if c == "collapsed@2000" else i / 10) for c in IDS} for i in range(10)
    }
    sel = _selector(overhead=100)
    theta = calibrate_theta("dev", probs, sel, target_mean_tokens=1500)
    costs = [sel.cost_model.total(sel.select(p, theta).config) for p in probs.values()]
    assert sum(costs) / len(costs) <= 1500
    higher = min(t for t in (x / 100 for x in range(101)) if t > theta)
    costs_hi = [sel.cost_model.total(sel.select(p, higher).config) for p in probs.values()]
    assert sum(costs_hi) / len(costs_hi) > 1500  # theta is the largest feasible
    with pytest.raises(ValueError):
        calibrate_theta("dev", probs, sel, target_mean_tokens=100)
    with pytest.raises(TestSplitLeakageError):
        calibrate_theta("test", probs, sel, 1500)


def test_theta_sweep_replays_recorded_labels():
    def rec(c, f1, tok):
        return LabelRecord("q", "d", "f100", 224, c, "A", tok, f1)

    lookup = {"q": {c: rec(c, 1.0 if c == "collapsed@2000" else 0.0, int(COSTS[c])) for c in IDS}}
    probs = {"q": {c: (0.9 if c == "collapsed@2000" else 0.3) for c in IDS}}
    sweep = dict(
        (t, (tok, f1))
        for t, tok, f1 in theta_sweep("dev", probs, _selector(50), lookup, grid=[0.0, 0.5])
    )
    assert sweep[0.0] == (240 + 50, 0.0) and sweep[0.5] == (1940 + 50, 1.0)


# --------------------------------------------------------------------------- router


def _router(threshold=0.9, theta=0.5):
    opt = OptimizerSection(
        profiler=ProfilerSection(provider="mock", confidence_threshold=threshold)
    )
    table = ProfileTableModel(IDS).fit([prof()] * 10, [_labels(True)] * 10)
    return Router(
        opt,
        CATALOG,
        model=table,
        cost_model=CostModel(COSTS, 100),
        theta=theta,
        profiler=MockProfiler(),
    )


def test_router_fallback_reasons_and_cost_fields():
    r = _router()
    good = ProfileResult("q", "mock-v1", prof(), confidence=0.95, input_tokens=210)
    d = r.decide(good)
    assert d.config.id == "leaf_only@400" and not d.used_fallback
    assert d.router_input_tokens == 210 and d.profile["answer_form"] == "short_span"
    low = r.decide(ProfileResult("q", "mock-v1", prof(), confidence=0.5))
    assert low.config == FALLBACK and low.fallback_reason == "low_confidence"
    bad = r.decide(ProfileResult("q", "mock-v1", None, error="no JSON"))
    assert bad.fallback_reason == "invalid_profile"
    no_lp = r.decide(ProfileResult("q", "mock-v1", prof(), confidence=None))
    assert not no_lp.used_fallback  # no logprobs -> confidence check is skipped


def test_checkpoint_roundtrip_and_guards(tmp_path):
    r = _router()
    ckpt = RouterCheckpoint(
        "h",
        IDS,
        0.5,
        FALLBACK.id,
        {
            "model": r.model.to_dict(),
            "cost_model": r.selector.cost_model.to_dict(),
            "prompt_hash": "mock-v1",
        },
    )
    save_checkpoint(ckpt, tmp_path / "router")
    assert load_checkpoint(tmp_path / "router") == ckpt
    loaded = Router.from_checkpoint(tmp_path / "router", r.cfg, CATALOG, MockProfiler())
    probe = ProfileResult("q", "mock-v1", prof(), confidence=1.0)
    assert loaded.decide(probe).config == r.decide(probe).config
    other = MockProfiler()
    other.prompt_hash = "different"
    with pytest.raises(ValueError, match="prompt hash"):
        Router.from_checkpoint(tmp_path / "router", r.cfg, CATALOG, other)
    with pytest.raises(ValueError, match="catalog"):
        Router.from_checkpoint(tmp_path / "router", r.cfg, CATALOG[:3], MockProfiler())


# --------------------------------------------------------------------------- experiments


def _cli(tmp_path, cfg_path, *sets):
    from adaptive_raptor.run import main

    return main(["--config", str(cfg_path), *[x for s in sets for x in ("--set", s)]])


def _write_cfg(tmp_path, data):
    import yaml

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_query_profiling_then_router_train_end_to_end(tmp_path, no_api_keys):
    from adaptive_raptor.config.schema import DatasetSection
    from adaptive_raptor.data import load_dataset
    from adaptive_raptor.profiling import LabelStore

    out = tmp_path / "out"
    base = {
        "dataset": {"name": "qasper_tiny", "split": "dev"},
        "generation": {"provider": "mock", "model": "m"},
        "raptor": {
            "embedding": {"provider": "mock", "model": "h"},
            "summarization": {"provider": "mock", "model": "m"},
        },
        "runtime": {"cache_dir": str(tmp_path / "cache")},
        "optimizer": {"profiler": {"provider": "mock"}},
    }
    prof_cfg = _write_cfg(
        tmp_path,
        {**base, "experiment": {"name": "prof", "type": "query_profiling", "output_dir": str(out)}},
    )
    assert _cli(tmp_path, prof_cfg) == 0
    (run,) = (out / "prof").iterdir()
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "succeeded" and status["metrics"]["invalid"] == 0
    profiles = run / "profiles.jsonl"
    assert len(profiles.read_text().splitlines()) == 6

    # synthetic labels for the 6 fixture questions: yes/no-style questions are fine cheap
    qs = load_dataset(DatasetSection(name="qasper_tiny", split="dev")).questions
    store = ProfileStore.load([profiles])
    rows = []
    for q in qs:
        cheap = store.get(q.question_id, "mock-v1").profile.answer_form != "explanation"
        for c in CATALOG:
            f1 = 1.0 if (c.id == "collapsed@2000" or (cheap and c.strategy is L)) else 0.0
            rows.append(
                LabelRecord(
                    q.question_id,
                    q.doc_id,
                    "f100",
                    224,
                    c.id,
                    "A",
                    int(c.budget * 0.95),
                    f1,
                    "extractive",
                )
            )
    labels = tmp_path / "labels.jsonl"
    LabelStore(labels).append(rows)

    train_cfg = _write_cfg(
        tmp_path,
        {
            **base,
            "experiment": {"name": "train", "type": "router_train", "output_dir": str(out)},
            "evaluation": {"bootstrap_samples": 200},
            "optimizer": {
                "enabled": True,
                "target_mean_tokens": 1500,
                "train_labels": [str(labels)],
                "dev_labels": [str(labels)],
                "profiler": {"provider": "mock", "profiles_path": [str(profiles)]},
            },
        },
    )
    assert _cli(tmp_path, train_cfg) == 0
    (trun,) = (out / "train").iterdir()
    report = json.loads((trun / "dev_report.json").read_text())
    assert report["n_dev"] == 6 and report["router"]["total_tokens"] <= 1500
    assert report["router"]["answer_f1"] == pytest.approx(1.0)  # learned the synthetic rule
    prim = [d for d in report["deltas_vs_matched_fixed"] if d["primary"]]
    assert len(prim) == 1 and prim[0]["verdict"] in {"supported", "inconclusive", "equivalent"}
    ckpt = load_checkpoint(trun / "router")
    assert ckpt.artifacts["prompt_hash"] == "mock-v1" and ckpt.artifacts["n_train_queries"] == 6
    assert (trun / "theta_sweep.csv").is_file()


def test_router_policy_in_pipeline(tmp_path, no_api_keys):
    from adaptive_raptor.config.loader import load_config
    from adaptive_raptor.optimizer import RouterPolicy
    from adaptive_raptor.pipeline import QAPipeline
    from tests.conftest import SMOKE, smoke_overrides

    cfg = load_config(
        SMOKE, smoke_overrides(tmp_path) + ["retrieval.include_raptor_original=false"]
    )
    run = tmp_path / "run"
    run.mkdir()
    policy = RouterPolicy(_router(threshold=None))
    report = QAPipeline(cfg, run, policy=policy).run()
    traces = [json.loads(x) for x in (run / "traces.jsonl").read_text().splitlines()]
    assert report.n_traces == report.n_questions == len(traces) == 6
    assert all(t["router_model"].startswith("llm_profile_table/mock") for t in traces)
    assert all(t["profile"] is not None and t["router_input_tokens"] > 0 for t in traces)


def test_frozen_profiler_prompt():
    """profile_v3 is frozen (2026-11-09). Editing it must be a deliberate new version."""
    assert ProfilerSection().prompt == "profile_v3"
    hashes = {
        name: LLMProfiler(ProfilerSection(prompt=name)).prompt_hash
        for name in ("profile_v1", "profile_v2", "profile_v3")
    }
    assert hashes["profile_v3"] == "e22d6935b289c85d"
    assert len(set(hashes.values())) == 3  # versions never share stored profiles
    store = ProfileStore([ProfileResult("q", hashes["profile_v1"], prof())])
    assert store.get("q", hashes["profile_v3"]) is None


def test_h1_replay_end_to_end(tmp_path, no_api_keys):
    """Mock pipeline -> labels (dev, test) -> profiles -> router_train (A1, A2) -> H1 on test."""
    from adaptive_raptor.run import main
    from tests.conftest import SMOKE, smoke_overrides

    out, ov = tmp_path / "out", smoke_overrides(tmp_path)

    def run(*args):
        assert main(list(args)) == 0, args
        return sorted((out / name_of(args)).iterdir())[-1]

    def name_of(args):
        return next(a.split("=", 1)[1] for a in args if str(a).startswith("experiment.name="))

    def sets(*kv):
        return [x for s in [*ov, *kv] for x in ("--set", s)]

    labels = {}
    for split in ("dev", "test"):
        r = run(
            "--config", str(SMOKE), *sets(f"dataset.split={split}", f"experiment.name=cat_{split}")
        )
        labels[split] = r / "labels.jsonl"
    prof_cfg = _write_cfg(
        tmp_path,
        {
            "experiment": {"name": "p", "type": "query_profiling", "output_dir": str(out)},
            "dataset": {"name": "qasper_tiny", "split": "dev"},
            "generation": {"provider": "mock", "model": "m"},
            "raptor": {
                "embedding": {"provider": "mock", "model": "h"},
                "summarization": {"provider": "mock", "model": "m"},
            },
            "optimizer": {"profiler": {"provider": "mock"}},
        },
    )
    profiles = {
        split: run(
            "--config",
            str(prof_cfg),
            "--set",
            f"dataset.split={split}",
            "--set",
            f"experiment.name=prof_{split}",
        )
        / "profiles.jsonl"
        for split in ("dev", "test")
    }
    train_cfg = ROOT_CFG / "router_train.yaml"
    common = [
        "dataset.name=qasper_tiny",
        "generation.price_input_per_mtok=0.15",
        "generation.price_output_per_mtok=0.6",
        "evaluation.bootstrap_samples=200",
        f"optimizer.train_labels=[{labels['dev']}]",  # the fixture has no train split
        f"optimizer.dev_labels=[{labels['dev']}]",
        f"optimizer.profiler.profiles_path=[{profiles['dev']}]",
        "optimizer.profiler.provider=mock",
        "optimizer.target_mean_tokens=2000",
    ]
    a1 = run("--config", str(train_cfg), *sets(*common, "experiment.name=a1"))
    a2 = run(
        "--config",
        str(train_cfg),
        *sets(*common, "experiment.name=a2", "optimizer.model_type=llm_profile_rules"),
    )
    ckpt = load_checkpoint(a1 / "router")
    assert set(ckpt.artifacts["comparators"]) == {"best_within_budget", "nearest_higher"}
    assert ckpt.artifacts["dev_docs"] and ckpt.artifacts["dev_router"]["profiler_tokens"] > 0
    rep = json.loads((a1 / "dev_report.json").read_text())
    assert rep["operating_points"] and rep["profiles_dev"]["answer_form"]
    assert rep["raptor_original"] is not None  # smoke catalog runs include it
    assert any(o["fixed"] == "raptor_original" for o in rep["operating_points"])
    r = rep["router"]  # total = generation in + out + profiler; components reported apart
    assert r["total_tokens"] == pytest.approx(
        r["generation_input_tokens"] + r["generation_output_tokens"] + r["profiler_tokens"],
        abs=0.2,
    )

    h1 = [
        "dataset.name=qasper_tiny",
        "dataset.split=test",
        "optimizer.profiler.provider=mock",
        "evaluation.bootstrap_samples=200",
        f"optimizer.checkpoint={a1 / 'router'}",
        f"optimizer.ablation_checkpoints=[{a2 / 'router'}]",
        f"optimizer.eval_labels=[{labels['test']}]",
        f"optimizer.profiler.profiles_path=[{profiles['test']}]",
    ]
    res = run("--config", str(ROOT_CFG / "h1.yaml"), *sets(*h1, "experiment.name=h1"))
    status = json.loads((res / "status.json").read_text())
    systems = {r["system"]: r for r in status["metrics"]["systems"]}
    assert {"router:llm_profile_table", "router:llm_profile_rules", "raptor_original"} <= set(
        systems
    )
    for r in systems.values():  # every cost component is reported separately
        assert r["total_tokens"] == pytest.approx(
            r["generation_input_tokens"] + r["generation_output_tokens"] + r["profiler_tokens"],
            abs=0.2,
        )
        assert r["usd_per_query"] is not None and r["n"] == status["metrics"]["n_questions"]
    assert systems["router:llm_profile_table"]["profiler_tokens"] > 0
    assert all(r["profiler_tokens"] == 0 for k, r in systems.items() if k.startswith("fixed"))
    deltas = status["metrics"]["deltas"]
    assert {d["system"] for d in deltas} == {"router:llm_profile_table", "router:llm_profile_rules"}
    assert sum(d["primary"] for d in deltas) == 2 and all(d["ci_lo"] <= d["ci_hi"] for d in deltas)
    assert (res / "h1_systems.csv").is_file() and (res / "h1_deltas.csv").is_file()

    # the comparator is the one frozen on dev, not re-selected on test
    frozen = ckpt.artifacts["comparators"]["best_within_budget"]["weights"]
    assert any(all(c in k for c in frozen) for k in systems if k.startswith("fixed[best_within"))

    # refuse to evaluate on papers the router was calibrated on
    bad = [x.replace(str(labels["test"]), str(labels["dev"])) for x in h1]
    bad = [x.replace(str(profiles["test"]), str(profiles["dev"])) for x in bad]
    from adaptive_raptor.run import main as cli

    assert cli(["--config", str(ROOT_CFG / "h1.yaml"), *sets(*bad, "experiment.name=bad")]) != 0
    (bad_run,) = (out / "bad").iterdir()
    assert (
        "used to train or calibrate" in json.loads((bad_run / "status.json").read_text())["message"]
    )

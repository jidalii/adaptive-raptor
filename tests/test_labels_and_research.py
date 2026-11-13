"""T06 labels/merge and T07 decision rules on synthetic label tables (offline, no API)."""

import json

import numpy as np
import pytest

from adaptive_raptor.evaluation.research import (
    baseline_scores,
    fixed_config_frontier,
    match_baseline,
    match_fixed_baseline,
    oracle_at_budget,
    oracle_frontier,
    oracle_points,
    paired_document_bootstrap,
    verdict,
)
from adaptive_raptor.profiling import (
    LabelRecord,
    LabelReport,
    LabelStore,
    acceptability_labels,
    merge_label_tables,
    select_records,
)


def rec(q, c, f1, tok, run="A", doc=None, atype="extractive", variant="f100", seed=224):
    return LabelRecord(
        question_id=q,
        doc_id=doc or q.split("-")[0],
        index_variant=variant,
        seed=seed,
        config_id=c,
        run=run,
        tokens_used=tok,
        answer_f1=f1,
        answer_type=atype,
    )


# --------------------------------------------------------------------------- T06 labels


def test_epsilon_boundary_and_all_fail():
    recs = [
        rec("d1-q1", "a", 0.60, 400),
        rec("d1-q1", "b", 0.55, 1000),  # exactly best - eps -> acceptable
        rec("d1-q1", "c", 0.549, 2000),  # just below -> not
        rec("d1-q2", "a", 0.0, 400, atype="boolean"),
        rec("d1-q2", "b", 0.0, 1000, atype="boolean"),
        rec("d1-q2", "c", 0.0, 2000, atype="boolean"),
    ]
    rep = LabelReport(0.05, [])
    y = acceptability_labels(recs, 0.05, report=rep)
    assert y["d1-q1"] == {"a": True, "b": True, "c": False}
    assert y["d1-q2"] is None  # all-fail: excluded from training, kept in the output
    assert rep.n_labeled == 1 and rep.n_all_fail == 1
    assert rep.by_answer_type["boolean"]["all_fail"] == 1
    assert rep.acceptable_rate == {"a": 1.0, "b": 1.0, "c": 0.0}
    assert acceptability_labels(recs, 0.0)["d1-q1"] == {"a": True, "b": False, "c": False}


def test_ties_are_all_acceptable_and_original_is_ignored():
    recs = [rec("d-q", "a", 1.0, 400), rec("d-q", "b", 1.0, 2000)]
    recs.append(rec("d-q", "raptor_original", 0.0, 900))
    assert acceptability_labels(recs, 0.05) == {"d-q": {"a": True, "b": True}}


def test_incomplete_queries_are_excluded_and_counted():
    recs = [rec("d-q1", "a", 1, 1), rec("d-q1", "b", 1, 2), rec("d-q2", "a", 1, 1)]
    rep = LabelReport(0.05, [])
    y = acceptability_labels(recs, 0.05, ["a", "b"], report=rep)
    assert list(y) == ["d-q1"] and rep.incomplete == ["d-q2"]


def test_mixed_cells_and_duplicates_are_rejected():
    with pytest.raises(ValueError, match="mix"):
        acceptability_labels([rec("q", "a", 1, 1), rec("q", "a", 1, 1, run="B")], 0.05)
    with pytest.raises(ValueError, match="duplicate"):
        acceptability_labels([rec("q", "a", 1, 1), rec("q", "a", 1, 1)], 0.05)
    mixed = [rec("q", "a", 1, 1), rec("q", "a", 1, 1, run="B")]
    assert len(select_records(mixed, run="B")) == 1


def test_merge_dedups_and_reports_conflicts(tmp_path):
    p1, p2 = tmp_path / "r1/labels.jsonl", tmp_path / "r2/labels.jsonl"
    LabelStore(p1).append([rec("d-q1", "a", 0.5, 1), rec("d-q2", "a", 1.0, 1)])
    LabelStore(p2).append([rec("d-q1", "a", 0.5, 1), rec("d-q2", "a", 0.0, 1)])  # q2 differs
    merged, rep = merge_label_tables([p1, p2])
    assert rep.n_input == 4 and rep.n_unique == 2 and rep.n_duplicates == 2
    assert [k[0] for k in rep.conflicts] == ["d-q2"]
    assert {r.question_id: r.answer_f1 for r in merged}["d-q2"] == 1.0  # first wins
    assert json.loads(p1.read_text().splitlines()[0])["config_id"] == "a"


# --------------------------------------------------------------------------- frontier


FIXED = [
    # q1 and q2, four configs; means: a(400, .3) b(1000, .5) c(2000, .45) d(1000, .4)
    *(
        rec("d1-q1", c, f, t)
        for c, f, t in [("a", 0.2, 400), ("b", 0.6, 1000), ("c", 0.5, 2000), ("d", 0.4, 1000)]
    ),
    *(
        rec("d2-q2", c, f, t)
        for c, f, t in [("a", 0.4, 400), ("b", 0.4, 1000), ("c", 0.4, 2000), ("d", 0.4, 1000)]
    ),
]


def test_fixed_frontier_is_upper_envelope():
    front = fixed_config_frontier(FIXED)
    assert [c for c, _, _ in front] == ["a", "b"]
    assert front[1] == pytest.approx(("b", 1000.0, 0.5))


# --------------------------------------------------------------------------- oracle


def _toy(run="A", q1=(1.0, 1.0), q2=(0.0, 1.0)):
    return [
        rec("d1-q1", "a", q1[0], 400, run),
        rec("d1-q1", "b", q1[1], 1000, run),
        rec("d2-q2", "a", q2[0], 400, run),
        rec("d2-q2", "b", q2[1], 1000, run),
    ]


def test_oracle_without_cross_fitting():
    pts = oracle_frontier(_toy())
    assert pts[0] == pytest.approx((400.0, 0.5))  # large lambda: always cheapest
    assert pts[-1] == pytest.approx((700.0, 1.0))  # lambda=0: best F1, ties -> cheaper
    p = oracle_at_budget(_toy(), None, 700)
    assert p.choice == {"d1-q1": "a", "d2-q2": "b"} and p.mean_f1 == pytest.approx(1.0)
    assert oracle_at_budget(_toy(), None, 699).mean_f1 == pytest.approx(0.5)
    with pytest.raises(ValueError):
        oracle_at_budget(_toy(), None, 100)


def test_cross_fitting_removes_winners_curse():
    run_a = _toy("A", q1=(1.0, 0.0), q2=(0.0, 1.0))  # A picks q1->a, q2->b
    run_b = _toy("B", q1=(0.0, 1.0), q2=(0.0, 1.0))  # B disagrees on q1
    in_sample = oracle_at_budget(run_a, None, 2000).mean_f1
    cross = oracle_at_budget(run_a, run_b, 2000)
    assert in_sample == pytest.approx(1.0)
    assert cross.choice == {"d1-q1": "a", "d2-q2": "b"}
    assert cross.mean_f1 == pytest.approx(0.5)  # scored on B, chosen on A
    assert len(oracle_points(run_a, run_b)) >= 2


# --------------------------------------------------------------------------- matching

FRONT = [
    ("leaf_only@400", 380.0, 0.30),
    ("collapsed@1000", 950.0, 0.40),
    ("collapsed@2000", 1900.0, 0.45),
]


def test_conservative_matching_never_gives_baseline_fewer_tokens():
    assert match_fixed_baseline(600, FRONT) == "collapsed@1000"
    assert match_fixed_baseline(950, FRONT) == "collapsed@1000"
    assert match_fixed_baseline(951, FRONT) == "collapsed@2000"
    m = match_baseline(2500, FRONT)
    assert m.config_id == "collapsed@2000" and not m.satisfied
    # frontier ends at 950 tokens (more context stopped helping) but a 1900-token config
    # exists: the best fixed config within a >= router budget is the dominating one
    short = FRONT[:2]
    m = match_baseline(1500, short, max_fixed_tokens=1900)
    assert m.config_id == "collapsed@1000" and m.satisfied
    assert not match_baseline(2500, short, max_fixed_tokens=1900).satisfied
    assert match_baseline(1500, short, "interpolate", max_fixed_tokens=1900).satisfied


def test_interpolated_matching_and_per_query_scores():
    m = match_baseline(665, FRONT, "interpolate")
    assert m.weights == pytest.approx({"leaf_only@400": 0.5, "collapsed@1000": 0.5})
    assert m.mean_f1 == pytest.approx(0.35) and m.satisfied
    recs = [rec("d-q", "leaf_only@400", 0.2, 380), rec("d-q", "collapsed@1000", 0.6, 950)]
    assert baseline_scores(m, recs) == {"d-q": pytest.approx(0.4)}
    with pytest.raises(ValueError):
        match_baseline(500, FRONT, "nearest")


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_mean_is_query_weighted_and_deterministic():
    delta = {"q1": 1.0, "q2": 1.0, "q3": 0.0}
    docs = {"q1": "d1", "q2": "d1", "q3": "d2"}
    a = paired_document_bootstrap(delta, docs, 500, 0.95, seed=3)
    assert a == paired_document_bootstrap(delta, docs, 500, 0.95, seed=3)
    assert a[0] == pytest.approx(2 / 3) and a[1] <= a[0] <= a[2]
    one_doc = paired_document_bootstrap({"q": 0.3}, {"q": "d"}, 100, 0.9)
    assert one_doc == pytest.approx((0.3, 0.3, 0.3))
    with pytest.raises(KeyError):
        paired_document_bootstrap({"q": 1.0}, {}, 10, 0.95)


def test_bootstrap_coverage_on_zero_effect_simulation():
    """A 95% CI on clustered zero-mean deltas should contain 0 about 95% of the time."""
    rng = np.random.default_rng(0)
    n_sims, covered = 200, 0
    for sim in range(n_sims):
        delta, docs = {}, {}
        for d in range(40):
            doc_effect = rng.normal(0, 0.1)  # within-paper correlation
            for j in range(int(rng.integers(1, 7))):
                q = f"s{sim}-d{d}-q{j}"
                delta[q] = doc_effect + rng.normal(0, 0.2)
                docs[q] = f"d{d}"
        _, lo, hi = paired_document_bootstrap(delta, docs, 1000, 0.95, seed=sim)
        covered += lo <= 0 <= hi
    assert 0.89 <= covered / n_sims <= 0.99, covered / n_sims


def test_verdicts():
    assert verdict((0.01, 0.05), 0.02) == "supported"
    assert verdict((-0.05, -0.01), 0.02) == "worse"
    assert verdict((-0.01, 0.015), 0.02) == "equivalent"
    assert verdict((-0.01, 0.05), 0.02) == "inconclusive"
    assert verdict((-0.01, 0.015), None) == "inconclusive"


def test_billed_cost_excludes_cache_hits():
    from adaptive_raptor.evaluation.traces import summarize

    base = dict(
        index_variant="f",
        config_id="c",
        error=None,
        answer_f1=1.0,
        context_tokens=1,
        retrieval_ms=1.0,
        generation_ms=1.0,
        answer_type="none",
        input_tokens=1_000_000,
        output_tokens=0,
    )
    traces = [{**base, "generation_cache_hit": True}, {**base, "generation_cache_hit": False}]
    (row,) = summarize(traces, 0.15, 0.60)
    assert row["cost_usd"] == pytest.approx(0.30) and row["billed_usd"] == pytest.approx(0.15)


def test_select_comparator_rules():
    from adaptive_raptor.evaluation.research import ConfigPoint, select_comparator

    pts = [
        ConfigPoint("leaf_only@400", 550.0, 0.30, 10),
        ConfigPoint("summary_only@1000", 900.0, 0.25, 10),  # dominated
        ConfigPoint("collapsed@1000", 1150.0, 0.40, 10),
        ConfigPoint("leaf_only@2000", 2150.0, 0.38, 10),  # more cost, lower F1
    ]
    best = select_comparator(1300, pts, "best_within_budget")
    assert best.config_id == "collapsed@1000" and best.satisfied
    # budget covers the 2150 config, but the best affordable config is still collapsed@1000
    assert select_comparator(2500, pts, "best_within_budget").config_id == "collapsed@1000"
    assert not select_comparator(500, pts, "best_within_budget").satisfied
    near = select_comparator(1300, pts, "nearest_higher")
    assert near.config_id == "leaf_only@2000" and near.satisfied  # sensitivity rule
    assert not select_comparator(3000, pts, "nearest_higher").satisfied
    with pytest.raises(ValueError):
        select_comparator(1000, pts, "conservative")

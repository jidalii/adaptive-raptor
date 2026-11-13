"""Behavioral checks for the standalone P0b diagnostic (no models or APIs)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_tree_navigation import context, match, navigate, pack


def test_navigation_excludes_unexpanded_branch_and_deduplicates():
    nodes = {i: SimpleNamespace(index=i, children=set()) for i in range(7)}
    nodes[4].children = {0, 1}
    nodes[5].children = {1, 2, 3}
    nodes[6].children = {4, 5}
    tree = SimpleNamespace(
        all_nodes=nodes,
        layer_to_nodes={0: [nodes[i] for i in range(4)], 1: [nodes[4], nodes[5]], 2: [nodes[6]]},
    )
    scores = {i: float(i) for i in nodes}
    reached, path = navigate(tree, scores, 1)
    assert reached == [0, 1]  # beam=1 must not cap the final leaves to one
    assert path[-1]["selected"] == [4]
    reached, _ = navigate(tree, scores, 2)
    assert reached == [0, 1, 2, 3]
    nodes[6].children.add(0)
    with pytest.raises(ValueError, match="invalid child"):
        navigate(tree, scores, 2)


def test_leaf_only_and_empty_frontier():
    leaf = SimpleNamespace(index=0, children=set())
    root = SimpleNamespace(index=1, children=set())
    tree = SimpleNamespace(all_nodes={0: leaf}, layer_to_nodes={0: [leaf]})
    assert navigate(tree, {0: 0.0}, 1) == ([0], [])
    tree.all_nodes[1] = root
    tree.layer_to_nodes[1] = [root]
    assert navigate(tree, {0: 0.0, 1: 1.0}, 1)[0] == []


def test_packing_counts_separators_and_stops_at_first_nonfit():
    texts = {0: "aa", 1: "bbbb", 2: "x"}
    assert context([0, 1], texts) == "aa\n\nbbbb"
    assert pack([0, 1, 2], texts, 5, list) == [0]
    assert pack([0, 1], texts, 8, list) == [0, 1]
    assert pack([0], texts, 0, list) == []


def test_mix_uses_upper_hull_and_cost_support():
    cost = np.array([0.0, 100.0, 200.0, 300.0])
    quality = np.array([0.0, 0.1, 0.5, 0.4])
    i, j, w = match(cost, quality, 100.0)
    assert (i, j, w) == (0, 2, 0.5)
    assert match(cost, quality, 250.0) == (2, 2, 0.0)
    with pytest.raises(ValueError, match="outside"):
        match(cost, quality, 301.0)


def test_comparison_bootstraps_documents_and_excludes_missing_evidence():
    from check_tree_navigation import PROTOCOL, compare

    rows = []
    for i in range(6):
        methods = {
            f"leaf@{b}": {"context_tokens": b, "hl_recall": 0.5 if i < 4 else None}
            for b in PROTOCOL["leaf_budgets"]
        }
        methods["leaf@0"]["hl_recall"] = 0.0 if i < 4 else None
        methods["beam1@1000"] = {"context_tokens": 900, "hl_recall": 0.75 if i < 4 else None}
        rows.append({"question_id": str(i), "doc_id": str(i // 2), "methods": methods})
    result = compare(rows, "beam1@1000", bootstrap=100)
    assert result["n_evidence"] == 4
    assert result["actual_tokens"] == 900
    assert result["delta"] == 0.25
    assert result["ci_97_5"] == [0.25, 0.25]

"""Smoke tests for the PyO3-exposed `flybrain_native` module.

These tests assert that the Rust↔Python boundary is wired correctly: a few
JSON round-trips, a synthetic graph build, a stable graph hash, and the
budget verifier. They do not depend on Yandex credentials.
"""

from __future__ import annotations

import pytest

native = pytest.importorskip("flybrain.flybrain_native")


def test_module_metadata() -> None:
    assert native.__version__
    assert native.__modinfo__["crate"] == "flybrain-py"


def test_agent_spec_round_trip_keeps_required_fields() -> None:
    obj = {
        "name": "Planner",
        "role": "planner",
        "system_prompt": "you plan",
    }
    out = native.agent_spec_round_trip(obj)
    assert out["name"] == "Planner"
    assert out["model_tier"] == "lite"
    assert out["cost_weight"] == 1.0
    assert out["tools"] == []


def test_agent_spec_round_trip_preserves_pro_tier() -> None:
    obj = {
        "name": "Critic",
        "role": "critic",
        "system_prompt": "you criticise",
        "model_tier": "pro",
        "cost_weight": 2.0,
    }
    out = native.agent_spec_round_trip(obj)
    assert out["model_tier"] == "pro"
    assert out["cost_weight"] == 2.0


def test_graph_action_terminate_round_trip() -> None:
    out = native.graph_action_round_trip({"kind": "terminate"})
    assert out == {"kind": "terminate"}


def test_graph_action_add_edge_round_trip() -> None:
    obj = {"kind": "add_edge", "from": "A", "to": "B", "weight": 0.5}
    out = native.graph_action_round_trip(obj)
    assert out == obj


def test_synthetic_graph_is_deterministic() -> None:
    a = native.build_synthetic_fly_graph(64, 42)
    b = native.build_synthetic_fly_graph(64, 42)
    assert a["num_nodes"] == b["num_nodes"] == 64
    assert a["edge_index"] == b["edge_index"]
    assert a["edge_weight"] == b["edge_weight"]


def test_synthetic_graph_seed_changes_topology() -> None:
    a = native.build_synthetic_fly_graph(64, 1)
    b = native.build_synthetic_fly_graph(64, 2)
    assert a["edge_index"] != b["edge_index"]


def test_agent_graph_hash_changes_with_edges() -> None:
    g = {"nodes": ["A", "B", "C"], "edges": {}}
    h1 = native.agent_graph_hash(g)
    g["edges"]["A"] = {"B": 1.0}
    h2 = native.agent_graph_hash(g)
    assert h1 != h2


def test_budget_check_pass_under_cap() -> None:
    r = native.budget_check(2000.0, 100.0, 1000, 500, 5)
    assert r["passed"] is True
    assert r["score"] > 0


def test_budget_check_fail_over_cap() -> None:
    r = native.budget_check(50.0, 100.0, 1000, 500, 5)
    assert r["passed"] is False
    assert r["failed_component"] == "budget"

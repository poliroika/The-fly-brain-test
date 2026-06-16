"""Unit tests for `ControllerStateBuilder.from_runtime`.

These cover the Phase-4 exit criterion: a tensor-friendly snapshot
assembled in <50 ms on CPU, with all expected fields populated and
the right shapes.
"""

from __future__ import annotations

import time

import numpy as np

from flybrain.agents.specs import MINIMAL_15
from flybrain.embeddings import (
    AgentEmbedder,
    AgentGraphEmbedder,
    ControllerStateBuilder,
    FlyGraphEmbedder,
    MockEmbeddingClient,
    TaskEmbedder,
    TraceEmbedder,
)
from flybrain.graph.dataclasses import FlyGraph
from flybrain.runtime.state import RuntimeState


def _builder(*, emb_dim: int = 32, fly_dim: int = 16) -> ControllerStateBuilder:
    client = MockEmbeddingClient(output_dim=emb_dim)
    agents = AgentEmbedder(client=client)
    agents.precompute_sync(MINIMAL_15)
    return ControllerStateBuilder(
        task=TaskEmbedder(client=client),
        agents=agents,
        trace=TraceEmbedder(client=client),
        fly=FlyGraphEmbedder(dim=fly_dim),
        agent_graph=AgentGraphEmbedder(in_dim=emb_dim, hidden_dim=fly_dim, out_dim=fly_dim),
    )


def _runtime() -> RuntimeState:
    names = [s.name for s in MINIMAL_15]
    return RuntimeState(
        task_id="t1",
        task_type="coding",
        prompt="implement a fizzbuzz",
        step_id=3,
        available_agents=names,
        pending_inbox={n: 0 for n in names},
        last_active_agent="Planner",
        produced_components={"plan", "code"},
    )


def test_controller_state_has_expected_shapes() -> None:
    b = _builder(emb_dim=32, fly_dim=16)
    rt = _runtime()
    state = b.from_runtime_sync(rt)
    shapes = state.shapes
    assert shapes["task_vec"] == (32,)
    assert shapes["agent_node_vecs"] == (15, 32)
    assert shapes["agent_graph_vec"] == (16,)
    assert shapes["agent_node_emb"] == (15, 16)
    assert shapes["trace_vec"] == (32 + 13,)
    assert shapes["fly_vec"] == (16,)
    assert shapes["inbox_vec"] == (15,)
    assert shapes["produced_mask"] == (6,)


def test_controller_state_produced_mask_reflects_runtime() -> None:
    b = _builder()
    rt = _runtime()
    s = b.from_runtime_sync(rt)
    plan_idx = s.component_tags.index("plan")
    code_idx = s.component_tags.index("code")
    final_idx = s.component_tags.index("final_answer")
    assert s.produced_mask[plan_idx] == 1.0
    assert s.produced_mask[code_idx] == 1.0
    assert s.produced_mask[final_idx] == 0.0


def test_controller_state_is_deterministic() -> None:
    b = _builder()
    rt = _runtime()
    s1 = b.from_runtime_sync(rt)
    s2 = b.from_runtime_sync(rt)
    np.testing.assert_array_equal(s1.task_vec, s2.task_vec)
    np.testing.assert_array_equal(s1.agent_graph_vec, s2.agent_graph_vec)
    np.testing.assert_array_equal(s1.fly_vec, s2.fly_vec)


def test_controller_state_uses_fly_graph_when_supplied() -> None:
    import flybrain.flybrain_native as native

    b = _builder()
    rt = _runtime()
    no_prior = b.from_runtime_sync(rt)
    assert np.all(no_prior.fly_vec == 0.0)

    fly_dict = native.build_synthetic_fly_graph(64, 7)
    b.fly_graph = FlyGraph.from_dict(fly_dict)
    with_prior = b.from_runtime_sync(rt)
    assert not np.all(with_prior.fly_vec == 0.0)


def test_controller_state_meets_50ms_warm_budget() -> None:
    """Phase-4 exit criterion. We allow the first call to amortise the
    one-time embedding-cache miss; only warm runs are timed."""
    b = _builder(emb_dim=64, fly_dim=32)
    rt = _runtime()
    # Warm-up.
    b.from_runtime_sync(rt)
    # Time 5 warm runs and assert the median is well under the budget.
    runs: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        s = b.from_runtime_sync(rt)
        runs.append((time.perf_counter() - t0) * 1000)
        assert s.build_ms < 50.0, f"build_ms={s.build_ms:.2f}ms exceeded 50ms budget"
    runs.sort()
    median = runs[len(runs) // 2]
    assert median < 50.0, f"median wall_ms={median:.2f}ms exceeded 50ms budget"


def test_controller_state_includes_trace_features() -> None:
    b = _builder()
    rt = _runtime()
    steps = [
        {
            "output_summary": "Plan: do X",
            "tokens_in": 5,
            "tokens_out": 3,
            "latency_ms": 10,
            "verifier_score": None,
            "errors": [],
            "tool_calls": [],
            "graph_action": {"kind": "run_agent"},
            "current_graph_hash": "h0",
            "cost_rub": 0.0,
        },
        {
            "output_summary": "Code written",
            "tokens_in": 7,
            "tokens_out": 4,
            "latency_ms": 15,
            "verifier_score": 0.9,
            "errors": [],
            "tool_calls": [{"name": "python_exec"}],
            "graph_action": {"kind": "run_agent"},
            "current_graph_hash": "h1",
            "cost_rub": 0.01,
        },
    ]
    s = b.from_runtime_sync(rt, trace_steps=steps)
    # Last 13 entries are the handcrafted features.
    feats = s.trace_vec[-13:]
    assert feats[0] == 2.0  # num_steps
    assert feats[1] == 2.0  # num_agent_runs
    assert feats[2] == 1.0  # num_tool_calls
    assert feats[5] == 0.9  # mean_verifier_score


def test_controller_state_inbox_vec_matches_pending() -> None:
    names = [s.name for s in MINIMAL_15]
    pending = {n: 0 for n in names}
    pending["Planner"] = 3
    pending["Coder"] = 1
    rt = RuntimeState(
        task_id="t",
        task_type="coding",
        prompt="x",
        step_id=0,
        available_agents=names,
        pending_inbox=pending,
        last_active_agent=None,
    )
    b = _builder()
    s = b.from_runtime_sync(rt)
    planner_idx = names.index("Planner")
    coder_idx = names.index("Coder")
    assert s.inbox_vec[planner_idx] == 3.0
    assert s.inbox_vec[coder_idx] == 1.0

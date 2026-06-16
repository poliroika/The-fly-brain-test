"""Unit tests for the per-domain embedders (task / agent / trace / graph / fly)."""

from __future__ import annotations

import asyncio

import numpy as np

from flybrain.embeddings.agent_emb import AgentEmbedder, _agent_text
from flybrain.embeddings.fly_emb import FlyGraphEmbedder
from flybrain.embeddings.graph_emb import AgentGraphEmbedder
from flybrain.embeddings.mock_client import MockEmbeddingClient
from flybrain.embeddings.task_emb import TaskEmbedder
from flybrain.embeddings.trace_emb import TraceEmbedder, extract_features
from flybrain.graph.dataclasses import FlyGraph, NodeMetadata
from flybrain.runtime.agent import AgentSpec

# ---------------------------------------------------------------- task_emb


def test_task_embedder_returns_float32_of_expected_dim() -> None:
    client = MockEmbeddingClient(output_dim=32)
    emb = TaskEmbedder(client=client)
    out = asyncio.run(emb.embed("write hello world", task_type="coding"))
    assert out.shape == (32,)
    assert out.dtype == np.float32


def test_task_embedder_task_type_changes_vector() -> None:
    client = MockEmbeddingClient(output_dim=16)
    emb = TaskEmbedder(client=client)
    a = asyncio.run(emb.embed("solve x+1=2", task_type="math"))
    b = asyncio.run(emb.embed("solve x+1=2", task_type="research"))
    assert not np.allclose(a, b)


# ---------------------------------------------------------------- agent_emb


def _spec(name: str, role: str, prompt: str) -> AgentSpec:
    return AgentSpec(name=name, role=role, system_prompt=prompt)


def test_agent_text_includes_role_and_prompt() -> None:
    spec = _spec("Coder", "coder", "Write Python code")
    text = _agent_text(spec)
    assert "name=Coder" in text
    assert "role=coder" in text
    assert "Write Python code" in text


def test_agent_embedder_precompute_and_lookup() -> None:
    client = MockEmbeddingClient(output_dim=16)
    emb = AgentEmbedder(client=client)
    specs = [_spec("A", "planner", "Plan"), _spec("B", "coder", "Code")]
    emb.precompute_sync(specs)
    a = emb.get("A")
    b = emb.get("B")
    assert a.shape == b.shape == (16,)
    assert not np.allclose(a, b)
    assert emb.known_names == ["A", "B"]


def test_agent_embedder_stack_zero_pads_unknown_names() -> None:
    client = MockEmbeddingClient(output_dim=8)
    emb = AgentEmbedder(client=client)
    specs = [_spec("Known", "x", "y")]
    emb.precompute_sync(specs)
    stacked = emb.stack(["Known", "Missing"])
    assert stacked.shape == (2, 8)
    assert np.linalg.norm(stacked[0]) > 0
    assert np.allclose(stacked[1], 0.0)


def test_agent_embedder_stack_empty_returns_empty_matrix() -> None:
    client = MockEmbeddingClient(output_dim=4)
    emb = AgentEmbedder(client=client)
    out = emb.stack([])
    assert out.shape == (0, 4)


def test_agent_embedder_get_raises_when_not_precomputed() -> None:
    client = MockEmbeddingClient(output_dim=4)
    emb = AgentEmbedder(client=client)
    try:
        emb.get("Nope")
    except KeyError as e:
        assert "precomputed" in str(e)
    else:
        raise AssertionError("expected KeyError")


# ---------------------------------------------------------------- trace_emb


def _step(
    *,
    output: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency: int = 0,
    score: float | None = None,
    errors: list[str] | None = None,
    tool_calls: list[dict] | None = None,
    graph_action_kind: str = "noop",
    graph_hash: str = "h0",
    cost: float = 0.0,
) -> dict:
    return {
        "output_summary": output,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency,
        "verifier_score": score,
        "errors": errors or [],
        "tool_calls": tool_calls or [],
        "graph_action": {"kind": graph_action_kind},
        "current_graph_hash": graph_hash,
        "cost_rub": cost,
    }


def test_extract_features_handles_empty_trace() -> None:
    feats = extract_features([])
    assert feats.shape == (13,)
    assert np.all(feats == 0.0)


def test_extract_features_counts_run_agent_and_verifier_calls() -> None:
    steps = [
        _step(graph_action_kind="run_agent", output="plan", tokens_in=10, tokens_out=5, latency=20),
        _step(
            graph_action_kind="run_agent", output="code", tokens_in=20, tokens_out=10, latency=40
        ),
        _step(graph_action_kind="call_verifier", score=0.8, latency=5),
        _step(graph_action_kind="call_verifier", score=0.6, latency=5, errors=["missing"]),
    ]
    feats = extract_features(steps)
    assert feats[0] == 4  # num_steps
    assert feats[1] == 2  # num_agent_runs
    assert feats[3] == 1  # num_errors
    assert feats[4] == 2  # num_verifier_calls
    assert abs(feats[5] - 0.7) < 1e-5  # mean_verifier_score
    assert feats[6] == 0.6  # min_verifier_score


def test_trace_embedder_output_shape_matches_dim_plus_features() -> None:
    client = MockEmbeddingClient(output_dim=8)
    trace = TraceEmbedder(client=client)
    assert trace.feature_dim == 13
    assert trace.output_dim == 8 + 13

    steps = [_step(output="hello"), _step(output="world", score=0.9)]
    out = trace.embed_sync(steps)
    assert out.shape == (8 + 13,)


def test_trace_embedder_empty_yields_zeros_in_pool() -> None:
    client = MockEmbeddingClient(output_dim=8)
    trace = TraceEmbedder(client=client)
    out = trace.embed_sync([])
    assert np.all(out == 0.0)


def test_trace_embedder_pool_changes_with_step_outputs() -> None:
    client = MockEmbeddingClient(output_dim=8)
    trace = TraceEmbedder(client=client)
    a = trace.embed_sync([_step(output="alpha")])
    b = trace.embed_sync([_step(output="beta")])
    # Pool changes; handcrafted features (no errors / 0 latency) match.
    assert not np.allclose(a[:8], b[:8])
    np.testing.assert_array_equal(a[8:], b[8:])


# ---------------------------------------------------------------- graph_emb (agent GCN)


def test_agent_graph_embedder_handles_empty_graph() -> None:
    gcn = AgentGraphEmbedder(in_dim=4, hidden_dim=4, out_dim=4)
    graph_vec, node_emb = gcn.embed(
        {"nodes": [], "edges": {}}, [], np.zeros((0, 4), dtype=np.float32)
    )
    assert graph_vec.shape == (4,)
    assert node_emb.shape == (0, 4)
    assert np.all(graph_vec == 0.0)


def test_agent_graph_embedder_propagates_signal_along_edges() -> None:
    gcn = AgentGraphEmbedder(in_dim=4, hidden_dim=8, out_dim=8)
    nodes = ["a", "b"]
    feats = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    isolated = {"nodes": nodes, "edges": {}}
    connected = {"nodes": nodes, "edges": {"a": {"b": 1.0}}}
    g_iso, _ = gcn.embed(isolated, nodes, feats)
    g_con, _ = gcn.embed(connected, nodes, feats)
    # Adding the edge MUST change at least one entry of the graph vector.
    assert not np.allclose(g_iso, g_con)


def test_agent_graph_embedder_rejects_shape_mismatch() -> None:
    gcn = AgentGraphEmbedder(in_dim=4)
    nodes = ["a", "b"]
    bad_feats = np.zeros((1, 4), dtype=np.float32)
    try:
        gcn.embed({"nodes": nodes, "edges": {}}, nodes, bad_feats)
    except ValueError as e:
        assert "node_features has 1 rows" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------- fly_emb (spectral)


def _toy_fly_graph(n: int = 8) -> FlyGraph:
    edges = []
    weights = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        weights.append(1.0)
    return FlyGraph(
        num_nodes=n,
        edge_index=edges,
        edge_weight=weights,
        is_excitatory=[True] * len(edges),
        nodes=[NodeMetadata(id=i) for i in range(n)],
    )


def test_fly_embedder_returns_expected_shape() -> None:
    emb = FlyGraphEmbedder(dim=4)
    g = _toy_fly_graph(8)
    out = emb.embed(g)
    assert out.shape == (8, 4)
    vec = emb.graph_vector(g)
    assert vec.shape == (4,)


def test_fly_embedder_zero_node_graph_returns_zero_vec() -> None:
    emb = FlyGraphEmbedder(dim=4)
    empty = FlyGraph(num_nodes=0, edge_index=[], edge_weight=[], is_excitatory=[], nodes=[])
    vec = emb.graph_vector(empty)
    assert vec.shape == (4,)
    assert np.all(vec == 0.0)


def test_fly_embedder_pads_when_dim_exceeds_n_minus_one() -> None:
    emb = FlyGraphEmbedder(dim=16)
    g = _toy_fly_graph(4)  # only 3 non-trivial eigenvectors available
    out = emb.embed(g)
    assert out.shape == (4, 16)
    # Last columns must be zero-padding.
    assert np.allclose(out[:, 3:], 0.0)

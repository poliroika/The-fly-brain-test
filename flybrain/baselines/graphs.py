"""Static AgentGraph builders used by Phase-9 baselines (README §15).

Each builder returns a JSON-shaped graph dict ``{"nodes": [...],
"edges": {src: {dst: weight}}}`` matching ``flybrain_core::AgentGraph``
so the runtime ingests it via ``MAS.run(initial_graph=...)``.
"""

from __future__ import annotations

import random
from typing import Any


def empty_graph(agent_names: list[str]) -> dict[str, Any]:
    """The default graph the runtime uses when no initial graph is
    supplied: a node set with no edges. Equivalent to baseline #1
    (Manual MAS graph) when paired with ``ManualController``."""
    return {"nodes": list(agent_names), "edges": {}}


def fully_connected_graph(
    agent_names: list[str],
    *,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Baseline #2 — every agent broadcasts to every other agent.
    Equivalent to a complete directed graph on ``len(agent_names)``
    nodes. Self-loops are omitted."""
    edges: dict[str, dict[str, float]] = {}
    for src in agent_names:
        edges[src] = {dst: weight for dst in agent_names if dst != src}
    return {"nodes": list(agent_names), "edges": edges}


def random_sparse_graph(
    agent_names: list[str],
    *,
    edge_prob: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Baseline #3 — Erdős–Rényi style random sparse graph.

    Each directed edge is sampled independently with probability
    ``edge_prob``; weight is fixed at 1.0. The seed makes the result
    reproducible across runs.
    """
    rng = random.Random(seed)
    edges: dict[str, dict[str, float]] = {}
    for src in agent_names:
        bucket: dict[str, float] = {}
        for dst in agent_names:
            if src == dst:
                continue
            if rng.random() < edge_prob:
                bucket[dst] = 1.0
        if bucket:
            edges[src] = bucket
    return {"nodes": list(agent_names), "edges": edges}


def degree_preserving_random_graph(
    agent_names: list[str],
    *,
    fly_adjacency: dict[str, list[str]] | None = None,
    target_out_degree: float = 2.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Baseline #4 — random graph that *matches the out-degree* of an
    underlying graph (defaults to a target average). Edges are
    rewired uniformly at random while keeping each source's out-degree
    intact.

    Pass ``fly_adjacency`` (a ``{src: [dst, ...]}`` map derived from
    the FlyBrain graph) to preserve the actual fly-prior degree
    sequence; otherwise every node gets ``target_out_degree``
    out-edges to random non-self neighbours.
    """
    rng = random.Random(seed)
    edges: dict[str, dict[str, float]] = {}
    for src in agent_names:
        if fly_adjacency is not None and src in fly_adjacency:
            k = len(fly_adjacency[src])
        else:
            k = max(1, round(target_out_degree))
        candidates = [n for n in agent_names if n != src]
        rng.shuffle(candidates)
        chosen = candidates[: min(k, len(candidates))]
        if chosen:
            edges[src] = {dst: 1.0 for dst in chosen}
    return {"nodes": list(agent_names), "edges": edges}

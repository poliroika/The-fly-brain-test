"""`AgentGraphEmbedder` — tiny GCN over the live agent graph.

We do *not* depend on `torch` for Phase 4 — the controller variants
(GNN, RNN, learned-router) that *do* use torch land in Phase 5. The
embedder here is a pure-numpy 2-layer message-passing block that
preserves shape compatibility with the future torch GCN and is fast
enough to fit inside the <50 ms `ControllerState.from_runtime`
budget.

Concretely we apply two rounds of:

    H_{l+1} = ReLU( D^{-1/2} (A + I) D^{-1/2} H_l W_l )

where `A` is the dynamic agent graph (directed → symmetrised by
adding the transpose), `I` is identity self-loops, `D` is the degree
matrix of `(A + I)_sym`, and `W_l` is a deterministic Gaussian
random projection seeded by `(layer_index, in_dim, out_dim)`.

The output is the **mean-pooled** node-embedding matrix (so the
controller gets a single graph-level vector). Per-node embeddings
are also exposed for the GNN controller in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _seeded_projection(in_dim: int, out_dim: int, layer: int) -> np.ndarray:
    """Deterministic random projection. Seed encodes the layer index so
    a 2-layer stack uses different matrices."""
    rng = np.random.default_rng(seed=hash(("agent-gcn", layer, in_dim, out_dim)) & 0xFFFFFFFF)
    w = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    # Glorot scaling keeps the activation magnitudes roughly invariant
    # across in/out dim changes.
    w *= np.sqrt(2.0 / max(1, in_dim))
    return w


def _build_adjacency(graph: dict[str, Any], node_order: list[str]) -> np.ndarray:
    """Return the dense `(n, n)` adjacency matrix in `node_order`.

    Edges in the agent graph are stored as `{src: {dst: weight}}`.
    Missing nodes / edges are treated as zero. The returned matrix is
    *symmetrised* (`A + A^T`) so an undirected GCN sees both ends.
    """
    n = len(node_order)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    idx = {name: i for i, name in enumerate(node_order)}
    a = np.zeros((n, n), dtype=np.float32)
    edges = graph.get("edges", {}) or {}
    for src, neighbours in edges.items():
        if src not in idx:
            continue
        i = idx[src]
        if not isinstance(neighbours, dict):
            continue
        for dst, w in neighbours.items():
            if dst not in idx:
                continue
            j = idx[dst]
            try:
                weight = float(w)
            except (TypeError, ValueError):
                weight = 1.0
            a[i, j] = weight
    # Symmetrise. Directed edges become undirected for message passing.
    a = a + a.T
    return a


def _normalise(adj: np.ndarray) -> np.ndarray:
    """Symmetric Kipf normalisation `D^{-1/2} (A + I) D^{-1/2}`."""
    n = adj.shape[0]
    if n == 0:
        return adj
    a = adj + np.eye(n, dtype=np.float32)
    deg = a.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    d = np.diag(deg_inv_sqrt)
    return d @ a @ d


@dataclass(slots=True)
class AgentGraphEmbedder:
    """Stateless 2-layer GCN with deterministic random weights."""

    in_dim: int
    hidden_dim: int = 128
    out_dim: int = 128
    _w0: np.ndarray = field(init=False, repr=False)
    _w1: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._w0 = _seeded_projection(self.in_dim, self.hidden_dim, layer=0)
        self._w1 = _seeded_projection(self.hidden_dim, self.out_dim, layer=1)

    @property
    def dim(self) -> int:
        return self.out_dim

    def embed(
        self,
        agent_graph: dict[str, Any],
        node_order: list[str],
        node_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(graph_vec, node_vecs)`.

        `graph_vec` is the mean-pooled node embedding (shape `(out_dim,)`);
        `node_vecs` is `(n, out_dim)` and is what the Phase-5 GNN
        controller uses for per-agent action heads."""
        n = len(node_order)
        if n == 0 or node_features.size == 0:
            empty_nodes = np.zeros((0, self.out_dim), dtype=np.float32)
            empty_graph = np.zeros(self.out_dim, dtype=np.float32)
            return empty_graph, empty_nodes
        if node_features.shape[0] != n:
            raise ValueError(
                f"node_features has {node_features.shape[0]} rows, expected {n} (one per node)"
            )
        if node_features.shape[1] != self.in_dim:
            raise ValueError(
                f"node_features has dim {node_features.shape[1]}, embedder was configured "
                f"with in_dim={self.in_dim}"
            )
        adj_norm = _normalise(_build_adjacency(agent_graph, node_order))
        h = adj_norm @ node_features.astype(np.float32, copy=False) @ self._w0
        h = np.maximum(h, 0.0, dtype=np.float32)
        h = adj_norm @ h @ self._w1
        h = np.maximum(h, 0.0, dtype=np.float32)
        graph_vec = h.mean(axis=0).astype(np.float32, copy=False)
        return graph_vec, h.astype(np.float32, copy=False)

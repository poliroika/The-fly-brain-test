"""`FlyGraphEmbedder` — spectral / node2vec embedding of the fly prior.

The fly connectome (or its compressed K-node version) is the structural
prior of the controller. Per `PLAN.md` Phase 4 §574: "node2vec на fly
graph + GraphSAGE pretrain (по бюджету)". Real node2vec needs `gensim`
or `torch_geometric`, neither of which we want as a hard dependency
of the unit-test suite.

Phase 4 ships a deterministic **spectral embedding** as the default:
the bottom `dim` non-trivial eigenvectors of the symmetric normalised
graph Laplacian. This:

* runs on CPU in milliseconds for `K ≤ 256` (the canonical compressed
  graphs);
* is deterministic for a given graph;
* is a known good *initial* prior — node2vec / GraphSAGE refine on
  top of it in Phase 8 once we have torch installed.

When ``backend == "node2vec"`` and ``torch`` is available we delegate
to `torch_geometric.nn.Node2Vec`. The unit tests only exercise the
spectral path; node2vec stays untested in CI but is wired up so
production training runs can flip the flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from flybrain.graph.dataclasses import FlyGraph

Backend = Literal["spectral", "node2vec"]


def _build_dense_adjacency(graph: FlyGraph) -> np.ndarray:
    n = int(graph.num_nodes)
    a = np.zeros((n, n), dtype=np.float32)
    for (s, t), w in zip(graph.edge_index, graph.edge_weight, strict=True):
        if 0 <= s < n and 0 <= t < n:
            a[s, t] += float(w)
    # Symmetrise — fly graph is directed but spectral methods need a
    # symmetric Laplacian.
    a = 0.5 * (a + a.T)
    return a


def _spectral_embedding(adj: np.ndarray, dim: int) -> np.ndarray:
    """Bottom `dim` non-trivial eigenvectors of the symmetric normalised
    Laplacian. Skips the trivial constant eigenvector (eigenvalue 0)."""
    n = adj.shape[0]
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float64)
    d_inv_sqrt = np.diag(deg_inv_sqrt)
    # Symmetric normalised Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
    a64 = adj.astype(np.float64)
    l_sym = np.eye(n, dtype=np.float64) - d_inv_sqrt @ a64 @ d_inv_sqrt
    # eigh returns ascending eigenvalues. Skip index 0 (≈0 eigenvalue
    # corresponding to the constant eigenvector).
    _eigvals, eigvecs = np.linalg.eigh(l_sym)
    # We want columns 1 .. dim+1 (avoid the trivial eigenvector).
    take = min(dim, max(0, n - 1))
    selected = eigvecs[:, 1 : 1 + take]
    if take < dim:
        # Pad with zeros if the graph is too small to provide `dim`
        # non-trivial eigenvectors.
        pad = np.zeros((n, dim - take), dtype=np.float64)
        selected = np.concatenate([selected, pad], axis=1)
    return selected.astype(np.float32, copy=False)


@dataclass(slots=True)
class FlyGraphEmbedder:
    dim: int = 64
    backend: Backend = "spectral"
    seed: int = 42

    def embed(self, fly_graph: FlyGraph) -> np.ndarray:
        """Return a `(num_nodes, dim)` float32 matrix."""
        adj = _build_dense_adjacency(fly_graph)
        if self.backend == "spectral":
            return _spectral_embedding(adj, self.dim)
        if self.backend == "node2vec":  # pragma: no cover - optional torch path
            return self._node2vec(fly_graph, adj)
        raise ValueError(f"unknown backend: {self.backend!r}")

    def _node2vec(self, graph: FlyGraph, adj: np.ndarray) -> np.ndarray:  # pragma: no cover
        try:
            import torch
            from torch_geometric.nn import Node2Vec
        except ImportError as e:
            raise RuntimeError(
                "node2vec backend requires torch + torch_geometric "
                "(install via `pip install -e .[ml]`)"
            ) from e
        n = int(graph.num_nodes)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        edge_index = (
            torch.tensor(np.asarray(graph.edge_index, dtype=np.int64), dtype=torch.long)
            .t()
            .contiguous()
        )
        torch.manual_seed(self.seed)
        model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=self.dim,
            walk_length=10,
            context_size=5,
            walks_per_node=5,
            num_negative_samples=1,
            sparse=False,
        )
        loader = model.loader(batch_size=64, shuffle=True, num_workers=0)
        opt = torch.optim.Adam(list(model.parameters()), lr=0.01)
        for _ in range(3):
            for pos_rw, neg_rw in loader:
                opt.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                opt.step()
        with torch.no_grad():
            emb = model.embedding.weight.detach().cpu().numpy()
        return emb.astype(np.float32, copy=False)

    def graph_vector(self, fly_graph: FlyGraph) -> np.ndarray:
        """Mean-pool the per-node spectral embedding into a single
        `(dim,)` vector. This is what `ControllerState` consumes."""
        emb = self.embed(fly_graph)
        if emb.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return emb.mean(axis=0).astype(np.float32, copy=False)

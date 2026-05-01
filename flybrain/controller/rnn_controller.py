"""`FlyBrainRNNController` (variant B — RNN with `A_fly` as sparse weight).

Stack:

    ControllerState
      -> StateEncoder
      -> A_fly-init Linear (agent_vecs)         | gradients flow,
      -> GRUCell over time (state_vec)          | hidden carries across ticks
      -> PolicyHeads

The fly graph is consumed at *construction* — its adjacency seeds the
weight of the per-agent linear layer. PLAN.md §580: "A_fly как sparse
weight". For Phase 5 we keep the layer dense (to match torch primitives)
but the initial weights are masked by the fly adjacency so the
controller starts off respecting the fly prior. Optimisation is
unconstrained from there.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from flybrain.controller._torch_base import TorchControllerBase
from flybrain.embeddings.state import ControllerState, ControllerStateBuilder
from flybrain.graph.dataclasses import FlyGraph


def _adjacency_from_fly_graph(fly_graph: FlyGraph | None, k: int) -> np.ndarray:
    """Project a `FlyGraph` onto a dense ``(k, k)`` adjacency matrix.

    This is a small, deterministic projection (top-k random sampling)
    chosen so the agent count `k` doesn't have to match the fly graph
    node count. The seed `42` keeps the projection reproducible across
    runs so the smoke test is deterministic.
    """
    if k == 0 or fly_graph is None or fly_graph.num_nodes == 0:
        return np.zeros((k, k), dtype=np.float32)

    rng = np.random.default_rng(42)
    n = int(fly_graph.num_nodes)
    # Sample ``k`` random fly-graph node ids and project the local
    # neighbourhood onto a ``(k, k)`` mask.
    pick = rng.choice(n, size=k, replace=(k > n))
    pick_set = {int(p): i for i, p in enumerate(pick)}

    adj = np.zeros((k, k), dtype=np.float32)
    for src, dst in fly_graph.edge_index:
        i = pick_set.get(int(src))
        j = pick_set.get(int(dst))
        if i is None or j is None or i == j:
            continue
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


class _AFlyLinear(nn.Module):
    """Linear layer whose weight is initialised to ``A_fly``-weighted
    Glorot-style noise."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        nn.init.zeros_(self.bias)
        self.dim = dim

    def init_from_fly(self, adj: np.ndarray) -> None:
        """Mask the per-agent weight with the fly adjacency.

        ``adj`` has shape ``(K, K)`` for K==num_agents. We tile it onto
        ``(dim, dim)`` by broadcasting (averaged), and use that as a
        soft mask on the existing Glorot-initialised weight.
        """
        if adj.size == 0:
            return
        # Compute a (dim, dim) soft mask: every dim block gets the same
        # adjacency pattern, so the prior is structural rather than
        # per-feature.
        if adj.shape != (self.dim, self.dim):
            # Resize via mean-pool / repeat to (dim, dim).
            target = np.zeros((self.dim, self.dim), dtype=np.float32)
            n = adj.shape[0]
            if n > 0:
                xs = np.linspace(0, n - 1, self.dim).astype(int)
                target[:] = adj[np.ix_(xs, xs)]
            adj = target
        mask = torch.as_tensor(adj, dtype=self.weight.dtype)
        # Soft mask: keep 25% of original weight + 75% scaled by adjacency.
        with torch.no_grad():
            self.weight.mul_(0.25 + 0.75 * mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class FlyBrainRNNController(TorchControllerBase):
    """Variant B — GRUCell over time + ``A_fly``-init agent linear."""

    name = "flybrain-rnn"

    def __init__(
        self,
        *,
        builder: ControllerStateBuilder,
        task_dim: int,
        agent_dim: int,
        graph_dim: int,
        trace_dim: int,
        fly_dim: int,
        produced_dim: int,
        hidden_dim: int = 128,
        head_hidden: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__(
            builder=builder,
            task_dim=task_dim,
            agent_dim=agent_dim,
            graph_dim=graph_dim,
            trace_dim=trace_dim,
            fly_dim=fly_dim,
            produced_dim=produced_dim,
            hidden_dim=hidden_dim,
            head_hidden=head_hidden,
            seed=seed,
        )
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.a_fly = _AFlyLinear(hidden_dim)

        self._hidden: torch.Tensor | None = None
        self._last_step_id: int | None = None
        self._last_task_id: str | None = None

    def init_from_fly_graph(self, fly_graph: FlyGraph | None, num_agents: int) -> None:
        if fly_graph is None or num_agents == 0:
            return
        adj = _adjacency_from_fly_graph(fly_graph, num_agents)
        self.a_fly.init_from_fly(adj)

    def reset_hidden(self) -> None:
        self._hidden = None
        self._last_step_id = None
        self._last_task_id = None

    def _maybe_reset_hidden(self, controller_state: ControllerState) -> None:
        # Reset across task boundaries so episodes don't leak hidden
        # state into one another. The runner gives us ``step_id``: a
        # decrement (or different task_type label) means new task.
        if controller_state.task_type != self._last_task_id:
            self._hidden = None
        elif self._last_step_id is not None and controller_state.step_id < self._last_step_id:
            self._hidden = None
        self._last_step_id = controller_state.step_id
        self._last_task_id = controller_state.task_type

    def _combine(
        self,
        state_vec: torch.Tensor,
        agent_vecs: torch.Tensor,
        controller_state: ControllerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._maybe_reset_hidden(controller_state)

        # 2-D batch dim for GRUCell.
        x = state_vec.unsqueeze(0)
        h_prev = self._hidden if self._hidden is not None else torch.zeros_like(x)
        h_new = self.cell(x, h_prev)
        # Detach across calls so the rolled-out gradient doesn't grow
        # without bound — Phase-6 will swap this for BPTT once we wire
        # the optimiser. The smoke test still gets a non-trivial graph
        # within a single tick.
        self._hidden = h_new.detach()
        new_state_vec = h_new.squeeze(0)

        if agent_vecs.numel() > 0:
            new_agent_vecs = self.a_fly(agent_vecs)
        else:
            new_agent_vecs = agent_vecs

        return new_state_vec, new_agent_vecs


__all__ = ["FlyBrainRNNController"]

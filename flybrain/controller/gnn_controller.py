"""`FlyBrainGNNController` (variant A — primary).

Stack:

    ControllerState
      -> StateEncoder (linear projections)
      -> AgentGNN (2-layer GCN over the live AgentGraph)
      -> PolicyHeads (kind / agent / edge / value / aux)

The GCN message-passing layer runs on torch tensors so gradients
flow through every parameter, satisfying PLAN.md §583's
"shape/grad smoke-test" exit criterion.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from flybrain.controller._torch_base import (
    TorchControllerBase,
    adjacency_from_agent_graph,
    normalised_adjacency,
)
from flybrain.embeddings.state import ControllerState, ControllerStateBuilder


class _GCNLayer(nn.Module):
    """Single Kipf GCN layer with a learnable linear projection."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return self.act(a_norm @ self.linear(x))


class FlyBrainGNNController(TorchControllerBase):
    """Variant A — torch GCN over the live AgentGraph."""

    name = "flybrain-gnn"

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
        gnn_hidden: int = 64,
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
        self.gnn1 = _GCNLayer(hidden_dim, gnn_hidden)
        self.gnn2 = _GCNLayer(gnn_hidden, hidden_dim)

    def _agent_graph_for(self, controller_state: ControllerState) -> dict[str, Any]:
        # Phase-5 wires the AgentGraph dict through ControllerState.extras
        # (the runner stashes it there). Fall back to a nodes-only graph
        # so the GNN smoke-tests can run without a runtime.
        ag = controller_state.extras.get("agent_graph") if controller_state.extras else None
        if isinstance(ag, dict):
            return ag
        return {"nodes": list(controller_state.agent_names), "edges": {}}

    def _combine(
        self,
        state_vec: torch.Tensor,
        agent_vecs: torch.Tensor,
        controller_state: ControllerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if agent_vecs.numel() == 0:
            return state_vec, agent_vecs

        agent_graph = self._agent_graph_for(controller_state)
        adj = adjacency_from_agent_graph(agent_graph, list(controller_state.agent_names))
        a_norm = torch.as_tensor(normalised_adjacency(adj), dtype=torch.float32)

        h = self.gnn1(agent_vecs, a_norm)
        h = self.gnn2(h, a_norm)

        # Pool agent reps into a global update of the state vector.
        agent_pool = h.mean(dim=0)
        state_vec = state_vec + agent_pool
        return state_vec, h


__all__ = ["FlyBrainGNNController"]

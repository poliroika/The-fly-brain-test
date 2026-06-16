"""`StateEncoder` — turn a `ControllerState` into a fixed-shape tensor.

Phase-5 controllers all share the same input contract: a
`ControllerState` (Phase 4 deliverable). The state encoder lifts the
numpy bag of features into torch tensors, projects them to a common
hidden size and concatenates them into a "state vector" the policy
heads can consume.

Per-agent vectors (used by the GNN / agent-attention paths) are
returned alongside as a `(K, D_hidden)` tensor.

The encoder has no learned parameters of its own beyond a couple of
linear projections; controllers wrap their own `StateEncoder` so each
controller can pick its own hidden size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from flybrain.embeddings.state import ControllerState


@dataclass(slots=True)
class EncoderShapes:
    """Resolved encoder dimensions, kept handy for tests + heads."""

    task_dim: int
    agent_dim: int
    graph_dim: int
    trace_dim: int
    fly_dim: int
    inbox_dim: int
    produced_dim: int
    hidden_dim: int

    @property
    def state_dim(self) -> int:
        """Concatenated dim *before* projection to ``hidden_dim``."""
        return (
            self.task_dim
            + self.graph_dim
            + self.trace_dim
            + self.fly_dim
            + self.produced_dim
            # global scalars: mean inbox, max inbox.
            + 2
        )


class StateEncoder(nn.Module):
    """Projects a `ControllerState` to ``(state_vec, agent_vecs)``.

    state_vec  : (B, hidden_dim) global state representation
    agent_vecs : (B, K, hidden_dim) per-agent representation
    """

    def __init__(
        self,
        *,
        task_dim: int,
        agent_dim: int,
        graph_dim: int,
        trace_dim: int,
        fly_dim: int,
        produced_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.shapes = EncoderShapes(
            task_dim=task_dim,
            agent_dim=agent_dim,
            graph_dim=graph_dim,
            trace_dim=trace_dim,
            fly_dim=fly_dim,
            inbox_dim=1,
            produced_dim=produced_dim,
            hidden_dim=hidden_dim,
        )
        # Per-agent: agent_vec ⊕ inbox scalar → hidden.
        self.agent_proj = nn.Linear(agent_dim + 1, hidden_dim)
        # Global: see EncoderShapes.state_dim.
        self.state_proj = nn.Linear(self.shapes.state_dim, hidden_dim)
        self.act = nn.GELU()

    @staticmethod
    def _t(x: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype)

    def encode(self, controller_state: ControllerState) -> tuple[torch.Tensor, torch.Tensor]:
        cs = controller_state
        task = self._t(cs.task_vec)
        agent_node = self._t(cs.agent_node_vecs)
        agent_graph = self._t(cs.agent_graph_vec)
        trace = self._t(cs.trace_vec)
        fly = self._t(cs.fly_vec)
        inbox = self._t(cs.inbox_vec)
        produced = self._t(cs.produced_mask)

        if inbox.numel() == 0:
            mean_inbox = torch.zeros(())
            max_inbox = torch.zeros(())
        else:
            mean_inbox = inbox.mean()
            max_inbox = inbox.max()

        global_vec = torch.cat(
            [
                task,
                agent_graph,
                trace,
                fly,
                produced,
                mean_inbox.unsqueeze(0),
                max_inbox.unsqueeze(0),
            ],
            dim=0,
        )
        state_vec = self.act(self.state_proj(global_vec))

        if agent_node.shape[0] == 0:
            agent_vecs = torch.zeros((0, self.shapes.hidden_dim), dtype=torch.float32)
        else:
            per_agent = torch.cat([agent_node, inbox.unsqueeze(-1)], dim=-1)
            agent_vecs = self.act(self.agent_proj(per_agent))

        return state_vec, agent_vecs

    def forward(
        self, controller_state: ControllerState
    ) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        return self.encode(controller_state)


__all__ = ["EncoderShapes", "StateEncoder"]

"""`LearnedRouterController` (variant C — attention + fly regularizer).

Stack:

    ControllerState
      -> StateEncoder
      -> Cross-attention (state_vec ⨂ agent_vecs)
      -> PolicyHeads
      +  fly_regularizer_loss = ||W_route - A_fly||^2 (training only)

The fly regularizer encourages the learned routing weights to stay
close to the fly graph adjacency. The penalty is provided through
``fly_regularizer_loss`` and is meant to be added to the policy loss
during Phase-7/8 training; it does **not** affect the masked argmax
inference path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from flybrain.controller._torch_base import TorchControllerBase
from flybrain.controller.rnn_controller import _adjacency_from_fly_graph
from flybrain.embeddings.state import ControllerState, ControllerStateBuilder
from flybrain.graph.dataclasses import FlyGraph


class _RoutingAttention(nn.Module):
    """Multi-head attention from the global state to per-agent vectors.

    Returns updated ``(state_vec', agent_vecs')`` plus the attention
    weights as a side-channel so the regularizer can pull them towards
    the fly adjacency.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 2) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self, state_vec: torch.Tensor, agent_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if agent_vecs.numel() == 0:
            self.last_attn_weights = torch.zeros((1, 0), dtype=state_vec.dtype)
            return state_vec, agent_vecs
        q = state_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        kv = agent_vecs.unsqueeze(0)  # (1, K, H)
        attended, weights = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        # weights : (1, 1, K) → (K,)
        self.last_attn_weights = weights.squeeze(0).squeeze(0)
        new_state = self.norm(state_vec + attended.squeeze(0).squeeze(0))
        # Per-agent: light residual so the routing decision can also
        # update local agent reps.
        agent_vecs = self.norm(agent_vecs + attended.squeeze(0))
        return new_state, agent_vecs


class LearnedRouterController(TorchControllerBase):
    """Variant C — attention router + fly regularizer hook."""

    name = "learned-router"

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
        num_heads: int = 2,
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
        self.routing = _RoutingAttention(hidden_dim, num_heads=num_heads)
        self._fly_prior: torch.Tensor | None = None

    def init_from_fly_graph(self, fly_graph: FlyGraph | None, num_agents: int) -> None:
        if fly_graph is None or num_agents == 0:
            self._fly_prior = None
            return
        adj = _adjacency_from_fly_graph(fly_graph, num_agents).astype(np.float32)
        if adj.size == 0:
            self._fly_prior = None
            return
        # Use the row-normalised neighbourhood as the per-agent prior
        # over routing weights.
        deg = adj.sum(axis=1, keepdims=True)
        deg = np.where(deg > 0, deg, 1.0)
        prior = adj / deg
        # Average across agents so the prior collapses to a (K,) vector
        # — the attention weights for picking the next agent.
        self._fly_prior = torch.as_tensor(prior.mean(axis=0), dtype=torch.float32)

    def fly_regularizer_loss(self) -> torch.Tensor:
        """Squared distance between last attention weights and the fly
        prior. Returns a ``0.0`` scalar tensor when no prior is set or
        the agent set is empty."""
        weights = self.routing.last_attn_weights
        if weights is None or self._fly_prior is None or weights.numel() == 0:
            return torch.zeros((), dtype=torch.float32)
        prior = self._fly_prior
        if prior.shape[0] != weights.shape[0]:
            # Project prior down by mean-pooling (or repeat) to match.
            new = torch.zeros(weights.shape[0], dtype=prior.dtype)
            stride = prior.shape[0] / max(1, weights.shape[0])
            for i in range(weights.shape[0]):
                lo = int(i * stride)
                hi = max(lo + 1, int((i + 1) * stride))
                new[i] = prior[lo:hi].mean()
            prior = new
        return torch.nn.functional.mse_loss(weights, prior)

    def _combine(
        self,
        state_vec: torch.Tensor,
        agent_vecs: torch.Tensor,
        controller_state: ControllerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.routing(state_vec, agent_vecs)


__all__ = ["LearnedRouterController"]

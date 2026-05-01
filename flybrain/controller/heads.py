"""Policy / value / aux heads consumed by Phase-5 controllers.

Each head is a small `nn.Module` so the same heads can be reused
across the GNN / RNN / Learned-Router controllers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from flybrain.controller.action_space import NUM_KINDS


def _mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class ActionKindHead(nn.Module):
    """Logits over the 9 GraphAction kinds."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = _mlp(in_dim, NUM_KINDS, hidden)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(state_vec)


class AgentHead(nn.Module):
    """Score every available agent given the global state vec.

    Implementation: dot-product attention between the state query and
    per-agent keys + a small MLP bias term. Returns logits of shape
    ``(K,)``.
    """

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.q = nn.Linear(in_dim, hidden)
        self.k = nn.Linear(in_dim, hidden)
        self.bias = nn.Linear(in_dim, 1)
        self.scale = float(hidden) ** -0.5

    def forward(self, state_vec: torch.Tensor, agent_vecs: torch.Tensor) -> torch.Tensor:
        if agent_vecs.shape[0] == 0:
            return torch.zeros(0, dtype=state_vec.dtype, device=state_vec.device)
        q = self.q(state_vec)  # (H,)
        k = self.k(agent_vecs)  # (K, H)
        logits = (k @ q) * self.scale  # (K,)
        logits = logits + self.bias(agent_vecs).squeeze(-1)
        return logits


class EdgeHead(nn.Module):
    """Two agent heads for picking ``(from, to)`` for edge-mutation
    actions. Self-loops are handled by the controller's masker."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.from_head = AgentHead(in_dim, hidden)
        self.to_head = AgentHead(in_dim, hidden)
        # Continuous scalar (weight or factor depending on kind) — clamped
        # by the decoder to a reasonable range.
        self.scalar = _mlp(in_dim, 1, hidden)

    def forward(
        self, state_vec: torch.Tensor, agent_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from_logits = self.from_head(state_vec, agent_vecs)
        to_logits = self.to_head(state_vec, agent_vecs)
        scalar = self.scalar(state_vec).squeeze(-1)
        return from_logits, to_logits, scalar


class ValueHead(nn.Module):
    """Scalar V(s)."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = _mlp(in_dim, 1, hidden)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(state_vec).squeeze(-1)


class AuxVerifierHead(nn.Module):
    """Predicts the upcoming verifier score in [0, 1] (auxiliary loss).

    PLAN.md §577-583 lists this as the third head — it gives the
    controller a denser learning signal during simulation and IL
    pretraining (Phases 6 + 7).
    """

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = _mlp(in_dim, 1, hidden)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(state_vec)).squeeze(-1)


@dataclass(slots=True)
class HeadOutputs:
    """All head outputs for one ControllerState forward pass.

    Fields are *raw* logits / scalars; sampling + masking happens in
    the controller's `select_action` path.
    """

    kind_logits: torch.Tensor
    agent_logits: torch.Tensor
    edge_from_logits: torch.Tensor
    edge_to_logits: torch.Tensor
    edge_scalar: torch.Tensor
    value: torch.Tensor
    aux_verifier: torch.Tensor


class PolicyHeads(nn.Module):
    """Bundle of all five heads, sharing the same ``in_dim``."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.kind = ActionKindHead(in_dim, hidden)
        self.agent = AgentHead(in_dim, hidden)
        self.edge = EdgeHead(in_dim, hidden)
        self.value = ValueHead(in_dim, hidden)
        self.aux = AuxVerifierHead(in_dim, hidden)

    def forward(self, state_vec: torch.Tensor, agent_vecs: torch.Tensor) -> HeadOutputs:
        kind_logits = self.kind(state_vec)
        agent_logits = self.agent(state_vec, agent_vecs)
        e_from, e_to, e_scalar = self.edge(state_vec, agent_vecs)
        value = self.value(state_vec)
        aux = self.aux(state_vec)
        return HeadOutputs(
            kind_logits=kind_logits,
            agent_logits=agent_logits,
            edge_from_logits=e_from,
            edge_to_logits=e_to,
            edge_scalar=e_scalar,
            value=value,
            aux_verifier=aux,
        )


__all__ = [
    "ActionKindHead",
    "AgentHead",
    "AuxVerifierHead",
    "EdgeHead",
    "HeadOutputs",
    "PolicyHeads",
    "ValueHead",
]

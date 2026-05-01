"""Shared boilerplate for the three Phase-5 torch controllers.

All Phase-5 controllers:

1. Hold a `ControllerStateBuilder` so they satisfy the
   `flybrain.controller.base.Controller` Protocol — `select_action`
   takes a `RuntimeState` and produces a `GraphAction` JSON dict.
2. Wrap a `StateEncoder` + `PolicyHeads` (subclasses add their own
   variant-specific layer between the encoder and the heads).
3. Expose `forward(controller_state) -> HeadOutputs` for training,
   bypassing the masker so gradients flow through every head.

Subclasses override `_combine(state_vec, agent_vecs, controller_state)
-> tuple[state_vec', agent_vecs']`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from flybrain.controller.action_space import (
    KIND_ACTIVATE_AGENT,
    KIND_ADD_EDGE,
    KIND_REMOVE_EDGE,
    KIND_SCALE_EDGE,
    NUM_KINDS,
    ActionMask,
    ActionSpace,
)
from flybrain.controller.encoder import StateEncoder
from flybrain.controller.heads import HeadOutputs, PolicyHeads
from flybrain.embeddings.state import ControllerState, ControllerStateBuilder
from flybrain.runtime.state import RuntimeState


@dataclass(slots=True)
class ControllerForwardResult:
    """Bundle of `HeadOutputs` + the masked tensors used for sampling."""

    raw: HeadOutputs
    masked_kind_logits: torch.Tensor
    masked_agent_logits: torch.Tensor
    masked_edge_from_logits: torch.Tensor
    masked_edge_to_logits: torch.Tensor
    action_mask: ActionMask


class TorchControllerBase(nn.Module):
    """Common scaffolding for the three Phase-5 controllers.

    Subclasses must:

    * call ``super().__init__(builder=..., hidden_dim=...)``,
    * implement ``_combine(state_vec, agent_vecs, controller_state)``,
      which returns the (possibly transformed) ``(state_vec, agent_vecs)``
      to feed into the heads.
    """

    name: str = "torch-controller"

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
        super().__init__()
        self.builder = builder
        self._init_torch_seed(seed)
        self.encoder = StateEncoder(
            task_dim=task_dim,
            agent_dim=agent_dim,
            graph_dim=graph_dim,
            trace_dim=trace_dim,
            fly_dim=fly_dim,
            produced_dim=produced_dim,
            hidden_dim=hidden_dim,
        )
        self.heads = PolicyHeads(in_dim=hidden_dim, hidden=head_hidden)

    @staticmethod
    def _init_torch_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)

    # -- subclasses override this to inject GNN/RNN/Router transforms ----------
    def _combine(
        self,
        state_vec: torch.Tensor,
        agent_vecs: torch.Tensor,
        controller_state: ControllerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    # -- training-time forward (no masking, raw logits + value/aux) -----------
    def forward(self, controller_state: ControllerState) -> HeadOutputs:
        state_vec, agent_vecs = self.encoder.encode(controller_state)
        state_vec, agent_vecs = self._combine(state_vec, agent_vecs, controller_state)
        return self.heads(state_vec, agent_vecs)

    # -- inference-time forward (logits + masked logits + value) --------------
    def _masked_forward(
        self, controller_state: ControllerState, action_space: ActionSpace
    ) -> ControllerForwardResult:
        raw = self.forward(controller_state)
        action_mask = action_space.legal_mask()

        kind_mask = torch.as_tensor(action_mask.kind_mask, dtype=torch.float32)
        masked_kind = _apply_mask(raw.kind_logits, kind_mask)

        if raw.agent_logits.numel() > 0:
            agent_mask = torch.as_tensor(action_mask.agent_mask, dtype=torch.float32)
            masked_agent = _apply_mask(raw.agent_logits, agent_mask)
        else:
            masked_agent = raw.agent_logits

        if raw.edge_from_logits.numel() > 0:
            # Use a uniform per-agent mask for from/to (self-loop fix is
            # done after sampling the from index — once we know `from`
            # we mask out that entry in the to logits).
            agent_mask = torch.as_tensor(action_mask.agent_mask, dtype=torch.float32)
            masked_from = _apply_mask(raw.edge_from_logits, agent_mask)
            masked_to = _apply_mask(raw.edge_to_logits, agent_mask)
        else:
            masked_from = raw.edge_from_logits
            masked_to = raw.edge_to_logits

        return ControllerForwardResult(
            raw=raw,
            masked_kind_logits=masked_kind,
            masked_agent_logits=masked_agent,
            masked_edge_from_logits=masked_from,
            masked_edge_to_logits=masked_to,
            action_mask=action_mask,
        )

    # -- protocol: select_action(RuntimeState) ---------------------------------
    @torch.no_grad()
    def select_action(self, state: RuntimeState) -> dict[str, Any]:
        cs = self.builder.from_runtime_sync(state)
        action_space = ActionSpace(agent_names=list(cs.agent_names))
        result = self._masked_forward(cs, action_space)

        kind_id = int(torch.argmax(result.masked_kind_logits).item())

        agent_id: int | None = None
        if kind_id == KIND_ACTIVATE_AGENT and result.masked_agent_logits.numel() > 0:
            agent_id = int(torch.argmax(result.masked_agent_logits).item())

        edge_from_id: int | None = None
        edge_to_id: int | None = None
        edge_scalar: float | None = None
        if (
            kind_id in (KIND_ADD_EDGE, KIND_REMOVE_EDGE, KIND_SCALE_EDGE)
            and result.masked_edge_from_logits.numel() > 1
        ):
            edge_from_id = int(torch.argmax(result.masked_edge_from_logits).item())
            # Mask out the self-loop on the to head before argmax.
            to_logits = result.masked_edge_to_logits.clone()
            to_logits[edge_from_id] = -math.inf
            edge_to_id = int(torch.argmax(to_logits).item())
            edge_scalar = float(result.raw.edge_scalar.item())

        return action_space.decode(
            kind_id,
            agent_id=agent_id,
            edge_from_id=edge_from_id,
            edge_to_id=edge_to_id,
            edge_weight=edge_scalar,
        )

    # ---------- training helpers ----------
    def value_loss(self, controller_state: ControllerState, target: float) -> torch.Tensor:
        out = self.forward(controller_state)
        target_t = torch.as_tensor(target, dtype=out.value.dtype)
        return torch.nn.functional.mse_loss(out.value, target_t)

    def aux_verifier_loss(self, controller_state: ControllerState, target: float) -> torch.Tensor:
        out = self.forward(controller_state)
        target_t = torch.as_tensor(target, dtype=out.aux_verifier.dtype).clamp(0.0, 1.0)
        return torch.nn.functional.binary_cross_entropy(out.aux_verifier, target_t)


def _apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Use a large negative finite number to keep softmax numerically
    # well-behaved while still effectively masking out entries.
    return torch.where(mask > 0.5, logits, torch.full_like(logits, -1e9))


def adjacency_from_agent_graph(
    agent_graph: dict[str, Any],
    node_order: list[str],
) -> np.ndarray:
    """Build a symmetric ``(K, K)`` adjacency matrix from an agent-graph dict.

    Same routine as `flybrain.embeddings.graph_emb.AgentGraphEmbedder._adjacency`
    but lifted up to a shared helper so both the GCN embedder *and* the
    Phase-5 controllers use the same convention.
    """
    k = len(node_order)
    adj = np.eye(k, dtype=np.float32)
    if k == 0:
        return adj

    index = {name: i for i, name in enumerate(node_order)}
    edges = agent_graph.get("edges", {}) if isinstance(agent_graph, dict) else {}
    if isinstance(edges, dict):
        for src, dsts in edges.items():
            i = index.get(src)
            if i is None:
                continue
            if isinstance(dsts, (list, tuple)):
                for dst in dsts:
                    j = index.get(dst)
                    if j is None or i == j:
                        continue
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
            elif isinstance(dsts, dict):
                for dst, weight in dsts.items():
                    j = index.get(dst)
                    if j is None or i == j:
                        continue
                    w = float(weight)
                    adj[i, j] = max(adj[i, j], w)
                    adj[j, i] = max(adj[j, i], w)
    return adj


def normalised_adjacency(adj: np.ndarray) -> np.ndarray:
    """Kipf-style symmetric normalisation of a (K, K) adjacency matrix."""
    k = adj.shape[0]
    if k == 0:
        return adj.astype(np.float32, copy=False)
    deg = adj.sum(axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d = np.diag(inv_sqrt.astype(np.float32))
    return (d @ adj.astype(np.float32) @ d).astype(np.float32, copy=False)


__all__ = [
    "NUM_KINDS",
    "ControllerForwardResult",
    "TorchControllerBase",
    "adjacency_from_agent_graph",
    "normalised_adjacency",
]

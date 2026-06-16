"""`ActionSpace` — the discrete action space exposed to Phase-5 controllers.

The action layout mirrors the Rust `GraphAction` enum
(`crates/flybrain-core/src/action.rs`). The discriminants are stable so
that trained controller weights can be reloaded across versions:

| id | kind             | extra fields                                    |
|----|------------------|-------------------------------------------------|
| 0  | activate_agent   | agent (sampled from the K available agents)     |
| 1  | add_edge         | from, to (sampled), weight (default 1.0)        |
| 2  | remove_edge      | from, to (sampled)                              |
| 3  | scale_edge       | from, to (sampled), factor (sampled scalar)     |
| 4  | call_memory      | -                                               |
| 5  | call_retriever   | -                                               |
| 6  | call_tool_executor | -                                             |
| 7  | call_verifier    | -                                               |
| 8  | terminate        | -                                               |

Phase-5 keeps the action space minimal: kind selection over 9 IDs +
two agent picks (one for ``activate_agent`` and one ``(from, to)`` pair
for the three edge ops). Continuous outputs (``weight`` / ``factor``)
are produced by a small scalar head; for the smoke tests they are
sampled deterministically (mean of a Normal head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Stable kind IDs. Mirror `GraphAction::discriminant`.
KIND_ACTIVATE_AGENT = 0
KIND_ADD_EDGE = 1
KIND_REMOVE_EDGE = 2
KIND_SCALE_EDGE = 3
KIND_CALL_MEMORY = 4
KIND_CALL_RETRIEVER = 5
KIND_CALL_TOOL_EXECUTOR = 6
KIND_CALL_VERIFIER = 7
KIND_TERMINATE = 8

NUM_KINDS = 9

KIND_NAMES: tuple[str, ...] = (
    "activate_agent",
    "add_edge",
    "remove_edge",
    "scale_edge",
    "call_memory",
    "call_retriever",
    "call_tool_executor",
    "call_verifier",
    "terminate",
)

# Action kinds that need an agent (single index).
_NEEDS_AGENT: frozenset[int] = frozenset({KIND_ACTIVATE_AGENT})

# Action kinds that need a (from, to) edge selection.
_NEEDS_EDGE: frozenset[int] = frozenset({KIND_ADD_EDGE, KIND_REMOVE_EDGE, KIND_SCALE_EDGE})


@dataclass(slots=True)
class ActionMask:
    """Per-tick legal-action mask emitted by `ActionSpace.legal_mask`."""

    kind_mask: np.ndarray  # (NUM_KINDS,) float32, 1.0 == legal
    agent_mask: np.ndarray  # (K,) float32, 1.0 == legal
    edge_mask: np.ndarray  # (K, K) float32, 1.0 == legal (excludes self-loops)


@dataclass(slots=True)
class ActionSpace:
    """Decoder + masker for the controller's discrete action heads.

    Construct one per MAS tick (cheap — just dataclass + a couple of
    masks). The decoder takes integer choices from the heads and
    produces the JSON-serialised `GraphAction` dict the runner consumes.
    """

    agent_names: list[str]

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)

    def legal_mask(self) -> ActionMask:
        k = self.num_agents
        kind_mask = np.ones(NUM_KINDS, dtype=np.float32)
        agent_mask = np.ones(k, dtype=np.float32) if k > 0 else np.zeros(0, dtype=np.float32)
        edge_mask = (
            (1.0 - np.eye(k, dtype=np.float32)) if k > 0 else np.zeros((0, 0), dtype=np.float32)
        )

        if k == 0:
            kind_mask[KIND_ACTIVATE_AGENT] = 0.0
        if k < 2:
            for kid in _NEEDS_EDGE:
                kind_mask[kid] = 0.0

        return ActionMask(kind_mask=kind_mask, agent_mask=agent_mask, edge_mask=edge_mask)

    def decode(
        self,
        kind_id: int,
        *,
        agent_id: int | None = None,
        edge_from_id: int | None = None,
        edge_to_id: int | None = None,
        edge_weight: float | None = None,
    ) -> dict[str, Any]:
        """Project head outputs onto a `GraphAction` JSON dict.

        The runtime ignores unknown fields, so missing values are fine
        for kinds that don't use them.
        """
        if not 0 <= kind_id < NUM_KINDS:
            raise ValueError(f"kind_id {kind_id} out of range [0, {NUM_KINDS})")
        kind = KIND_NAMES[kind_id]

        if kind_id in _NEEDS_AGENT:
            if agent_id is None or not 0 <= agent_id < self.num_agents:
                # Mask must have prevented this; fall back to terminate.
                return {"kind": "terminate"}
            return {"kind": kind, "agent": self.agent_names[agent_id]}

        if kind_id in _NEEDS_EDGE:
            if (
                edge_from_id is None
                or edge_to_id is None
                or not 0 <= edge_from_id < self.num_agents
                or not 0 <= edge_to_id < self.num_agents
                or edge_from_id == edge_to_id
            ):
                return {"kind": "terminate"}
            payload: dict[str, Any] = {
                "kind": kind,
                "from": self.agent_names[edge_from_id],
                "to": self.agent_names[edge_to_id],
            }
            if kind_id == KIND_ADD_EDGE:
                payload["weight"] = float(edge_weight) if edge_weight is not None else 1.0
            elif kind_id == KIND_SCALE_EDGE:
                payload["factor"] = float(edge_weight) if edge_weight is not None else 1.0
            return payload

        return {"kind": kind}


__all__ = [
    "KIND_ACTIVATE_AGENT",
    "KIND_ADD_EDGE",
    "KIND_CALL_MEMORY",
    "KIND_CALL_RETRIEVER",
    "KIND_CALL_TOOL_EXECUTOR",
    "KIND_CALL_VERIFIER",
    "KIND_NAMES",
    "KIND_REMOVE_EDGE",
    "KIND_SCALE_EDGE",
    "KIND_TERMINATE",
    "NUM_KINDS",
    "ActionMask",
    "ActionSpace",
]

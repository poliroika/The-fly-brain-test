"""`RandomController` — baseline that picks a random valid action.

Used (a) as a sanity check that the runtime survives any sequence of
actions the action space allows, and (b) as the §15.C "random sparse"
baseline once we wire it into the benchmark loop in Phase 9.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from flybrain.runtime.state import RuntimeState


@dataclass(slots=True)
class RandomController:
    name: str = "random"
    seed: int = 42
    p_terminate: float = 0.05
    """Per-step probability of emitting `terminate`. Keeps episodes
    bounded even if every other action is sampled with equal weight."""

    p_activate: float = 0.6
    """Probability mass given to `activate_agent`. The remaining mass is
    split equally between the four call_* actions."""

    _rng: random.Random = None  # type: ignore[assignment]
    _last_task_id: str | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._last_task_id = None

    def select_action(self, state: RuntimeState) -> dict[str, Any]:
        if state.task_id != self._last_task_id:
            self._rng = random.Random(self.seed ^ hash(state.task_id) & 0xFFFFFFFF)
            self._last_task_id = state.task_id

        u = self._rng.random()
        if u < self.p_terminate:
            return {"kind": "terminate"}
        if u < self.p_terminate + self.p_activate and state.available_agents:
            agent = self._rng.choice(state.available_agents)
            return {"kind": "activate_agent", "agent": agent}
        # Distribute the rest among the call_* actions.
        rest = self._rng.choice(
            ["call_memory", "call_retriever", "call_tool_executor", "call_verifier"]
        )
        return {"kind": rest}

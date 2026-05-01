"""A trivial scheduling baseline: cycle through ``available_agents``
in order, calling the verifier once at the end, then terminating.

Not part of the canonical 9-baseline list but useful as a *floor*
for benchmark comparisons (any controller worth its salt should
beat round-robin on success rate, even if the round-robin schedule
is a competent default for some task types)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flybrain.runtime.state import RuntimeState


@dataclass(slots=True)
class RoundRobinController:
    name: str = "round_robin"
    _cursor: int = field(default=0, init=False)
    _last_task_id: str | None = field(default=None, init=False)
    _verifier_called: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._cursor = 0
        self._last_task_id = None
        self._verifier_called = False

    def select_action(self, state: RuntimeState) -> dict[str, Any]:
        if state.task_id != self._last_task_id:
            self._cursor = 0
            self._last_task_id = state.task_id
            self._verifier_called = False

        agents = list(state.available_agents)
        if not agents:
            return {"kind": "terminate"}

        if self._cursor < len(agents):
            agent = agents[self._cursor]
            self._cursor += 1
            return {"kind": "activate_agent", "agent": agent}

        if not self._verifier_called:
            self._verifier_called = True
            return {"kind": "call_verifier"}

        return {"kind": "terminate"}

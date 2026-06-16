"""`ManualController` — scripted, hand-tuned plan per task type.

This is the *baseline-of-baselines* (README §15.A): a fixed agent
ordering selected to be a competent (but not optimal) MAS plan. The
RL/learned controllers should beat it once Phase 8 lands; for Phase 2
it is the controller used by `tests/python/integration/test_mas_runtime_mock.py`
to exercise the runtime end-to-end.

The plan is encoded as a list of `(action_template, prerequisite)` pairs.
At each tick the controller walks the plan from the top and emits the
first action whose prerequisite is satisfied (or always-satisfied) and
that has not yet been emitted in the current run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flybrain.runtime.state import RuntimeState

PLANS: dict[str, list[dict[str, Any]]] = {
    "coding": [
        {"kind": "activate_agent", "agent": "Planner"},
        {"kind": "activate_agent", "agent": "Coder"},
        {"kind": "activate_agent", "agent": "TestRunner"},
        {"kind": "call_verifier"},
        {"kind": "activate_agent", "agent": "Finalizer"},
        {"kind": "terminate"},
    ],
    "math": [
        {"kind": "activate_agent", "agent": "Planner"},
        {"kind": "activate_agent", "agent": "MathSolver"},
        {"kind": "call_verifier"},
        {"kind": "activate_agent", "agent": "Finalizer"},
        {"kind": "terminate"},
    ],
    "research": [
        {"kind": "activate_agent", "agent": "Planner"},
        {"kind": "activate_agent", "agent": "Retriever"},
        {"kind": "activate_agent", "agent": "Researcher"},
        {"kind": "call_verifier"},
        {"kind": "activate_agent", "agent": "Finalizer"},
        {"kind": "terminate"},
    ],
    "tool_use": [
        {"kind": "activate_agent", "agent": "Planner"},
        {"kind": "activate_agent", "agent": "ToolExecutor"},
        {"kind": "call_verifier"},
        {"kind": "activate_agent", "agent": "Finalizer"},
        {"kind": "terminate"},
    ],
    "synthetic_routing": [
        {"kind": "activate_agent", "agent": "Planner"},
        {"kind": "activate_agent", "agent": "TaskDecomposer"},
        {"kind": "call_verifier"},
        {"kind": "activate_agent", "agent": "Finalizer"},
        {"kind": "terminate"},
    ],
}

DEFAULT_PLAN = [
    {"kind": "activate_agent", "agent": "Planner"},
    {"kind": "activate_agent", "agent": "Finalizer"},
    {"kind": "terminate"},
]


@dataclass(slots=True)
class ManualController:
    name: str = "manual"
    plans: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: PLANS)
    _cursor: int = field(default=0, init=False)
    _last_task_id: str | None = field(default=None, init=False)

    def reset(self) -> None:
        self._cursor = 0
        self._last_task_id = None

    def _plan_for(self, state: RuntimeState) -> list[dict[str, Any]]:
        return self.plans.get(state.task_type, DEFAULT_PLAN)

    def select_action(self, state: RuntimeState) -> dict[str, Any]:
        if state.task_id != self._last_task_id:
            self._cursor = 0
            self._last_task_id = state.task_id

        plan = self._plan_for(state)

        # Skip activate_agent calls for unknown agents (e.g. when running with
        # `MINIMAL_15` and the plan references `Researcher` from EXTENDED_25).
        # Skip until we hit a known one or fall off the end.
        while self._cursor < len(plan):
            step = plan[self._cursor]
            self._cursor += 1
            if step["kind"] == "activate_agent" and step["agent"] not in state.available_agents:
                continue
            return step
        return {"kind": "terminate"}

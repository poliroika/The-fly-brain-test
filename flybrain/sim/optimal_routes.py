"""Ground-truth optimal routes per task type (README §12.1).

The routes use names from `flybrain.agents.specs.MINIMAL_15` and
`EXTENDED_25` so the supervised pretrain trains the controller on the
same agent vocabulary the real runner uses.

Each agent in the route also produces a tagged ``component``. The
runtime tracks these in `RuntimeState.produced_components` and the
supervised pretrain uses them as side-channel features so the
controller learns "given what's already produced, do X next".
"""

from __future__ import annotations

# Map agent name -> the component tag the runtime would observe after
# the agent fires. These must be present in the
# `ControllerStateBuilder._DEFAULT_COMPONENT_TAGS` list (Phase 4) for
# the masking heads to pick them up.
_AGENT_TO_COMPONENT: dict[str, str] = {
    "Planner": "plan",
    "TaskDecomposer": "plan",
    "Coder": "code",
    "TestRunner": "tests_run",
    "Debugger": "code",
    "MathSolver": "code",
    "Researcher": "plan",
    "Retriever": "tool_used",
    "ToolExecutor": "tool_used",
    "CitationChecker": "verifier_called",
    "Critic": "verifier_called",
    "Verifier": "verifier_called",
    "Judge": "verifier_called",
    "SchemaValidator": "verifier_called",
    "Finalizer": "final_answer",
}


def component_for_agent(agent_name: str) -> str:
    """Return the component tag the runtime would observe after
    ``agent_name`` fires. Falls back to ``"plan"`` for agents not in
    the map (so unknown specs don't crash the pretrain loop)."""
    return _AGENT_TO_COMPONENT.get(agent_name, "plan")


# README §12.1 routes. Names match `flybrain.agents.specs`.
OPTIMAL_ROUTES: dict[str, list[str]] = {
    "coding": ["Planner", "Coder", "TestRunner", "Debugger", "Verifier"],
    "math": ["Planner", "MathSolver", "Critic", "Verifier"],
    "research": [
        "Planner",
        "Researcher",
        "Retriever",
        "CitationChecker",
        "Finalizer",
    ],
    "tool_use": ["Planner", "ToolExecutor", "SchemaValidator", "Verifier"],
}

TASK_TYPES: tuple[str, ...] = tuple(OPTIMAL_ROUTES)


def optimal_action_at(task_type: str, step: int) -> dict[str, str]:
    """Return the GraphAction dict the *expert* would emit at ``step``.

    The route stops when ``step >= len(route)``: the expert emits
    ``terminate``.
    """
    route = OPTIMAL_ROUTES.get(task_type)
    if route is None:
        raise KeyError(f"unknown task_type {task_type!r}")
    if step >= len(route):
        return {"kind": "terminate"}
    return {"kind": "activate_agent", "agent": route[step]}


__all__ = [
    "OPTIMAL_ROUTES",
    "TASK_TYPES",
    "component_for_agent",
    "optimal_action_at",
]

"""`RuntimeState` — what the controller observes at each tick.

Phase 5 (the actual GNN / RNN / Learned-Router controllers) will turn this
into tensors via `flybrain.embeddings`. For Phase 2 it is a plain
dataclass that controllers (`ManualController`, `RandomController`) read
field-by-field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeState:
    """Snapshot fed to `Controller.select_action(state)` each tick."""

    task_id: str
    task_type: str
    prompt: str

    step_id: int
    """Current scheduler step counter (0-indexed)."""

    available_agents: list[str]
    """Agents currently registered in the AgentGraph (i.e. callable)."""

    pending_inbox: dict[str, int]
    """`agent_name -> num pending messages` from the message bus."""

    last_active_agent: str | None
    """Most recently activated agent (carried across pure-graph mutations)."""

    last_output_summary: str | None = None
    """Short summary of what the last agent produced (passed straight from
    the Python runner)."""

    last_verifier_score: float | None = None
    last_verifier_passed: bool | None = None
    last_verifier_failed_component: str | None = None

    last_errors: list[str] = field(default_factory=list)

    produced_components: set[str] = field(default_factory=set)
    """Set of `(role|name)` tags the runtime has observed in agent
    outputs so far. Used by `ManualController` to decide whether
    prerequisites are satisfied. Examples: `"plan"`, `"code"`,
    `"tests_run"`, `"final_answer"`."""

    totals_tokens: int = 0
    totals_calls: int = 0
    totals_cost_rub: float = 0.0

    extras: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for controllers that want to stash their own state."""

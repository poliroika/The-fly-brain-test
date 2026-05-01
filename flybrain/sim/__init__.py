"""Cheap MAS simulation used for Phase-6 supervised pretraining.

The simulation is *not* an LLM call — every step is deterministic
(or seedable RNG) and runs in microseconds, so the controller can be
trained on millions of state→action pairs at zero API cost. The
simulation exposes the same `RuntimeState` surface as the real
runner, which means the same controllers can be plugged in without
any code changes.

Phase-7 (Imitation) overlays expert traces from real LLM runs on top
of this synthetic dataset so the learner doesn't degrade when it
hits real tasks.
"""

from __future__ import annotations

from flybrain.sim.optimal_routes import (
    OPTIMAL_ROUTES,
    TASK_TYPES,
    component_for_agent,
    optimal_action_at,
)
from flybrain.sim.synthetic_mas import SyntheticMAS, SyntheticOutcome
from flybrain.sim.task_generator import SyntheticTask, TaskGenerator

__all__ = [
    "OPTIMAL_ROUTES",
    "TASK_TYPES",
    "SyntheticMAS",
    "SyntheticOutcome",
    "SyntheticTask",
    "TaskGenerator",
    "component_for_agent",
    "optimal_action_at",
]

"""`SyntheticMAS` — a tiny LLM-free MAS simulator.

Walks an arbitrary controller through a synthetic task and returns a
trace + per-step `RuntimeState` snapshots. The simulator is what
makes Phase-6 simulation pretraining cheap: no Yandex calls, no
torch model in the loop (only the controller's `select_action`
forward), microsecond per step on CPU.

Per-agent dynamics (PLAN.md §587):

* ``success_prob[task_type]`` — probability the agent's component is
  produced when activated.
* ``cost`` — fake-rouble cost of activating the agent (used by the
  Phase-8 reward shaping later).
* ``error_rate`` — probability the agent emits an error instead of
  a useful component.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from flybrain.controller.base import Controller
from flybrain.runtime.state import RuntimeState
from flybrain.sim.optimal_routes import OPTIMAL_ROUTES, component_for_agent
from flybrain.sim.task_generator import SyntheticTask

# Per-agent configuration. Real Phase-6 runs override these via a
# YAML config; the defaults here are picked so the optimal route
# converges to ~0.85 success on a tiny held-out set.
_DEFAULT_AGENT_CONFIG: dict[str, dict[str, float]] = {
    "Planner": {"coding": 0.95, "math": 0.95, "research": 0.95, "tool_use": 0.95},
    "Coder": {"coding": 0.85, "math": 0.20, "research": 0.10, "tool_use": 0.20},
    "TestRunner": {"coding": 0.85, "math": 0.10, "research": 0.05, "tool_use": 0.05},
    "Debugger": {"coding": 0.80, "math": 0.10, "research": 0.05, "tool_use": 0.05},
    "MathSolver": {"coding": 0.10, "math": 0.85, "research": 0.10, "tool_use": 0.05},
    "Researcher": {"coding": 0.10, "math": 0.10, "research": 0.85, "tool_use": 0.10},
    "Retriever": {"coding": 0.20, "math": 0.05, "research": 0.85, "tool_use": 0.40},
    "CitationChecker": {"coding": 0.05, "math": 0.05, "research": 0.85, "tool_use": 0.05},
    "Critic": {"coding": 0.40, "math": 0.85, "research": 0.30, "tool_use": 0.30},
    "Verifier": {"coding": 0.85, "math": 0.85, "research": 0.20, "tool_use": 0.85},
    "Judge": {"coding": 0.30, "math": 0.30, "research": 0.30, "tool_use": 0.30},
    "Finalizer": {"coding": 0.30, "math": 0.30, "research": 0.85, "tool_use": 0.30},
    "ToolExecutor": {"coding": 0.30, "math": 0.10, "research": 0.20, "tool_use": 0.85},
    "SchemaValidator": {"coding": 0.20, "math": 0.10, "research": 0.10, "tool_use": 0.85},
    "TaskDecomposer": {"coding": 0.50, "math": 0.50, "research": 0.50, "tool_use": 0.50},
    "MemoryReader": {"coding": 0.20, "math": 0.20, "research": 0.30, "tool_use": 0.20},
    "MemoryWriter": {"coding": 0.20, "math": 0.20, "research": 0.30, "tool_use": 0.20},
}

_DEFAULT_AGENT_COST: dict[str, float] = {}
_DEFAULT_AGENT_ERROR_RATE: dict[str, float] = {}


@dataclass(slots=True)
class SyntheticOutcome:
    """Result of running one synthetic task through the simulator."""

    task: SyntheticTask
    success: bool
    final_score: float
    actions: list[dict[str, Any]] = field(default_factory=list)
    states: list[RuntimeState] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    cost: float = 0.0
    steps: int = 0


@dataclass(slots=True)
class SyntheticMAS:
    """Stateful simulator: tracks per-task RuntimeState, success_prob,
    cost, error_rate."""

    agent_names: list[str]
    success_prob: dict[str, dict[str, float]] = field(
        default_factory=lambda: dict(_DEFAULT_AGENT_CONFIG)
    )
    cost_per_agent: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_AGENT_COST))
    error_rate: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_AGENT_ERROR_RATE))
    max_steps: int = 12
    seed: int = 0

    def _initial_state(self, task: SyntheticTask) -> RuntimeState:
        return RuntimeState(
            task_id=task.task_id,
            task_type=task.task_type,
            prompt=task.prompt,
            step_id=0,
            available_agents=list(self.agent_names),
            pending_inbox={},
            last_active_agent=None,
        )

    def _agent_success(self, agent: str, task_type: str, rng: random.Random) -> bool:
        prob_table = self.success_prob.get(agent, {})
        prob = prob_table.get(task_type, 0.2)
        return rng.random() < prob

    def _final_score(self, task: SyntheticTask, produced: set[str]) -> float:
        """Crude success heuristic: fraction of optimal-route components
        that ended up in ``produced``."""
        route = OPTIMAL_ROUTES.get(task.task_type, [])
        if not route:
            return 0.0
        wanted = {component_for_agent(a) for a in route}
        if not wanted:
            return 0.0
        hit = sum(1 for tag in wanted if tag in produced)
        return hit / len(wanted)

    def run(self, controller: Controller, task: SyntheticTask) -> SyntheticOutcome:
        rng = random.Random(self.seed + hash(task.task_id) & 0xFFFFFFFF)
        state = self._initial_state(task)

        actions: list[dict[str, Any]] = []
        states: list[RuntimeState] = []
        rewards: list[float] = []
        total_cost = 0.0

        for _step in range(self.max_steps):
            states.append(_snapshot(state))
            action = controller.select_action(state)
            actions.append(action)

            kind = action.get("kind", "terminate")
            reward = 0.0

            if kind == "terminate":
                rewards.append(reward)
                break

            if kind == "activate_agent":
                agent = action.get("agent")
                if agent in self.agent_names:
                    cost = self.cost_per_agent.get(agent, 1.0)
                    total_cost += cost
                    err_rate = self.error_rate.get(agent, 0.0)
                    if rng.random() < err_rate:
                        state.last_errors.append(f"sim-error:{agent}")
                        reward = -0.1
                    elif self._agent_success(agent, task.task_type, rng):
                        tag = component_for_agent(agent)
                        state.produced_components.add(tag)
                        state.last_active_agent = agent
                        reward = 0.2
                    else:
                        state.last_active_agent = agent
                        reward = -0.05
            elif kind in {"call_memory", "call_retriever", "call_tool_executor"}:
                total_cost += 0.5
                reward = 0.0
            elif kind == "call_verifier":
                state.produced_components.add("verifier_called")
                reward = 0.05

            rewards.append(reward)
            state.step_id += 1

        final_score = self._final_score(task, state.produced_components)
        success = final_score >= 0.6
        return SyntheticOutcome(
            task=task,
            success=success,
            final_score=final_score,
            actions=actions,
            states=states,
            rewards=rewards,
            cost=total_cost,
            steps=len(actions),
        )


def _snapshot(state: RuntimeState) -> RuntimeState:
    """Cheap shallow copy that detaches the per-step snapshot from the
    mutable runtime state used in the loop."""
    return RuntimeState(
        task_id=state.task_id,
        task_type=state.task_type,
        prompt=state.prompt,
        step_id=state.step_id,
        available_agents=list(state.available_agents),
        pending_inbox=dict(state.pending_inbox),
        last_active_agent=state.last_active_agent,
        last_output_summary=state.last_output_summary,
        last_verifier_score=state.last_verifier_score,
        last_verifier_passed=state.last_verifier_passed,
        last_verifier_failed_component=state.last_verifier_failed_component,
        last_errors=list(state.last_errors),
        produced_components=set(state.produced_components),
        totals_tokens=state.totals_tokens,
        totals_calls=state.totals_calls,
        totals_cost_rub=state.totals_cost_rub,
        extras=dict(state.extras),
    )


__all__ = ["SyntheticMAS", "SyntheticOutcome", "_snapshot"]

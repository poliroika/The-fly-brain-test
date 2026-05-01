"""Expert-trace dataset loader (PLAN.md §592-595).

A "trace" is the JSON dict returned by :func:`flybrain.runtime.runner.MAS.run`
— it mirrors the Rust ``Trace`` struct: ``task_id`` + ``task_type`` +
a list of ``TraceStep`` rows, each carrying the controller's
``graph_action`` plus per-step token / cost / verifier metadata.

For Phase-7 imitation learning we need to *replay* each trace step:
reconstruct the ``RuntimeState`` the controller saw at that tick
(out of the cumulative trace prefix) and pair it with the action the
expert actually emitted. The resulting list of
``ImitationExample``\\s is then consumed by
:mod:`flybrain.training.imitation`.

The replay logic is intentionally lossy — we only reconstruct the
fields the controller actually looks at (task descriptors,
``produced_components``, ``last_active_agent``, recent verifier
score). Full state replay is unnecessary because the controller's
``ControllerStateBuilder`` re-derives embeddings + the agent graph
from the runtime state every tick.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flybrain.controller.action_space import (
    KIND_ACTIVATE_AGENT,
    KIND_ADD_EDGE,
    KIND_CALL_MEMORY,
    KIND_CALL_RETRIEVER,
    KIND_CALL_TOOL_EXECUTOR,
    KIND_CALL_VERIFIER,
    KIND_NAMES,
    KIND_REMOVE_EDGE,
    KIND_SCALE_EDGE,
    KIND_TERMINATE,
)
from flybrain.runtime.state import RuntimeState
from flybrain.sim.optimal_routes import component_for_agent

# Map serialised graph_action ``kind`` string -> stable id (matches
# ``GraphAction::discriminant`` in Rust + ``KIND_*`` constants in
# Python).
KIND_NAME_TO_ID: dict[str, int] = {
    "activate_agent": KIND_ACTIVATE_AGENT,
    "add_edge": KIND_ADD_EDGE,
    "remove_edge": KIND_REMOVE_EDGE,
    "scale_edge": KIND_SCALE_EDGE,
    "call_memory": KIND_CALL_MEMORY,
    "call_retriever": KIND_CALL_RETRIEVER,
    "call_tool_executor": KIND_CALL_TOOL_EXECUTOR,
    "call_verifier": KIND_CALL_VERIFIER,
    "terminate": KIND_TERMINATE,
}


@dataclass(slots=True)
class ImitationExample:
    """One ``(state, expert_action)`` pair extracted from a trace."""

    runtime_state: RuntimeState
    agent_names: list[str]
    label_kind: int
    label_agent: int  # 0 when label_kind != activate_agent
    label_edge_from: int  # 0 when label_kind not in NEEDS_EDGE
    label_edge_to: int
    label_edge_weight: float
    aux_target: float
    """Verifier score the expert observed on this step (0.0 if no
    verifier fired). Used by the auxiliary verifier-prediction head."""


@dataclass(slots=True)
class TraceFile:
    path: Path
    task_id: str
    task_type: str
    final_answer: str | None
    verification_passed: bool
    verification_score: float
    totals: dict[str, Any]
    steps: list[dict[str, Any]]


def load_trace(path: str | Path) -> TraceFile:
    """Read a trace JSON written by ``MAS.run`` + ``TraceWriter``."""
    p = Path(path)
    data = json.loads(p.read_text())
    verification = data.get("verification") or {}
    return TraceFile(
        path=p,
        task_id=str(data.get("task_id", p.stem)),
        task_type=str(data.get("task_type", "synthetic_routing")),
        final_answer=data.get("final_answer"),
        verification_passed=bool(verification.get("passed", False)),
        verification_score=float(verification.get("score", 0.0)),
        totals=dict(data.get("totals") or {}),
        steps=list(data.get("steps") or []),
    )


def iter_traces(directory: str | Path) -> Iterator[TraceFile]:
    """Walk every ``*.trace.json`` under ``directory`` and yield it."""
    root = Path(directory)
    if not root.exists():
        return
    for p in sorted(root.rglob("*.trace.json")):
        try:
            yield load_trace(p)
        except (OSError, json.JSONDecodeError):
            continue


def _kind_id(action: dict[str, Any]) -> int:
    return KIND_NAME_TO_ID.get(str(action.get("kind", "")), KIND_TERMINATE)


def _agent_index(name: str | None, agent_names: list[str]) -> int:
    if name is None or not agent_names:
        return 0
    try:
        return agent_names.index(name)
    except ValueError:
        return 0


def _components_so_far(steps: list[dict[str, Any]], upto: int) -> set[str]:
    """Return the set of produced components after the *previous* step.

    Heuristic: every successful ``run_agent`` step contributes the
    component-tag attached to its ``active_agent``; ``call_verifier``
    contributes ``"verifier_called"`` when the verifier passed.
    """
    produced: set[str] = set()
    for prev in steps[:upto]:
        active = prev.get("active_agent")
        if isinstance(active, str) and active:
            produced.add(component_for_agent(active))
        score = prev.get("verifier_score")
        if score is not None and float(score) >= 0.5:
            produced.add("verifier_called")
    return produced


def _runtime_state_for_step(
    trace: TraceFile,
    step_idx: int,
    *,
    agent_names: list[str],
) -> RuntimeState:
    """Reconstruct the RuntimeState the controller saw before
    ``trace.steps[step_idx]`` was emitted."""
    steps = trace.steps
    produced = _components_so_far(steps, step_idx)
    last_active = steps[step_idx - 1].get("active_agent") if step_idx > 0 else None
    last_score = steps[step_idx - 1].get("verifier_score") if step_idx > 0 else None
    last_errors = list(steps[step_idx - 1].get("errors") or []) if step_idx > 0 else []
    return RuntimeState(
        task_id=trace.task_id,
        task_type=trace.task_type,
        prompt=str(trace.totals.get("prompt", "") or ""),
        step_id=int(steps[step_idx].get("step_id", step_idx)),
        available_agents=list(agent_names),
        pending_inbox={},
        last_active_agent=last_active if isinstance(last_active, str) else None,
        last_output_summary=(steps[step_idx - 1].get("output_summary") if step_idx > 0 else None),
        last_verifier_score=float(last_score) if last_score is not None else None,
        last_errors=last_errors,
        produced_components=produced,
    )


def trace_to_examples(
    trace: TraceFile,
    *,
    agent_names: list[str],
) -> list[ImitationExample]:
    """Convert one trace into a list of imitation examples — one per
    step where the controller emitted a graph action."""
    name_to_id = {n: i for i, n in enumerate(agent_names)}
    examples: list[ImitationExample] = []

    for step_idx, step in enumerate(trace.steps):
        action = step.get("graph_action")
        if not isinstance(action, dict):
            continue

        rs = _runtime_state_for_step(trace, step_idx, agent_names=agent_names)
        kind_id = _kind_id(action)
        agent_idx = 0
        edge_from = 0
        edge_to = 0
        edge_weight = 1.0

        if kind_id == KIND_ACTIVATE_AGENT:
            agent_name = action.get("agent")
            if agent_name not in name_to_id:
                # Drop the example: expert chose an agent the trainee
                # doesn't have a slot for.
                continue
            agent_idx = name_to_id[agent_name]
        elif kind_id in {
            KIND_ADD_EDGE,
            KIND_REMOVE_EDGE,
            KIND_SCALE_EDGE,
        }:
            f = action.get("from")
            t = action.get("to")
            if f not in name_to_id or t not in name_to_id or f == t:
                continue
            edge_from = name_to_id[f]
            edge_to = name_to_id[t]
            edge_weight = float(action.get("weight") or action.get("factor") or 1.0)

        verifier_score = step.get("verifier_score")
        aux = float(verifier_score) if verifier_score is not None else 0.0
        aux = max(0.0, min(1.0, aux))

        examples.append(
            ImitationExample(
                runtime_state=rs,
                agent_names=list(agent_names),
                label_kind=kind_id,
                label_agent=agent_idx,
                label_edge_from=edge_from,
                label_edge_to=edge_to,
                label_edge_weight=edge_weight,
                aux_target=aux,
            )
        )
    return examples


def collect_examples(
    traces: Iterable[TraceFile],
    *,
    agent_names: list[str],
    only_passed: bool = False,
) -> list[ImitationExample]:
    """Concatenate examples across multiple traces.

    Set ``only_passed=True`` to keep only traces that the verifier
    marked as passed (recommended for the imitation phase to avoid
    learning from confused expert demos)."""
    out: list[ImitationExample] = []
    for trace in traces:
        if only_passed and not trace.verification_passed:
            continue
        out.extend(trace_to_examples(trace, agent_names=agent_names))
    return out


__all__ = [
    "KIND_NAMES",
    "KIND_NAME_TO_ID",
    "ImitationExample",
    "TraceFile",
    "collect_examples",
    "iter_traces",
    "load_trace",
    "trace_to_examples",
]

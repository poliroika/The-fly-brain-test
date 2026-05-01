"""Per-trace and aggregate metric extraction for Phase-10 evaluation.

Reads the trace dicts produced by `flybrain.runtime.MAS.run` and
projects them down to the metric set called out in README §16
("nice that we measure quality, but we also have to measure cost").

The two main entry points are:

* `metrics_from_trace(trace)` — pure function, takes a trace dict
  (in-memory or freshly loaded from JSON), returns a `TaskMetrics`.
* `aggregate(metrics)` — averages a list of `TaskMetrics` into an
  `AggregateMetrics` row suitable for the comparison table.

Both functions are dependency-free (numpy/pytorch not required) so
they're cheap to call from CI.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskMetrics:
    """Per-task metric row.

    All fields are pre-computed from the trace dict so downstream
    aggregation never has to inspect the raw `steps` list.
    """

    task_id: str
    benchmark: str
    task_type: str
    success: bool
    """Verifier's `passed` flag."""

    verifier_score: float
    tokens_in: int
    tokens_out: int
    total_tokens: int
    llm_calls: int
    tool_calls: int
    failed_tool_calls: int
    latency_ms: int
    cost_rub: float
    num_steps: int
    graph_density: float
    """Mean per-step density. ``0.0`` if no steps recorded."""

    failed_component: str | None = None


@dataclass(slots=True)
class AggregateMetrics:
    """Per-method roll-up. ``num_tasks`` is what every other field is
    averaged over."""

    name: str
    """Free-form group label (baseline name, controller variant, …)."""

    benchmark: str
    """Benchmark id; ``"_overall"`` for cross-benchmark roll-ups."""

    num_tasks: int
    success_rate: float
    verifier_pass_rate: float
    avg_total_tokens: float
    avg_llm_calls: float
    avg_failed_tool_calls: float
    avg_latency_ms: float
    avg_cost_rub: float
    avg_steps: float
    avg_graph_density: float
    cost_per_solved_rub: float
    """Total ₽ spent / number of solved tasks. ``inf`` if nothing solved."""

    extras: dict[str, Any] = field(default_factory=dict)


def _safe_int(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(x: Any) -> float:
    try:
        return float(x or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _density(steps: list[dict[str, Any]]) -> float:
    """Mean over per-step ``current_graph_density`` if present.

    The runtime doesn't always emit per-step density; the field may
    sit on the action payload (``add_edge`` etc.) or in `metadata`.
    Falling back to 0 keeps the metric optional.
    """
    if not steps:
        return 0.0
    densities: list[float] = []
    for step in steps:
        v = step.get("graph_density")
        if v is None:
            v = step.get("current_graph_density")
        if v is None:
            action = step.get("graph_action") or {}
            v = action.get("graph_density") if isinstance(action, dict) else None
        if v is not None:
            densities.append(_safe_float(v))
    if not densities:
        return 0.0
    return statistics.fmean(densities)


def metrics_from_trace(
    trace: dict[str, Any],
    *,
    benchmark: str = "",
) -> TaskMetrics:
    """Project a single MAS trace dict into a `TaskMetrics`."""
    totals = trace.get("totals") or {}
    verification = trace.get("verification") or {}
    steps = trace.get("steps") or []
    tokens_in = _safe_int(totals.get("tokens_in"))
    tokens_out = _safe_int(totals.get("tokens_out"))
    total_tokens = tokens_in + tokens_out

    return TaskMetrics(
        task_id=str(trace.get("task_id", "")),
        benchmark=benchmark or str((trace.get("metadata") or {}).get("benchmark", "")),
        task_type=str(trace.get("task_type", "")),
        success=bool(verification.get("passed", False)),
        verifier_score=_safe_float(verification.get("score")),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=total_tokens,
        llm_calls=_safe_int(totals.get("llm_calls")),
        tool_calls=_safe_int(totals.get("tool_calls")),
        failed_tool_calls=_safe_int(totals.get("failed_tool_calls")),
        latency_ms=_safe_int(totals.get("latency_ms")),
        cost_rub=_safe_float(totals.get("cost_rub")),
        num_steps=len(steps),
        graph_density=_density(steps),
        failed_component=verification.get("failed_component"),
    )


def metrics_from_trace_path(path: str | Path, *, benchmark: str = "") -> TaskMetrics:
    """Convenience: load a `*.trace.json` file from disk and project it."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        trace = json.load(fh)
    return metrics_from_trace(trace, benchmark=benchmark)


def aggregate(
    metrics: list[TaskMetrics],
    *,
    name: str,
    benchmark: str = "_overall",
) -> AggregateMetrics:
    """Average a list of `TaskMetrics` into an `AggregateMetrics`."""
    if not metrics:
        return AggregateMetrics(
            name=name,
            benchmark=benchmark,
            num_tasks=0,
            success_rate=0.0,
            verifier_pass_rate=0.0,
            avg_total_tokens=0.0,
            avg_llm_calls=0.0,
            avg_failed_tool_calls=0.0,
            avg_latency_ms=0.0,
            avg_cost_rub=0.0,
            avg_steps=0.0,
            avg_graph_density=0.0,
            cost_per_solved_rub=float("inf"),
        )
    n = len(metrics)
    solved = sum(1 for m in metrics if m.success)
    total_cost = sum(m.cost_rub for m in metrics)
    return AggregateMetrics(
        name=name,
        benchmark=benchmark,
        num_tasks=n,
        success_rate=solved / n,
        verifier_pass_rate=statistics.fmean(m.verifier_score for m in metrics),
        avg_total_tokens=statistics.fmean(m.total_tokens for m in metrics),
        avg_llm_calls=statistics.fmean(m.llm_calls for m in metrics),
        avg_failed_tool_calls=statistics.fmean(m.failed_tool_calls for m in metrics),
        avg_latency_ms=statistics.fmean(m.latency_ms for m in metrics),
        avg_cost_rub=statistics.fmean(m.cost_rub for m in metrics),
        avg_steps=statistics.fmean(m.num_steps for m in metrics),
        avg_graph_density=statistics.fmean(m.graph_density for m in metrics),
        cost_per_solved_rub=(total_cost / solved) if solved else float("inf"),
    )


__all__ = [
    "AggregateMetrics",
    "TaskMetrics",
    "aggregate",
    "metrics_from_trace",
    "metrics_from_trace_path",
]

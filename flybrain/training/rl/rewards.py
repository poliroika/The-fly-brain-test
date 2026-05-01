"""Reward shaping (README §12.3).

The canonical formula is::

    reward = (
        success_score
        + 0.5 * verifier_score
        - alpha * total_tokens
        - beta * llm_calls
        - gamma * latency
        - delta * failed_tool_calls
        - eta * graph_density
    )

We ship one dataclass that holds the coefficients and one pure
function that ingests a trace dict (matching the JSON the runtime
writes via :class:`flybrain_core::trace::Trace`) and returns the
scalar reward. The helpers are deliberately framework-agnostic so
they can be called from REINFORCE / PPO / bandits *or* offline from
saved traces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class RewardConfig:
    """Coefficients for the README §12.3 reward.

    Defaults are the "budget mode" values from PLAN.md §666 — small
    but non-zero penalties on tokens and latency so the controller
    doesn't drift to an arbitrarily expensive policy."""

    success_weight: float = 1.0
    verifier_weight: float = 0.5
    alpha_tokens: float = 1e-4
    """Per-token penalty (subtracted from the reward)."""
    beta_llm_calls: float = 0.01
    """Per-LLM-call penalty (counted as ``len(steps)`` heuristically)."""
    gamma_latency_s: float = 0.001
    """Per-second wall-clock penalty."""
    delta_failed_tool_calls: float = 0.05
    eta_graph_density: float = 0.01

    def asdict(self) -> dict[str, float]:
        return {
            "success_weight": self.success_weight,
            "verifier_weight": self.verifier_weight,
            "alpha_tokens": self.alpha_tokens,
            "beta_llm_calls": self.beta_llm_calls,
            "gamma_latency_s": self.gamma_latency_s,
            "delta_failed_tool_calls": self.delta_failed_tool_calls,
            "eta_graph_density": self.eta_graph_density,
        }


def _success_score(trace: Mapping[str, Any]) -> float:
    """1.0 if the trace's verification passed, else 0.0."""
    verification = trace.get("verification") or {}
    return 1.0 if bool(verification.get("passed", False)) else 0.0


def _verifier_score(trace: Mapping[str, Any]) -> float:
    """Mean verifier score across all steps that called the verifier."""
    steps = trace.get("steps") or []
    scores = [float(s.get("verifier_score")) for s in steps if s.get("verifier_score") is not None]
    if not scores:
        # Fall back to the verification-block score if individual
        # steps didn't include one.
        v = trace.get("verification") or {}
        if "score" in v:
            return float(v["score"])
        return 0.0
    return sum(scores) / len(scores)


def _failed_tool_calls(trace: Mapping[str, Any]) -> int:
    steps = trace.get("steps") or []
    failed = 0
    for s in steps:
        outcome = s.get("tool_outcome") or s.get("tool_call_outcome")
        if outcome and str(outcome).lower() in {"error", "failed", "fail"}:
            failed += 1
        # Step-level explicit boolean.
        if s.get("tool_failed") is True:
            failed += 1
    return failed


def _graph_density(trace: Mapping[str, Any]) -> float:
    """Return the *average* per-step graph density (#edges / #possible).

    Works whether the trace stores the running graph snapshot in the
    final step, in metadata, or under ``totals``.
    """
    totals = trace.get("totals") or {}
    if "graph_density" in totals:
        return float(totals["graph_density"])
    steps = trace.get("steps") or []
    densities: list[float] = []
    for s in steps:
        graph = s.get("graph") or s.get("agent_graph")
        if not isinstance(graph, dict):
            continue
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or {}
        n = len(nodes)
        if n < 2:
            continue
        edge_count = sum(len(v or {}) for v in edges.values())
        densities.append(edge_count / max(1, n * (n - 1)))
    if not densities:
        return 0.0
    return sum(densities) / len(densities)


def compute_reward(
    trace: Mapping[str, Any],
    config: RewardConfig | None = None,
) -> float:
    """Apply the README §12.3 reward formula to one trace dict.

    Robust to partial / malformed traces — missing fields just
    contribute zero to their respective term."""
    cfg = config or RewardConfig()

    totals = trace.get("totals") or {}
    tokens = float(totals.get("tokens_in", 0) or 0) + float(totals.get("tokens_out", 0) or 0)
    llm_calls = len(trace.get("steps") or [])
    latency = float(totals.get("wall_seconds", 0.0) or 0.0)
    failed_tools = _failed_tool_calls(trace)
    density = _graph_density(trace)

    return (
        cfg.success_weight * _success_score(trace)
        + cfg.verifier_weight * _verifier_score(trace)
        - cfg.alpha_tokens * tokens
        - cfg.beta_llm_calls * llm_calls
        - cfg.gamma_latency_s * latency
        - cfg.delta_failed_tool_calls * failed_tools
        - cfg.eta_graph_density * density
    )


__all__ = ["RewardConfig", "compute_reward"]

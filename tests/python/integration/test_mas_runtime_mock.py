"""End-to-end MAS runtime test on a deterministic mock LLM.

Per Phase 2 exit criteria: run the manual controller against three
task types and verify the trace is well-formed, persisted, and
contains all required components. No Yandex calls, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flybrain.agents import load_minimal_15
from flybrain.controller import ManualController, RandomController
from flybrain.llm import MockLLMClient
from flybrain.llm.mock_client import MockRule
from flybrain.runtime import MAS, Agent, AgentSpec, MASConfig, Task
from flybrain.runtime.tools import default_tool_registry

native = pytest.importorskip("flybrain.flybrain_native")


def _agent_factory(spec: AgentSpec, llm: MockLLMClient) -> Agent:
    """Build an Agent that talks to the same mock LLM (with role-aware rules
    pre-installed) and shares one tool registry across the run."""
    tools = default_tool_registry()
    default_tool: str | None = None
    if spec.name in {"Retriever", "Researcher", "SearchAgent"}:
        default_tool = "web_search"
    return Agent(spec=spec, llm=llm, tools=tools, default_tool=default_tool)


def _mock_with_role_rules() -> MockLLMClient:
    """Pre-seed the mock with rules that produce the marker tokens
    `Agent._infer_produced_components` looks for."""
    client = MockLLMClient()
    client.rules.extend(
        [
            MockRule(pattern=r"role: planner", response="Plan: 1) decompose 2) act"),
            MockRule(
                pattern=r"role: coder",
                response="```python\ndef solve(x): return x + 1\n```",
            ),
            MockRule(
                pattern=r"role: tester",
                response="tests_run: All tests passed (3/3).",
            ),
            MockRule(
                pattern=r"role: math",
                response="Step-by-step. final_answer: 42",
            ),
            MockRule(
                pattern=r"role: retriever",
                response="Top results bullet 1; bullet 2",
            ),
            MockRule(
                pattern=r"role: researcher",
                response="3 bullets. final_answer: synthesised gist.",
            ),
            MockRule(
                pattern=r"role: tool_executor",
                response="Tool dispatched. final_answer: 7",
            ),
            MockRule(
                pattern=r"role: finalizer",
                response="final_answer: combined output for the task.",
            ),
        ]
    )
    return client


@pytest.mark.asyncio
async def test_mas_runtime_runs_coding_task(tmp_path: Path) -> None:
    llm = _mock_with_role_rules()
    mas = MAS.from_specs(
        load_minimal_15(),
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=20, trace_dir=tmp_path),
    )
    trace = await mas.run(
        Task(task_id="t-coding", task_type="coding", prompt="add 1 to x"),
        ManualController(),
    )

    assert trace["task_id"] == "t-coding"
    assert trace["task_type"] == "coding"
    # Manual plan executes Planner → Coder → TestRunner → call_verifier → Finalizer → terminate
    activated = [s["active_agent"] for s in trace["steps"] if s["active_agent"]]
    assert "Planner" in activated
    assert "Coder" in activated
    assert "TestRunner" in activated
    assert "Finalizer" in activated

    # JSONL sink exists and has one line per step.
    sink = tmp_path / "t-coding.steps.jsonl"
    assert sink.exists()
    body = sink.read_text().strip().splitlines()
    assert len(body) == len(trace["steps"])
    for line in body:
        json.loads(line)  # each line must be valid JSON

    # Totals reflect every LLM call we made.
    assert trace["totals"]["llm_calls"] >= 4
    assert trace["totals"]["tokens_in"] > 0
    assert trace["totals"]["tokens_out"] > 0

    await llm.aclose()


@pytest.mark.asyncio
async def test_mas_runtime_runs_math_task(tmp_path: Path) -> None:
    llm = _mock_with_role_rules()
    mas = MAS.from_specs(
        load_minimal_15(),
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=20, trace_dir=tmp_path),
    )
    trace = await mas.run(
        Task(task_id="t-math", task_type="math", prompt="2+2"),
        ManualController(),
    )
    activated = [s["active_agent"] for s in trace["steps"] if s["active_agent"]]
    assert "MathSolver" in activated
    assert "Finalizer" in activated
    # The MathSolver rule emits `final_answer: 42` so the verifier
    # should be satisfied (verification.passed == True).
    assert trace["verification"]["passed"] is True
    await llm.aclose()


@pytest.mark.asyncio
async def test_mas_runtime_runs_research_task_with_tool_call(tmp_path: Path) -> None:
    llm = _mock_with_role_rules()
    mas = MAS.from_specs(
        load_minimal_15(),
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=20, trace_dir=tmp_path),
    )
    trace = await mas.run(
        Task(task_id="t-research", task_type="research", prompt="fly brain compression"),
        ManualController(),
    )
    activated = [s["active_agent"] for s in trace["steps"] if s["active_agent"]]
    # `Researcher` is in EXTENDED_25 only; with MINIMAL_15 the manual
    # plan steps over the missing agent, so we expect the Retriever path
    # to still fire and the run to terminate cleanly.
    assert "Retriever" in activated
    assert "Finalizer" in activated
    # Retriever calls `web_search`, which is a no-op fixture by default
    # — but the tool_call must show up in the trace anyway.
    tool_names = [c["name"] for s in trace["steps"] for c in s["tool_calls"]]
    assert "web_search" in tool_names
    await llm.aclose()


@pytest.mark.asyncio
async def test_random_controller_terminates_within_budget(tmp_path: Path) -> None:
    llm = _mock_with_role_rules()
    mas = MAS.from_specs(
        load_minimal_15(),
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=24, trace_dir=tmp_path),
    )
    trace = await mas.run(
        Task(task_id="t-random", task_type="coding", prompt="fizzbuzz"),
        RandomController(seed=7, p_terminate=0.15),
    )
    # Budget cap honoured: at most 24 recorded steps.
    assert len(trace["steps"]) <= 24
    # Trace JSONL persisted.
    sink = tmp_path / "t-random.steps.jsonl"
    assert sink.exists()
    await llm.aclose()

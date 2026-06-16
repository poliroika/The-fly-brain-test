#!/usr/bin/env python3
"""Phase-7 trace collection — runs an *expert* MAS against a YandexGPT
backend and persists per-task traces.

Usage::

    # Live collection on YandexGPT Lite, 100 tasks, 200 ₽ hard cap.
    YANDEX_API_KEY=... folder_id=... \\
    python scripts/collect_expert_traces.py \\
        --output data/traces/expert/v1 \\
        --backend yandex --tier lite \\
        --tasks 100 --budget-rub 200

    # Dry-run on the deterministic mock client (no API calls, no cost).
    python scripts/collect_expert_traces.py \\
        --output runs/expert_traces_dry --backend mock --tasks 4

The expert is the existing :class:`flybrain.controller.ManualController`
running on the full ``MINIMAL_15`` agent set. Optionally pass
``--initial-graph fully-connected`` to seed the AgentGraph with all
agent → agent edges so messages broadcast everywhere (bigger but
more diverse trace).

Per-task budget:
* `--budget-rub` is the *hard cap* across the whole run.
* The collector stops cleanly after the next ``BudgetExceededError``
  and writes a summary JSON next to the traces.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flybrain.agents import load_minimal_15
from flybrain.controller import ManualController
from flybrain.llm import (
    BudgetExceededError,
    BudgetTracker,
    MockLLMClient,
    SQLiteCache,
    YandexClient,
)
from flybrain.llm.mock_client import MockRule
from flybrain.llm.yandex_client import YandexConfig
from flybrain.runtime import MAS, Agent, AgentSpec, MASConfig, Task
from flybrain.runtime.tools import default_tool_registry
from flybrain.sim.task_generator import SyntheticTask, TaskGenerator


@dataclass(slots=True)
class CollectionSummary:
    backend: str
    requested_tasks: int
    completed_tasks: int
    skipped_tasks: int
    successful_traces: int
    total_cost_rub: float
    total_tokens_in: int
    total_tokens_out: int
    total_steps: int
    wall_seconds: float
    halted_on_budget: bool


# -- agent factory --------------------------------------------------------------


def _agent_factory(spec: AgentSpec, llm) -> Agent:  # type: ignore[no-untyped-def]
    tools = default_tool_registry()
    default_tool: str | None = None
    if spec.name in {"Retriever", "Researcher", "SearchAgent"}:
        default_tool = "web_search"
    return Agent(spec=spec, llm=llm, tools=tools, default_tool=default_tool)


def _mock_role_rules() -> MockLLMClient:
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
            MockRule(pattern=r"role: math", response="Step-by-step. final_answer: 42"),
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


# -- task generation ------------------------------------------------------------


def _make_tasks(n: int, seed: int) -> list[Task]:
    gen = TaskGenerator(seed=seed)
    syn: list[SyntheticTask] = []
    # Round-robin balanced over task types.
    types = ("coding", "math", "research", "tool_use")
    for i in range(n):
        syn.append(gen.sample(task_type=types[i % len(types)]))
    return [Task(task_id=t.task_id, task_type=t.task_type, prompt=t.prompt) for t in syn]


def _initial_graph_fully_connected(agent_names: list[str]) -> dict[str, Any]:
    edges: dict[str, dict[str, float]] = {}
    for src in agent_names:
        edges[src] = {dst: 1.0 for dst in agent_names if dst != src}
    return {"nodes": list(agent_names), "edges": edges}


# -- main loop ------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> CollectionSummary:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "mock":
        llm: Any = _mock_role_rules()
        backend_label = "mock"
    elif args.backend == "yandex":
        cache_path = out_dir / "yandex_cache.sqlite"
        cache = SQLiteCache(path=cache_path)
        budget = BudgetTracker(hard_cap_rub=float(args.budget_rub))
        config = YandexConfig.from_env()
        llm = YandexClient(config=config, cache=cache, budget=budget)
        backend_label = f"yandex/{config.lite_model}"
    else:  # pragma: no cover - argparse choices guard
        raise ValueError(f"unknown backend {args.backend}")

    specs = load_minimal_15()
    agent_names = [s.name for s in specs]
    initial_graph = (
        _initial_graph_fully_connected(agent_names)
        if args.initial_graph == "fully-connected"
        else None
    )

    mas = MAS.from_specs(
        specs,
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=args.max_steps, trace_dir=out_dir),
    )

    summary = CollectionSummary(
        backend=backend_label,
        requested_tasks=args.tasks,
        completed_tasks=0,
        skipped_tasks=0,
        successful_traces=0,
        total_cost_rub=0.0,
        total_tokens_in=0,
        total_tokens_out=0,
        total_steps=0,
        wall_seconds=0.0,
        halted_on_budget=False,
    )

    tasks = _make_tasks(args.tasks, seed=args.seed)
    t0 = time.perf_counter()
    for idx, task in enumerate(tasks):
        try:
            trace = await mas.run(
                task,
                ManualController(),
                initial_graph=initial_graph,
            )
        except BudgetExceededError as e:
            summary.halted_on_budget = True
            summary.skipped_tasks = args.tasks - idx
            print(f"[budget] hard cap reached: {e}; halting after {idx} tasks")
            break

        # Persist trace JSON next to the steps JSONL written by TraceWriter.
        trace_path = out_dir / f"{task.task_id}.trace.json"
        trace_path.write_text(json.dumps(trace, indent=2))
        summary.completed_tasks += 1
        if trace.get("verification", {}).get("passed", False):
            summary.successful_traces += 1
        totals = trace.get("totals") or {}
        summary.total_cost_rub += float(totals.get("cost_rub", 0.0) or 0.0)
        summary.total_tokens_in += int(totals.get("tokens_in", 0) or 0)
        summary.total_tokens_out += int(totals.get("tokens_out", 0) or 0)
        summary.total_steps += len(trace.get("steps") or [])

        if (idx + 1) % max(1, args.log_every) == 0:
            print(
                f"[{idx + 1}/{args.tasks}] cost={summary.total_cost_rub:.2f}₽ "
                f"steps={summary.total_steps} pass={summary.successful_traces}"
            )

    summary.wall_seconds = time.perf_counter() - t0

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    print(f"[done] saved {summary.completed_tasks} traces to {out_dir}")
    print(f"[done] summary: {summary_path}")

    if hasattr(llm, "aclose"):
        try:
            await llm.aclose()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write trace JSONs + summary.json into.",
    )
    parser.add_argument(
        "--backend",
        choices=("yandex", "mock"),
        default="mock",
        help="LLM backend; default mock = deterministic, no API calls.",
    )
    parser.add_argument(
        "--tier",
        choices=("lite", "pro"),
        default="lite",
        help="Yandex model tier (default: lite for cost).",
    )
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--budget-rub",
        type=float,
        default=200.0,
        help="Hard cost cap. The collector halts cleanly once this is hit.",
    )
    parser.add_argument(
        "--initial-graph",
        choices=("none", "fully-connected"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

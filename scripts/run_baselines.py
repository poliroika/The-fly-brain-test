#!/usr/bin/env python3
"""Phase-9 baseline harness — runs all 9 baselines on the same task
suite and emits a comparison table (PLAN.md §603-605, §17).

Usage::

    # Smoke-run on synthetic tasks with the deterministic mock LLM (no API).
    python scripts/run_baselines.py \\
        --suite full_min --backend mock --tasks 12 \\
        --output runs/baselines_smoke

    # Live run on YandexGPT Lite (caches + budget-tracker).
    YANDEX_API_KEY=... folder_id=... \\
    python scripts/run_baselines.py \\
        --suite full_min --backend yandex --tasks 40 \\
        --budget-rub 300 --output data/baselines/v1

The output directory ends up with::

    <run-name>/
        <baseline_name>/
            <task_id>.trace.json
            summary.json
        comparison.json   # one row per baseline + per-task aggregates
        comparison.md     # markdown table (drops into PLAN.md §17)

Each baseline runs on the *same* task seed so the comparison is
apples-to-apples. The command exits 0 even if some baselines fail
mid-way; per-baseline errors land in ``comparison.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flybrain.agents import load_minimal_15
from flybrain.baselines import BUILTIN_SUITES, BaselineSpec, list_baselines
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

# -- agent factory --------------------------------------------------------------


def _agent_factory(spec: AgentSpec, llm: Any) -> Agent:
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


# -- result aggregator ----------------------------------------------------------


@dataclass(slots=True)
class BaselineResult:
    name: str
    description: str
    completed_tasks: int
    successful_tasks: int
    avg_steps: float
    avg_tokens: float
    avg_cost_rub: float
    success_rate: float
    wall_seconds: float
    error: str | None = None
    tags: list[str] = field(default_factory=list)


def _format_markdown_table(results: list[BaselineResult]) -> str:
    header = (
        "| Baseline | Tasks | Success | Steps/task | Tokens/task | Cost/task |\n"
        "|----------|-------|---------|------------|-------------|-----------|\n"
    )
    rows = []
    for r in results:
        if r.error:
            rows.append(f"| {r.name} | — | error: {r.error[:40]} | — | — | — |")
            continue
        rows.append(
            f"| {r.name} | {r.completed_tasks} | "
            f"{r.success_rate:.2f} | "
            f"{r.avg_steps:.1f} | "
            f"{r.avg_tokens:.0f} | "
            f"{r.avg_cost_rub:.3f} ₽ |"
        )
    return header + "\n".join(rows) + "\n"


# -- main loop ------------------------------------------------------------------


async def _run_one_baseline(
    spec: BaselineSpec,
    *,
    tasks: list[Task],
    llm: Any,
    out_dir: Path,
    max_steps: int,
) -> BaselineResult:
    specs = load_minimal_15()
    agent_names = [s.name for s in specs]

    try:
        # Factories may internally call ``asyncio.run`` (e.g. to drive
        # the AgentEmbedder precompute). We're already inside a
        # running event loop here, so run the factory on a worker
        # thread to give it its own event loop context.
        controller, initial_graph = await asyncio.to_thread(spec.factory, agent_names)
    except Exception as e:  # pragma: no cover - factory bugs surface here
        traceback.print_exc()
        return BaselineResult(
            name=spec.name,
            description=spec.description,
            completed_tasks=0,
            successful_tasks=0,
            avg_steps=0.0,
            avg_tokens=0.0,
            avg_cost_rub=0.0,
            success_rate=0.0,
            wall_seconds=0.0,
            error=str(e),
            tags=list(spec.tags),
        )

    bdir = out_dir / spec.name
    bdir.mkdir(parents=True, exist_ok=True)
    mas = MAS.from_specs(
        specs,
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=max_steps, trace_dir=bdir),
    )

    completed = 0
    successful = 0
    total_steps = 0
    total_tokens = 0
    total_cost = 0.0
    t0 = time.perf_counter()
    halted = False

    for task in tasks:
        try:
            trace = await mas.run(task, controller, initial_graph=initial_graph)
        except BudgetExceededError as e:
            halted = True
            print(f"[{spec.name}] budget cap hit: {e}; stopping")
            break

        completed += 1
        if trace.get("verification", {}).get("passed", False):
            successful += 1
        totals = trace.get("totals") or {}
        total_steps += len(trace.get("steps") or [])
        total_tokens += int(totals.get("tokens_in", 0) or 0) + int(totals.get("tokens_out", 0) or 0)
        total_cost += float(totals.get("cost_rub", 0.0) or 0.0)
        (bdir / f"{task.task_id}.trace.json").write_text(json.dumps(trace, indent=2))

    elapsed = time.perf_counter() - t0
    n = max(1, completed)
    res = BaselineResult(
        name=spec.name,
        description=spec.description,
        completed_tasks=completed,
        successful_tasks=successful,
        avg_steps=total_steps / n,
        avg_tokens=total_tokens / n,
        avg_cost_rub=total_cost / n,
        success_rate=successful / n,
        wall_seconds=elapsed,
        error="budget_exhausted" if halted else None,
        tags=list(spec.tags),
    )
    (bdir / "summary.json").write_text(json.dumps(asdict(res), indent=2))
    print(
        f"[{spec.name}] completed={completed} success={successful}/{n} "
        f"steps={total_steps} cost={total_cost:.2f}₽ wall={elapsed:.1f}s"
    )
    return res


def _make_tasks(n: int, seed: int) -> list[Task]:
    gen = TaskGenerator(seed=seed)
    syn: list[SyntheticTask] = []
    types = ("coding", "math", "research", "tool_use")
    for i in range(n):
        syn.append(gen.sample(task_type=types[i % len(types)]))
    return [Task(task_id=t.task_id, task_type=t.task_type, prompt=t.prompt) for t in syn]


async def _run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "mock":
        llm: Any = _mock_role_rules()
    else:
        cache = SQLiteCache(path=out_dir / "yandex_cache.sqlite")
        budget = BudgetTracker(hard_cap_rub=float(args.budget_rub))
        llm = YandexClient(config=YandexConfig.from_env(), cache=cache, budget=budget)

    tasks = _make_tasks(args.tasks, seed=args.seed)
    suite = list_baselines(args.suite)
    print(f"[suite={args.suite}] {len(suite)} baselines x {len(tasks)} tasks")

    results: list[BaselineResult] = []
    for spec in suite:
        if args.only and spec.name not in args.only:
            continue
        results.append(
            await _run_one_baseline(
                spec,
                tasks=tasks,
                llm=llm,
                out_dir=out_dir,
                max_steps=args.max_steps,
            )
        )

    summary_path = out_dir / "comparison.json"
    summary_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    md_path = out_dir / "comparison.md"
    md_path.write_text(_format_markdown_table(results))
    print(f"[done] comparison: {summary_path}")
    print(f"[done] markdown : {md_path}")
    print()
    print(_format_markdown_table(results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=sorted(BUILTIN_SUITES),
        default="full_min",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to a subset of baselines by name.",
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "yandex"),
        default="mock",
    )
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--budget-rub", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/baselines_default"),
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

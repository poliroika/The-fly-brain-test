"""Programmatic entry point shared by `scripts/run_benchmarks.py` and
`flybrain-py bench` (Phase-10 CLI dispatch).

Keeping the actual orchestration in the package means both call
sites stay one-liners and the integration is unit-testable without
spawning a subprocess.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flybrain.agents import load_minimal_15
from flybrain.baselines import BaselineSpec, list_baselines
from flybrain.benchmarks.base import BenchmarkTask
from flybrain.benchmarks.loaders import (
    DEFAULT_BENCHMARK_SUITE,
    load_benchmark,
)
from flybrain.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkRunnerConfig,
)
from flybrain.eval import (
    AggregateMetrics,
    ReportInputs,
    aggregate,
    csv_table,
    markdown_table,
    metrics_from_trace_path,
    write_report,
)
from flybrain.eval.reports import select_cherry_picks
from flybrain.llm import (
    BudgetTracker,
    MockLLMClient,
    SQLiteCache,
    YandexClient,
)
from flybrain.llm.mock_client import MockRule
from flybrain.llm.yandex_client import YandexConfig
from flybrain.runtime import MAS, Agent, AgentSpec, MASConfig
from flybrain.runtime.tools import default_tool_registry


def _strict_json(value: Any) -> Any:
    """Recursively convert non-JSON-strict floats (`inf`, `nan`) to
    `None` so the resulting `json.dumps` output is valid strict JSON
    (parseable by browsers / `JSON.parse`)."""
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _strict_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strict_json(v) for v in value]
    return value


def _agent_factory(spec: AgentSpec, llm: Any) -> Agent:
    tools = default_tool_registry()
    default_tool: str | None = None
    if spec.name in {"Retriever", "Researcher", "SearchAgent"}:
        default_tool = "web_search"
    return Agent(spec=spec, llm=llm, tools=tools, default_tool=default_tool)


def _mock_client() -> MockLLMClient:
    """Same role-based mock as `scripts/run_baselines.py` so that the
    benchmark smoke run is comparable to the baselines'.
    """
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


def _load_suite(
    benchmarks: list[str],
    tasks_per_benchmark: int,
    seed: int,
) -> list[tuple[str, list[BenchmarkTask]]]:
    suite: list[tuple[str, list[BenchmarkTask]]] = []
    for name in benchmarks:
        try:
            tasks = load_benchmark(name, num_tasks=tasks_per_benchmark, seed=seed)
        except FileNotFoundError as e:
            print(f"[warn] skipping {name}: {e}", file=sys.stderr)
            continue
        suite.append((name, tasks))
    return suite


async def _run_one_baseline(
    spec: BaselineSpec,
    *,
    suite: list[tuple[str, list[BenchmarkTask]]],
    llm: Any,
    out_dir: Path,
    cfg: BenchmarkRunnerConfig,
    max_steps: int,
) -> dict[str, Any]:
    specs = load_minimal_15()
    agent_names = [s.name for s in specs]
    controller, initial_graph = await asyncio.to_thread(spec.factory, agent_names)

    bdir = out_dir / spec.name
    bdir.mkdir(parents=True, exist_ok=True)
    mas = MAS.from_specs(
        specs,
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=max_steps, trace_dir=bdir),
    )
    runner = BenchmarkRunner(mas, controller, initial_graph=initial_graph, config=cfg)

    summary: dict[str, Any] = {
        "baseline": spec.name,
        "description": spec.description,
        "benchmarks": {},
    }
    for benchmark, tasks in suite:
        report = await runner.run_benchmark(benchmark, tasks, output_dir=bdir)
        summary["benchmarks"][benchmark] = asdict(report)
        print(
            f"[{spec.name} / {benchmark}] "
            f"completed={report.completed} success={report.successful}/"
            f"{max(1, report.completed)} errored={report.errored} "
            f"wall={report.wall_seconds:.1f}s",
            flush=True,
        )
    return summary


def _collect_metrics_for(out_dir: Path, baseline: str, benchmark: str):
    bdir = out_dir / baseline / benchmark
    if not bdir.exists():
        return []
    return [
        metrics_from_trace_path(p, benchmark=benchmark) for p in sorted(bdir.glob("*.trace.json"))
    ]


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "mock":
        llm: Any = _mock_client()
    else:
        cache = SQLiteCache(path=out_dir / "yandex_cache.sqlite")
        budget = BudgetTracker(hard_cap_rub=float(args.budget_rub))
        llm = YandexClient(config=YandexConfig.from_env(), cache=cache, budget=budget)

    benchmarks = args.benchmarks or list(DEFAULT_BENCHMARK_SUITE)
    suite_tasks = _load_suite(benchmarks, args.tasks_per_benchmark, args.seed)

    cfg = BenchmarkRunnerConfig(
        parallelism=args.parallelism,
        max_retries=args.max_retries,
        timeout_s=args.timeout_s,
        persist_traces=True,
    )
    baselines = list_baselines(args.suite)
    if args.only:
        baselines = [b for b in baselines if b.name in args.only]
    print(
        f"[suite={args.suite}] {len(baselines)} baselines x {len(suite_tasks)} benchmarks",
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for spec in baselines:
        summaries.append(
            await _run_one_baseline(
                spec,
                suite=suite_tasks,
                llm=llm,
                out_dir=out_dir,
                cfg=cfg,
                max_steps=args.max_steps,
            )
        )
    (out_dir / "summaries.json").write_text(
        json.dumps(_strict_json(summaries), indent=2, allow_nan=False)
    )

    overall: list[AggregateMetrics] = []
    per_benchmark: dict[str, list[AggregateMetrics]] = {}
    for spec in baselines:
        all_metrics = []
        for benchmark, _ in suite_tasks:
            mets = _collect_metrics_for(out_dir, spec.name, benchmark)
            all_metrics.extend(mets)
            per_benchmark.setdefault(benchmark, []).append(
                aggregate(mets, name=spec.name, benchmark=benchmark)
            )
        overall.append(aggregate(all_metrics, name=spec.name, benchmark="_overall"))

    (out_dir / "comparison_overall.json").write_text(
        json.dumps([_strict_json(asdict(r)) for r in overall], indent=2, allow_nan=False)
    )
    (out_dir / "comparison_overall.md").write_text(markdown_table(overall))
    (out_dir / "comparison_overall.csv").write_text(csv_table(overall))
    for benchmark, rows in per_benchmark.items():
        (out_dir / f"comparison_{benchmark}.json").write_text(
            json.dumps([_strict_json(asdict(r)) for r in rows], indent=2, allow_nan=False)
        )
        (out_dir / f"comparison_{benchmark}.md").write_text(markdown_table(rows))
        (out_dir / f"comparison_{benchmark}.csv").write_text(csv_table(rows))

    cherry = select_cherry_picks(out_dir, max_picks=3)
    write_report(
        ReportInputs(
            suite_name=args.suite,
            overall=overall,
            per_benchmark=per_benchmark,
            trace_paths=[c.path for c in cherry],
            cherry_picks=cherry,
        ),
        out_dir / "report.md",
    )
    print(f"[done] {out_dir / 'comparison_overall.md'}", flush=True)
    print(f"[done] {out_dir / 'report.md'}", flush=True)
    return 0


__all__ = ["run"]

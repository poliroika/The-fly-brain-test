"""Async benchmark runner — Phase-10 entry point (PLAN.md §609).

The runner:

* Wraps `MAS.run` with a per-task budget guard (`BudgetExceededError`
  is caught and converts to a graceful ``error`` row, the suite
  continues with the rest of the tasks).
* Limits concurrency through an `asyncio.Semaphore` so the runner
  never floods the LLM proxy when a benchmark is large.
* Retries each task up to ``max_retries`` times on transient
  exceptions (anything except `BudgetExceededError`, which is
  terminal).
* Persists every successful run as ``<output_dir>/<task_id>.trace.json``
  so the rest of the eval pipeline (`flybrain.eval`) can rebuild the
  metric tables from disk.

The result is a `BenchmarkRunReport` summarising per-task outcomes —
metric extraction itself lives in `flybrain.eval.metrics` so this
module is single-responsibility (orchestration only).
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flybrain.benchmarks.base import BenchmarkTask
from flybrain.controller.base import Controller
from flybrain.llm.budget import BudgetExceededError
from flybrain.runtime import MAS, Task


@dataclass(slots=True)
class TaskOutcome:
    """One row of the benchmark run report."""

    task_id: str
    benchmark: str
    task_type: str
    completed: bool
    """True iff the run reached a verifier verdict (passed or failed)."""

    passed: bool
    """The verifier's `passed` flag. False if `completed=False`."""

    wall_seconds: float
    error: str | None = None
    trace_path: str | None = None


@dataclass(slots=True)
class BenchmarkRunReport:
    """Per-benchmark roll-up plus the underlying per-task outcomes."""

    benchmark: str
    completed: int
    successful: int
    failed: int
    errored: int
    wall_seconds: float
    outcomes: list[TaskOutcome] = field(default_factory=list)
    halted_on_budget: bool = False

    @property
    def success_rate(self) -> float:
        n = max(1, self.completed)
        return self.successful / n


def _slugify(task_id: str) -> str:
    return task_id.replace("/", "__").replace(" ", "_")


@dataclass(slots=True)
class BenchmarkRunnerConfig:
    """Tuneable knobs for `BenchmarkRunner.run_benchmark`."""

    parallelism: int = 1
    max_retries: int = 1
    """Total attempts per task. ``1`` = no retry."""

    timeout_s: float | None = None
    """Hard wall-clock cap per task; ``None`` = no timeout."""

    persist_traces: bool = True


class BenchmarkRunner:
    """Bind a benchmark suite to a single MAS + Controller setup."""

    def __init__(
        self,
        mas: MAS,
        controller: Controller,
        *,
        initial_graph: dict[str, Any] | None = None,
        config: BenchmarkRunnerConfig | None = None,
    ) -> None:
        self.mas = mas
        self.controller = controller
        self.initial_graph = initial_graph
        self.config = config or BenchmarkRunnerConfig()

    async def _run_single(
        self,
        task: BenchmarkTask,
        out_dir: Path,
        sem: asyncio.Semaphore,
    ) -> TaskOutcome:
        async with sem:
            t0 = time.perf_counter()
            last_error: str | None = None
            runtime_task: Task = task.to_runtime_task()
            cfg = self.config
            for attempt in range(max(1, cfg.max_retries)):
                try:
                    coro = self.mas.run(
                        runtime_task,
                        self.controller,
                        initial_graph=self.initial_graph,
                    )
                    if cfg.timeout_s is not None:
                        trace = await asyncio.wait_for(coro, timeout=cfg.timeout_s)
                    else:
                        trace = await coro
                except BudgetExceededError as e:
                    return TaskOutcome(
                        task_id=task.task_id,
                        benchmark=task.benchmark,
                        task_type=task.task_type,
                        completed=False,
                        passed=False,
                        wall_seconds=time.perf_counter() - t0,
                        error=f"budget_exceeded: {e}",
                    )
                except (TimeoutError, asyncio.TimeoutError) as e:  # noqa: UP041
                    last_error = f"timeout_after_{cfg.timeout_s}s"
                    if attempt + 1 >= cfg.max_retries:
                        return TaskOutcome(
                            task_id=task.task_id,
                            benchmark=task.benchmark,
                            task_type=task.task_type,
                            completed=False,
                            passed=False,
                            wall_seconds=time.perf_counter() - t0,
                            error=last_error or str(e),
                        )
                    continue
                except Exception as e:  # pragma: no cover - last-ditch retry
                    last_error = f"{e.__class__.__name__}: {e}"
                    traceback.print_exc()
                    if attempt + 1 >= cfg.max_retries:
                        return TaskOutcome(
                            task_id=task.task_id,
                            benchmark=task.benchmark,
                            task_type=task.task_type,
                            completed=False,
                            passed=False,
                            wall_seconds=time.perf_counter() - t0,
                            error=last_error,
                        )
                    continue
                else:
                    elapsed = time.perf_counter() - t0
                    trace_path: str | None = None
                    if cfg.persist_traces:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        path = out_dir / f"{_slugify(task.task_id)}.trace.json"
                        path.write_text(json.dumps(trace, indent=2))
                        trace_path = str(path)
                    verification = trace.get("verification") or {}
                    return TaskOutcome(
                        task_id=task.task_id,
                        benchmark=task.benchmark,
                        task_type=task.task_type,
                        completed=True,
                        passed=bool(verification.get("passed", False)),
                        wall_seconds=elapsed,
                        trace_path=trace_path,
                    )
            # Unreachable, but the type checker likes it explicit.
            return TaskOutcome(
                task_id=task.task_id,
                benchmark=task.benchmark,
                task_type=task.task_type,
                completed=False,
                passed=False,
                wall_seconds=time.perf_counter() - t0,
                error=last_error or "unknown_error",
            )

    async def run_benchmark(
        self,
        benchmark: str,
        tasks: list[BenchmarkTask],
        *,
        output_dir: Path,
    ) -> BenchmarkRunReport:
        """Run every task in ``tasks`` and return a per-benchmark report."""
        sem = asyncio.Semaphore(max(1, self.config.parallelism))
        out_dir = Path(output_dir) / benchmark
        out_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()
        coros = [self._run_single(t, out_dir, sem) for t in tasks]
        outcomes: list[TaskOutcome] = []
        halted = False
        for fut in asyncio.as_completed(coros):
            outcome = await fut
            outcomes.append(outcome)
            if outcome.error and outcome.error.startswith("budget_exceeded"):
                halted = True

        outcomes.sort(key=lambda o: o.task_id)
        elapsed = time.perf_counter() - t_start

        completed = sum(1 for o in outcomes if o.completed)
        successful = sum(1 for o in outcomes if o.completed and o.passed)
        failed = sum(1 for o in outcomes if o.completed and not o.passed)
        errored = sum(1 for o in outcomes if not o.completed)

        report = BenchmarkRunReport(
            benchmark=benchmark,
            completed=completed,
            successful=successful,
            failed=failed,
            errored=errored,
            wall_seconds=elapsed,
            outcomes=outcomes,
            halted_on_budget=halted,
        )
        (Path(output_dir) / benchmark / "summary.json").write_text(
            json.dumps(asdict(report), indent=2)
        )
        return report

    async def run_suite(
        self,
        suite: list[tuple[str, list[BenchmarkTask]]],
        *,
        output_dir: Path,
    ) -> dict[str, BenchmarkRunReport]:
        """Run multiple benchmarks back-to-back; returns one report per name."""
        results: dict[str, BenchmarkRunReport] = {}
        for name, tasks in suite:
            results[name] = await self.run_benchmark(name, tasks, output_dir=output_dir)
        return results


def run_benchmark_sync(
    runner: BenchmarkRunner,
    benchmark: str,
    tasks: list[BenchmarkTask],
    *,
    output_dir: Path,
    loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
) -> BenchmarkRunReport:
    """Synchronous wrapper around `BenchmarkRunner.run_benchmark` for
    callers that don't already live in an event loop."""
    factory = loop_factory or asyncio.new_event_loop
    loop = factory()
    try:
        return loop.run_until_complete(
            runner.run_benchmark(benchmark, tasks, output_dir=output_dir)
        )
    finally:
        loop.close()


__all__ = [
    "BenchmarkRunReport",
    "BenchmarkRunner",
    "BenchmarkRunnerConfig",
    "TaskOutcome",
    "run_benchmark_sync",
]

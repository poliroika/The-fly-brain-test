"""Phase-10 benchmark suite.

Implements the four canonical benchmarks called out in PLAN.md §608
(``humaneval``, ``gsm8k``, ``bbh_mini``, ``synthetic_routing``) plus
the async `BenchmarkRunner` that drives them through `MAS.run`.

Top-level entry points:

* `load_benchmark(name, ...)` — registry-driven loader returning a
  list of `BenchmarkTask`.
* `BenchmarkRunner` — drives the loaded tasks through a (MAS,
  Controller) pair, persists traces, returns a `BenchmarkRunReport`.

Metric extraction lives in `flybrain.eval.metrics`.
"""

from __future__ import annotations

from flybrain.benchmarks.base import KNOWN_TASK_TYPES, BenchmarkTask
from flybrain.benchmarks.bbh_mini import load_bbh_mini
from flybrain.benchmarks.gsm8k import load_gsm8k, parse_gsm8k_answer
from flybrain.benchmarks.humaneval import load_humaneval
from flybrain.benchmarks.loaders import (
    BENCHMARK_REGISTRY,
    DEFAULT_BENCHMARK_SUITE,
    BenchmarkLoader,
    list_benchmarks,
    load_benchmark,
)
from flybrain.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkRunnerConfig,
    BenchmarkRunReport,
    TaskOutcome,
    run_benchmark_sync,
)
from flybrain.benchmarks.synthetic_routing import load_synthetic_routing

__all__ = [
    "BENCHMARK_REGISTRY",
    "DEFAULT_BENCHMARK_SUITE",
    "KNOWN_TASK_TYPES",
    "BenchmarkLoader",
    "BenchmarkRunReport",
    "BenchmarkRunner",
    "BenchmarkRunnerConfig",
    "BenchmarkTask",
    "TaskOutcome",
    "list_benchmarks",
    "load_bbh_mini",
    "load_benchmark",
    "load_gsm8k",
    "load_humaneval",
    "load_synthetic_routing",
    "parse_gsm8k_answer",
    "run_benchmark_sync",
]

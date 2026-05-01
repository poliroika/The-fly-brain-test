"""Single registry over the four Phase-10 benchmark loaders.

Lets `scripts/run_benchmarks.py` and the eval pipeline iterate
benchmarks by name instead of importing each module. Each loader
exposes the same ``(num_tasks=None, seed=0, path=None) -> list[BenchmarkTask]``
signature so the runner stays format-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from flybrain.benchmarks.base import BenchmarkTask
from flybrain.benchmarks.bbh_mini import load_bbh_mini
from flybrain.benchmarks.gsm8k import load_gsm8k
from flybrain.benchmarks.humaneval import load_humaneval
from flybrain.benchmarks.synthetic_routing import load_synthetic_routing

BenchmarkLoader = Callable[..., list[BenchmarkTask]]

BENCHMARK_REGISTRY: dict[str, BenchmarkLoader] = {
    "humaneval": load_humaneval,
    "gsm8k": load_gsm8k,
    "bbh_mini": load_bbh_mini,
    "synthetic_routing": load_synthetic_routing,
}

# What the runner uses when nothing is specified.
DEFAULT_BENCHMARK_SUITE: tuple[str, ...] = (
    "humaneval",
    "gsm8k",
    "bbh_mini",
    "synthetic_routing",
)


def list_benchmarks() -> list[str]:
    """Names of all known benchmarks."""
    return sorted(BENCHMARK_REGISTRY)


def load_benchmark(
    name: str,
    *,
    num_tasks: int | None = None,
    seed: int = 0,
    path: Path | None = None,
) -> list[BenchmarkTask]:
    """Materialise one benchmark by name.

    `synthetic_routing` doesn't have a `path` argument; passing one is
    silently ignored so the four loaders share a uniform call site.
    """
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}; choose one of {list_benchmarks()}")
    loader = BENCHMARK_REGISTRY[name]
    if name == "synthetic_routing":
        n = 200 if num_tasks is None else num_tasks
        return loader(num_tasks=n, seed=seed)
    return loader(num_tasks=num_tasks, seed=seed, path=path)


__all__ = [
    "BENCHMARK_REGISTRY",
    "DEFAULT_BENCHMARK_SUITE",
    "BenchmarkLoader",
    "list_benchmarks",
    "load_benchmark",
]

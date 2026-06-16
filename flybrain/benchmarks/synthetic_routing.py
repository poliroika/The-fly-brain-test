"""Synthetic routing benchmark — re-uses the Phase-6 task generator.

This is the cheapest of the four benchmarks: the controller's job is
purely to pick the right agent route. There is no LLM call required
to score it (the synthetic verifier in `flybrain.sim.synthetic_mas`
already does that), so we keep the benchmark CI-friendly by relying
on `flybrain.sim.task_generator.TaskGenerator`.
"""

from __future__ import annotations

from collections.abc import Iterable

from flybrain.benchmarks.base import KNOWN_TASK_TYPES, BenchmarkTask
from flybrain.sim.task_generator import TaskGenerator


def load_synthetic_routing(
    *,
    num_tasks: int = 200,
    seed: int = 0,
    task_types: Iterable[str] | None = None,
) -> list[BenchmarkTask]:
    """Materialise a deterministic synthetic-routing dataset.

    Parameters
    ----------
    num_tasks:
        How many tasks to draw. The mix follows the round-robin from
        `task_types` so the four task families are evenly represented.
    seed:
        RNG seed; the same `(num_tasks, seed)` always produces the
        same task sequence.
    task_types:
        Iterable of `flybrain.benchmarks.base.KNOWN_TASK_TYPES`.
        Defaults to the four canonical task types.
    """
    types = tuple(task_types) if task_types is not None else KNOWN_TASK_TYPES
    if not types:
        raise ValueError("task_types must contain at least one entry")
    for tt in types:
        if tt not in KNOWN_TASK_TYPES:
            raise ValueError(f"unknown task_type {tt!r}; expected one of {KNOWN_TASK_TYPES}")

    gen = TaskGenerator(seed=seed)
    tasks: list[BenchmarkTask] = []
    for i in range(num_tasks):
        tt = types[i % len(types)]
        s = gen.sample(task_type=tt)
        tasks.append(
            BenchmarkTask(
                task_id=f"synthetic_routing/{i:04d}",
                task_type=s.task_type,
                prompt=s.prompt,
                ground_truth=None,
                benchmark="synthetic_routing",
                metadata={"source_id": s.task_id},
            )
        )
    return tasks


__all__ = ["load_synthetic_routing"]

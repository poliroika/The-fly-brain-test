"""Common benchmark task type used by all Phase-10 loaders.

`BenchmarkTask` extends `flybrain.runtime.Task` with optional reference
material (canonical answer, unit tests, multiple-choice options, …)
that the verification pipeline and per-task metric extractor consume.
The runtime itself only looks at `task_id`, `task_type`, `prompt` and
`ground_truth`; everything else lives in `metadata` so the underlying
`Task` shape stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flybrain.runtime import Task

# Task types accepted by the runner / controller. Mirrors
# ``flybrain.sim.optimal_routes.TASK_TYPES`` but is repeated here so
# the benchmarks package doesn't depend on the simulation layer.
KNOWN_TASK_TYPES: tuple[str, ...] = ("coding", "math", "research", "tool_use")


@dataclass(slots=True)
class BenchmarkTask:
    """One benchmark example.

    The fields are deliberately a superset of `flybrain.runtime.Task` —
    `to_runtime_task()` projects back down so the runner never has to
    know about benchmarks-specific fields.

    Attributes
    ----------
    task_id:
        Stable, dataset-prefixed id (e.g. ``"humaneval/HumanEval-0"``).
    task_type:
        One of `KNOWN_TASK_TYPES`. Used by the verification pipeline
        to dispatch to the right verifier.
    prompt:
        The text the MAS sees as the user prompt.
    ground_truth:
        Canonical answer. Whatever the verification pipeline expects
        for this task type — typically a string for math / research /
        tool_use, or an entry-point name for coding.
    unit_tests:
        Optional executable Python tests for coding tasks (HumanEval).
    options:
        Optional multiple-choice options (BBH).
    benchmark:
        Source dataset name. Used as the row group in the eval tables.
    metadata:
        Free-form bag for anything dataset-specific (entry_point,
        canonical_solution, target, …) — kept out of the trace's
        ``Task`` for hygiene.
    """

    task_id: str
    task_type: str
    prompt: str
    ground_truth: Any = None
    unit_tests: str | None = None
    options: list[str] = field(default_factory=list)
    benchmark: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_runtime_task(self) -> Task:
        """Project to the lightweight `Task` the runner ingests."""
        return Task(
            task_id=self.task_id,
            task_type=self.task_type,
            prompt=self.prompt,
            ground_truth=self.ground_truth,
        )


__all__ = ["KNOWN_TASK_TYPES", "BenchmarkTask"]

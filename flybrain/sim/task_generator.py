"""Synthetic task templates for Phase-6 simulation pretraining.

The templates are deliberately simple so the synthetic dataset is:

1. Cheap to generate (no LLM calls, microsecond per task).
2. Deterministic given a seed (re-runs in CI produce the same dataset).
3. Diverse enough that the controller learns the routing logic
   rather than memorising prompts.

Phase-7 (Imitation Learning) overlays *real* prompts from
HumanEval / GSM8K / etc. on top of this distribution.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from flybrain.sim.optimal_routes import TASK_TYPES

# A handful of natural-sounding stems per task type. These are
# concatenated with random integers / nouns so duplicates are rare.
_TEMPLATES: dict[str, tuple[str, ...]] = {
    "coding": (
        "Write a function that {verb} a list of integers.",
        "Implement {algo} in Python with full type hints.",
        "Refactor the following snippet so it passes mypy: ...{i}",
        "Add unit tests for the {component} module (id={i}).",
        "Fix the off-by-one bug in `{component}` (issue #{i}).",
    ),
    "math": (
        "Compute the {what} of {a} and {b}.",
        "Solve for x: {a}x + {b} = {c}.",
        "What is the probability that ... (case #{i})?",
        "Evaluate the integral of f(x) = x^{a} from 0 to {b}.",
        "Simplify: ({a} + {b})^{c}.",
    ),
    "research": (
        "Summarise the latest results on {topic}.",
        "Find three citations for the claim: {claim}",
        "Compare approach A (id={i}) and approach B (id={j}).",
        "Write a 200-word literature review on {topic}.",
        "Trace the history of {topic} since 19{a0}{b0}.",
    ),
    "tool_use": (
        "Use the {tool} tool to fetch ... (case #{i}).",
        "Validate the JSON payload against schema {schema_id}.",
        "Run `{cmd}` and parse the output for case #{i}.",
        "Call the {tool} API with parameter {a} and report {what}.",
        "Validate the input from form #{i}: ...",
    ),
}

_VERBS = ("sorts", "filters", "chunks", "deduplicates", "shuffles")
_ALGOS = ("merge sort", "Dijkstra", "DFS", "Kadane", "Floyd-Warshall")
_COMPONENTS = ("scheduler", "tokenizer", "cache", "router", "verifier")
_WHATS = ("mean", "variance", "median", "GCD", "LCM")
_TOPICS = ("transformers", "GNNs", "MAS routing", "RLHF", "fly connectomes")
_CLAIMS = (
    "GNNs match transformers on small graphs",
    "Mixture-of-experts beats dense models",
    "PPO is unstable below 32 minibatch",
)
_TOOLS = ("search", "python", "calculator", "ls", "wikipedia")
_CMDS = ("python -V", "ls -la", "git status", "uname -a")


@dataclass(slots=True, frozen=True)
class SyntheticTask:
    task_id: str
    task_type: str
    prompt: str


class TaskGenerator:
    """Deterministic random sampler over the four task templates."""

    def __init__(self, *, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def _format(self, template: str) -> str:
        rng = self.rng
        return template.format(
            verb=rng.choice(_VERBS),
            algo=rng.choice(_ALGOS),
            component=rng.choice(_COMPONENTS),
            what=rng.choice(_WHATS),
            topic=rng.choice(_TOPICS),
            claim=rng.choice(_CLAIMS),
            tool=rng.choice(_TOOLS),
            cmd=rng.choice(_CMDS),
            schema_id=rng.randrange(1000),
            i=rng.randrange(1, 10_000),
            j=rng.randrange(1, 10_000),
            a=rng.randrange(1, 100),
            b=rng.randrange(1, 100),
            c=rng.randrange(1, 100),
            a0=rng.randrange(0, 10),
            b0=rng.randrange(0, 10),
        )

    def sample(self, *, task_type: str | None = None) -> SyntheticTask:
        tt = task_type if task_type is not None else self.rng.choice(TASK_TYPES)
        templates = _TEMPLATES[tt]
        prompt = self._format(self.rng.choice(templates))
        task_id = f"sim-{tt}-{self.rng.randrange(2**31)}"
        return SyntheticTask(task_id=task_id, task_type=tt, prompt=prompt)

    def stream(self, *, n: int, task_type: str | None = None) -> Iterator[SyntheticTask]:
        for _ in range(n):
            yield self.sample(task_type=task_type)

    def balanced_dataset(self, *, n_per_type: int) -> list[SyntheticTask]:
        tasks: list[SyntheticTask] = []
        for tt in TASK_TYPES:
            tasks.extend(self.stream(n=n_per_type, task_type=tt))
        return tasks


__all__ = ["SyntheticTask", "TaskGenerator"]

"""HumanEval loader (PLAN.md §608).

Reads the OpenAI HumanEval JSONL release from
`data/benchmarks/humaneval/HumanEval.jsonl` (one JSON object per
line). When the file is missing the loader falls back to a small
bundled fixture in ``data/benchmarks/fixtures/humaneval.jsonl`` so
unit tests and CI never need network access.

Each line is expected to contain at least::

    {
        "task_id": "HumanEval/0",
        "prompt": "...function header + docstring...",
        "entry_point": "has_close_elements",
        "canonical_solution": "...",
        "test": "def check(candidate): ..."
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from flybrain.benchmarks.base import BenchmarkTask

DEFAULT_DATA_PATH = Path("data/benchmarks/humaneval/HumanEval.jsonl")
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/humaneval.jsonl"


def _resolve_path(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH
    return FIXTURE_PATH


def _format_prompt(prompt: str, entry_point: str) -> str:
    """Wrap the HumanEval prompt with a Coder-friendly instruction."""
    return (
        "Implement the following Python function. Your final answer must be a "
        "complete, runnable function definition.\n\n"
        f"Entry point: `{entry_point}`\n\n"
        "```python\n"
        f"{prompt.rstrip()}\n"
        "```\n"
    )


def load_humaneval(
    *,
    num_tasks: int | None = None,
    seed: int = 0,
    path: Path | None = None,
) -> list[BenchmarkTask]:
    """Load HumanEval tasks from `path` or the bundled fixture.

    Parameters
    ----------
    num_tasks:
        Cap the dataset to the first ``num_tasks`` entries (after a
        deterministic shuffle by `seed`). ``None`` returns all rows.
    seed:
        Seed for the deterministic shuffle. Same input file always
        produces the same ordering for a given seed.
    path:
        Optional explicit JSONL path. Defaults to the canonical
        download location, then the bundled fixture.
    """
    src = _resolve_path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"HumanEval JSONL not found at {src!s}. "
            "Either download via `scripts/download_benchmarks.sh humaneval` "
            "or pass an explicit `path=`."
        )
    rows: list[dict] = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("task_id", ""))
    # Deterministic shuffle keyed on seed so different seeds expose
    # different held-out subsets but a given seed is reproducible.
    import random as _random

    rng = _random.Random(seed)
    rng.shuffle(rows)
    if num_tasks is not None:
        rows = rows[:num_tasks]

    tasks: list[BenchmarkTask] = []
    for row in rows:
        raw_task_id = str(row.get("task_id", ""))
        entry_point = str(row.get("entry_point", ""))
        prompt = str(row.get("prompt", ""))
        test_block = row.get("test", "")
        canonical = row.get("canonical_solution", "")
        tasks.append(
            BenchmarkTask(
                task_id=f"humaneval/{raw_task_id.replace('/', '-')}",
                task_type="coding",
                prompt=_format_prompt(prompt, entry_point),
                ground_truth=canonical,
                unit_tests=str(test_block),
                benchmark="humaneval",
                metadata={
                    "entry_point": entry_point,
                    "raw_task_id": raw_task_id,
                    "canonical_solution": canonical,
                },
            )
        )
    return tasks


__all__ = ["load_humaneval"]

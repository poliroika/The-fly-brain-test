"""BIG-Bench Hard mini subset loader (PLAN.md §608).

BBH ships per-task JSON files (``"examples": [{"input": ..., "target": ...}]``).
The mini subset packs everything into a single JSONL where each row
represents one example::

    {
        "subtask": "logical_deduction_three_objects",
        "input": "...",
        "target": "(A)",
        "options": ["(A) ...", "(B) ...", "(C) ..."]   # optional
    }

Tasks are routed to the controller as ``research`` because BBH is
free-form reasoning over short prompts (closer to the README's
research route than to math/coding).
"""

from __future__ import annotations

import json
from pathlib import Path

from flybrain.benchmarks.base import BenchmarkTask

DEFAULT_DATA_PATH = Path("data/benchmarks/bbh_mini/bbh_mini.jsonl")
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/bbh_mini.jsonl"


def _resolve_path(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH
    return FIXTURE_PATH


def load_bbh_mini(
    *,
    num_tasks: int | None = None,
    seed: int = 0,
    path: Path | None = None,
) -> list[BenchmarkTask]:
    """Load the BBH-mini subset from `path` or the bundled fixture."""
    src = _resolve_path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"BBH-mini JSONL not found at {src!s}. "
            "Either download via `scripts/download_benchmarks.sh bbh_mini` "
            "or pass an explicit `path=`."
        )
    rows: list[dict] = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    import random as _random

    rng = _random.Random(seed)
    rng.shuffle(rows)
    if num_tasks is not None:
        rows = rows[:num_tasks]

    tasks: list[BenchmarkTask] = []
    for i, row in enumerate(rows):
        prompt = str(row.get("input", "")).strip()
        target = str(row.get("target", "")).strip()
        subtask = str(row.get("subtask", "general"))
        options = list(row.get("options") or [])
        rendered_prompt = prompt
        if options:
            rendered_prompt += "\n\nOptions:\n" + "\n".join(options)
        rendered_prompt += "\n\nGive the final answer as `final_answer: <choice>`."
        tasks.append(
            BenchmarkTask(
                task_id=f"bbh_mini/{subtask}/{i:04d}",
                task_type="research",
                prompt=rendered_prompt,
                ground_truth=target,
                options=options,
                benchmark="bbh_mini",
                metadata={
                    "subtask": subtask,
                    "raw_input": prompt,
                    "target": target,
                },
            )
        )
    return tasks


__all__ = ["load_bbh_mini"]

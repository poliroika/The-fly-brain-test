"""GSM8K loader (PLAN.md §608).

GSM8K is distributed as JSONL with one ``{"question": ..., "answer": ...}``
per line. The reference answer block ends with ``#### <number>`` —
that suffix is the canonical numeric answer and the rest is the
chain-of-thought used for IL only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from flybrain.benchmarks.base import BenchmarkTask

DEFAULT_DATA_PATH = Path("data/benchmarks/gsm8k/test.jsonl")
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/gsm8k.jsonl"

_FINAL_RE = re.compile(r"####\s*([\-\+]?[\d,\.]+)")


def _resolve_path(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH
    return FIXTURE_PATH


def parse_gsm8k_answer(answer_block: str) -> str:
    """Extract the canonical numeric answer from a GSM8K ``answer`` field.

    Falls back to the trimmed last line if no ``####`` marker is
    present (e.g. when callers feed in already-parsed answers).
    """
    m = _FINAL_RE.search(answer_block)
    if m is not None:
        return m.group(1).replace(",", "").strip()
    return answer_block.strip().splitlines()[-1].strip()


def load_gsm8k(
    *,
    num_tasks: int | None = None,
    seed: int = 0,
    path: Path | None = None,
) -> list[BenchmarkTask]:
    """Load GSM8K tasks from `path` or the bundled fixture."""
    src = _resolve_path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"GSM8K JSONL not found at {src!s}. "
            "Either download via `scripts/download_benchmarks.sh gsm8k` "
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
        question = str(row.get("question", "")).strip()
        answer_block = str(row.get("answer", ""))
        final = parse_gsm8k_answer(answer_block)
        tasks.append(
            BenchmarkTask(
                task_id=f"gsm8k/{i:05d}",
                task_type="math",
                prompt=question + "\n\nThink step by step and end with `final_answer: <number>`.",
                ground_truth=final,
                benchmark="gsm8k",
                metadata={
                    "answer_block": answer_block,
                    "question": question,
                },
            )
        )
    return tasks


__all__ = ["load_gsm8k", "parse_gsm8k_answer"]

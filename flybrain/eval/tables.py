"""Comparison-table builders for the Phase-10 evaluation step.

The output format follows README §17: one row per method, columns
for success / verifier-pass / tokens / calls / latency / cost-per-
solved. Two flavours are provided:

* `markdown_table(rows)` — plain markdown that drops verbatim into
  the README / final report.
* `csv_table(rows)` — a no-dependency CSV writer keyed on the same
  column order as the markdown.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable

from flybrain.eval.metrics import AggregateMetrics

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("name", "Method"),
    ("benchmark", "Benchmark"),
    ("num_tasks", "Tasks"),
    ("success_rate", "Success"),
    ("verifier_pass_rate", "Verifier"),
    ("avg_total_tokens", "Tokens/task"),
    ("avg_llm_calls", "Calls/task"),
    ("avg_latency_ms", "Latency (ms)"),
    ("avg_cost_rub", "Cost/task ₽"),
    ("cost_per_solved_rub", "Cost/solved ₽"),
)


def _fmt_value(value: float | int | str | None) -> str:
    # Strict-JSON serialisation maps non-finite floats (e.g. an infinite
    # `cost_per_solved_rub`) to `null`, which loads back as `None`; render
    # those as ∞ rather than crashing.
    if value is None:
        return "∞"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if value == float("inf"):
        return "∞"
    if abs(value) < 1.0:
        return f"{value:.3f}"
    if abs(value) < 100.0:
        return f"{value:.2f}"
    return f"{value:.0f}"


def _row_values(row: AggregateMetrics) -> list[str]:
    out: list[str] = []
    for attr, _ in _COLUMNS:
        v = getattr(row, attr)
        out.append(_fmt_value(v))
    return out


def markdown_table(rows: Iterable[AggregateMetrics]) -> str:
    """Render a list of `AggregateMetrics` as a markdown table."""
    headers = [label for _, label in _COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_row_values(row)) + " |")
    return "\n".join(lines) + "\n"


def csv_table(rows: Iterable[AggregateMetrics]) -> str:
    """Render a list of `AggregateMetrics` as CSV (header included)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([attr for attr, _ in _COLUMNS])
    for row in rows:
        writer.writerow([_fmt_value(getattr(row, attr)) for attr, _ in _COLUMNS])
    return buf.getvalue()


__all__ = ["csv_table", "markdown_table"]

"""Phase-10 evaluation helpers.

* `metrics.py` — per-task and per-method metric extraction from
  `flybrain.runtime` traces.
* `tables.py` — README §17 comparison-table builder (Markdown/CSV).
* `reports.py` — `flybrain-py report` skeleton stitching the tables
  and cherry-picked traces into a single research report.
"""

from __future__ import annotations

from flybrain.eval.metrics import (
    AggregateMetrics,
    TaskMetrics,
    aggregate,
    metrics_from_trace,
    metrics_from_trace_path,
)
from flybrain.eval.reports import ReportInputs, render_report, write_report
from flybrain.eval.tables import csv_table, markdown_table

__all__ = [
    "AggregateMetrics",
    "ReportInputs",
    "TaskMetrics",
    "aggregate",
    "csv_table",
    "markdown_table",
    "metrics_from_trace",
    "metrics_from_trace_path",
    "render_report",
    "write_report",
]

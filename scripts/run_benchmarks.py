#!/usr/bin/env python3
"""Phase-10 evaluation harness — runs N baselines × M benchmarks
end-to-end and emits comparison tables (PLAN.md §607-611, §17).

Usage::

    # Smoke run on bundled fixtures + mock LLM (no API, CI-friendly):
    python scripts/run_benchmarks.py \\
        --suite full_min --backend mock \\
        --benchmarks humaneval gsm8k bbh_mini synthetic_routing \\
        --tasks-per-benchmark 3 \\
        --output runs/bench_smoke

    # Live run on YandexGPT (caches + budget tracker):
    YANDEX_API_KEY=... folder_id=... \\
    python scripts/run_benchmarks.py \\
        --suite full_min --backend yandex \\
        --tasks-per-benchmark 40 --budget-rub 300 \\
        --output data/benchmarks/v1

The actual orchestration lives in `flybrain.benchmarks.cli` so the
same code path is shared with `flybrain-py bench`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from flybrain.baselines import BUILTIN_SUITES
from flybrain.benchmarks import list_benchmarks
from flybrain.benchmarks.cli import run as _run


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=sorted(BUILTIN_SUITES), default="full_min")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--benchmarks", nargs="*", choices=list_benchmarks(), default=None)
    p.add_argument("--backend", choices=("mock", "yandex"), default="mock")
    p.add_argument("--tasks-per-benchmark", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--budget-rub", type=float, default=300.0)
    p.add_argument("--parallelism", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("runs/benchmarks"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

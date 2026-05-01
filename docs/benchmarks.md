# Phase 10 — benchmarks & evaluation runbook

This document is the operator-facing companion to `flybrain/benchmarks/`,
`flybrain/eval/`, and `scripts/run_benchmarks.py`. It explains how to run
the four canonical benchmarks (HumanEval, GSM8K, BBH-mini, synthetic
routing), where to put the data, and what the comparison artefacts mean.

## What's evaluated

Suite (PLAN.md §608, README §17):

| Benchmark           | Task type       | Default size | Timeout |
| ------------------- | --------------- | ------------ | ------- |
| `humaneval`         | `coding`        | 80           | 30 s    |
| `gsm8k`             | `math`          | 100          | 20 s    |
| `bbh_mini`          | `research`      | 50           | 30 s    |
| `synthetic_routing` | mixed (4 types) | 200          | 5 s     |

Methods are the nine baselines from `flybrain.baselines.BUILTIN_SUITES`
(`full_min`).

## Data layout

```
data/
  benchmarks/
    fixtures/                        # tiny in-repo fallback (3-5 tasks each)
      humaneval.jsonl
      gsm8k.jsonl
      bbh_mini.jsonl
    humaneval/HumanEval.jsonl        # full dataset (downloaded)
    gsm8k/test.jsonl
    bbh_mini/bbh_mini.jsonl
```

Loaders auto-fall-back to the bundled fixtures when the canonical path
is missing, so smoke runs work without network access.

## Running the smoke suite (no network, no LLM cost)

```bash
python scripts/run_benchmarks.py \
    --suite static \
    --backend mock \
    --tasks-per-benchmark 3 \
    --output runs/bench_smoke
```

Or via the consolidated CLI:

```bash
flybrain-py bench --suite static --backend mock \
    --tasks-per-benchmark 3 \
    --output runs/bench_smoke
```

## Running the full suite against YandexGPT

Set credentials and budget cap, then:

```bash
export YANDEX_API_KEY=...
export folder_id=...
flybrain-py bench --suite full_min --backend yandex \
    --tasks-per-benchmark 40 \
    --budget-rub 300 \
    --parallelism 4 \
    --max-retries 2 \
    --output data/benchmarks/v1
```

The harness writes `<output>/<baseline>/<benchmark>/<task_id>.trace.json`
for every run, plus per-benchmark and overall comparison tables in
Markdown / JSON / CSV.

## Output layout

```
<output>/
  <baseline>/
    <benchmark>/
      <task_id>.trace.json           # one per task
      summary.json                   # per-benchmark roll-up
  summaries.json                     # everything in one file
  comparison_overall.{md,json,csv}   # one row per baseline
  comparison_<benchmark>.{md,json,csv}
  report.md                          # final-report skeleton
```

## Building the final report

```bash
flybrain-py report --input data/benchmarks/v1 \
    --output docs/final_report.md \
    --cherry-picks \
        data/benchmarks/v1/flybrain_rl/humaneval/HumanEval-23.trace.json
```

The renderer reads every `comparison_*.json` from `--input`, regenerates
the headline tables, and inlines the cherry-picked traces in the §4
section. Writing the discussion is on you.

## Adding a new benchmark

1. Drop a loader at `flybrain/benchmarks/<name>.py` that returns a
   `list[BenchmarkTask]`.
2. Register it in `flybrain.benchmarks.loaders.BENCHMARK_REGISTRY`.
3. Add a fixture at `data/benchmarks/fixtures/<name>.jsonl` so smoke
   runs stay deterministic.
4. Bump `configs/eval/full_min.yaml`.

The runner is format-agnostic: every loader produces the same
`BenchmarkTask` shape, so `BenchmarkRunner.run_benchmark` and the
`flybrain.eval` pipeline pick it up for free.

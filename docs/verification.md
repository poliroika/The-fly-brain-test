# Verification Layer (Phase 3)

The verification layer turns a finished or in-progress `Trace` into a single
`VerificationResult` that the controller can read every tick. It splits cleanly
into two halves:

* **Deterministic verifiers** (Rust, exposed via PyO3): schema, tool-use,
  trace-structure, unit-test, budget. No network, no randomness.
* **LLM-backed judges** (Python, async): factual, reasoning. They call the
  pluggable `LLMClient` (Yandex / mock / cached) and parse a structured JSON
  envelope.

Everything projects into one `VerificationResult`:

```python
@dataclass(slots=True)
class VerificationResult:
    passed: bool
    score: float
    errors: list[str]
    warnings: list[str]
    failed_component: str | None
    suggested_next_agent: str | None
    reward_delta: float
```

The Rust struct has the same shape (`flybrain_core::VerificationResult`) and
serialises through PyO3 as a plain dict, so a Phase-5 GNN/RNN controller can
consume the dict directly without ever touching Python verifier classes.

---

## Components

| Component  | Lives in                                      | Async? | Detects                                          |
|------------|-----------------------------------------------|:------:|--------------------------------------------------|
| `schema`   | `flybrain-verify::schema` + `verify_py.rs`    | sync   | Missing fields, wrong types, regex/range bounds  |
| `tool_use` | `flybrain-verify::tool_use`                   | sync   | Tools outside the allow-list, missing required args |
| `trace`    | `flybrain-verify::trace`                      | sync   | step_id discontinuities, totals/sum mismatches, missing final_answer |
| `unit_test`| `flybrain-verify::unit_test`                  | sync   | `failed > 0`, `all_passed == false`              |
| `budget`   | `flybrain-verify::budget`                     | sync   | Token / call / cost limit overruns               |
| `factual`  | `flybrain.verification.llm.factual`           | async  | Candidate answer vs reference fact-check         |
| `reasoning`| `flybrain.verification.llm.reasoning`         | async  | Coherence, task-addressing, internal consistency |

The schema verifier covers a hand-written subset of JSON Schema: `type`,
`required`, `properties`, `items`, `enum`, `minLength`, `maxLength`, `pattern`,
`minimum`, `maximum`. `pattern` is matched with `String::contains`, not a real
regex engine — sufficient for what agents emit and dependency-free.

---

## Pipeline

`flybrain.verification.VerificationPipeline` composes all of the above per
`task_type`. The default dispatch table:

| task_type           | schema | tool_use | unit_test | factual | reasoning |
|---------------------|:------:|:--------:|:---------:|:-------:|:---------:|
| `coding`            |   ✔    |    ✔     |     ✔     |         |           |
| `math`              |   ✔    |          |           |         |     ✔     |
| `research`          |   ✔    |          |           |    ✔    |           |
| `tool_use`          |   ✔    |    ✔     |           |         |           |
| `synthetic_routing` |   ✔    |          |           |         |           |
| any other           |   ✔    |    ✔     |     ✔     |         |           |

`VerificationConfig.for_task_type(...)` returns a config that mirrors this
table. The pipeline always runs the deterministic verifiers first; LLM judges
are only invoked when both the relevant flag is on **and** an `LLMClient`-backed
judge has been wired in. A flagged-but-missing judge is silently skipped — the
test suite can keep determinism by not configuring judges, while production runs
configure the Yandex client.

`aggregate(results)` is the projection rule:

* `passed`            = AND over every individual result
* `score`             = mean over scores; capped at 0.5 if any failed
* `errors` / `warnings` = concatenated, prefixed with `[component]`
* `failed_component`  = first failing component
* `suggested_next_agent` = inherited from that first failure
* `reward_delta`      = sum

---

## Wiring into `MAS.run`

The runner keeps a cheap **rule-based component check** (`_RuleVerifier`) that
inspects `state.produced_components` against the per-task-type plan
(`coding` → `{plan, code, tests_run}`, etc.). This catches "controller skipped
the Coder" without invoking the LLM.

On every `call_verifier` action and at the end of the run, the runner builds a
`VerificationContext` (candidate answer, tool calls observed so far, last
unit-test payload, ground truth) and calls `pipeline.run_async(...)`. The two
verdicts are merged via `MAS._merge_verifier_results`:

* `passed` = rule AND pipeline
* `score`  = mean
* errors / warnings concatenated (rule entries are tagged `[rule]`)
* `failed_component` / `suggested_next_agent` come from the rule check first
  (it's a more actionable signal than a downstream pipeline failure)
* `reward_delta` = sum

The merged dict is what the controller actually sees on `state.last_verifier_*`.

---

## Determinism

Phase 3 is fully deterministic when judges are off. The unit tests
(`tests/python/unit/test_verification_*`) and the integration test
(`tests/python/integration/test_mas_runtime_mock.py`) run without network and
produce byte-identical outputs across runs. When a judge is wired with a
`MockLLMClient`, the response is fixed by a regex-keyed rule, so determinism
holds end-to-end.

---

## Exit criteria (PLAN.md §562)

All four deterministic verifiers ship with unit tests on both pass and fail
paths:

* `cargo test -p flybrain-verify` → 24 passing
* `pytest tests/python/unit/test_verifier_native.py` → 9 passing
* `pytest tests/python/unit/test_verification_pipeline.py` → 12 passing
* `pytest tests/python/unit/test_verification_judges.py` → 9 passing
* `pytest tests/python/integration/test_mas_runtime_mock.py` → 4 passing
* `cargo clippy --all-targets -- -D warnings` clean
* `cargo fmt --all -- --check` clean
* `ruff check`, `ruff format --check`, `mypy flybrain` clean

The `VerificationResult` shape has not changed since Phase 2; both the Rust
and Python sides round-trip through `verification_result_round_trip` (see
`tests/python/unit/test_verifier_native.py::test_verification_result_round_trip_through_python`).

# Phase 0–11 verification — Yandex SDK migration branch

> Re-run by walking each PLAN.md phase end-to-end on top of `pr-11`
> after migrating from `yandex-cloud-ml-sdk` to `yandex-ai-studio-sdk`.

| Phase | Subject | Status | Evidence |
|---|---|---|---|
| 0 | Bootstrap (workspace, configs, infra, CI) | OK | 6 cargo crates compile; Hydra `default.yaml` composes; 3 GitHub workflows present (`ci.yml`, `python.yml`, `rust.yml`); `infra/Dockerfile`, `infra/docker-compose.yaml`, `infra/datasphere/*`, `infra/terraform/*` all in tree. |
| 1 | Graph builder (`flybrain-graph`) | OK | `flybrain-py build --source synthetic --num-nodes 200 --method <m> -k 16` succeeds for every method (`region_agg`, `celltype_agg`, `louvain`, `leiden`, `spectral`); each writes a valid `.fbg` + `.node_metadata.json`. |
| 2 | MAS runtime + agents + tools | OK | `tests/python/integration/test_mas_runtime_mock.py` (4 tests) green; `load_minimal_15` returns 15 specs, `load_extended_25` returns 25; `flybrain.flybrain_native` exposes `Scheduler`/`MessageBus`/`TraceWriter`/26 helpers. |
| 3 | Verification layer | OK | `test_verifier_native` (9), `test_verification_judges` (9), `test_verification_pipeline` (12) — all green. |
| 4 | Embeddings (Yandex API + cache + mock) | OK | `test_embeddings_clients` (8), `test_embeddings_wrappers` (18) — all green; `YandexEmbeddingClient` migrated to `yandex-ai-studio-sdk` (drops the legacy `embedding(uri)` fallback). |
| 5 | Controller (Manual / Random / GNN / LearnedRouter) | OK | `test_controllers_torch` (27) green with PyTorch installed; action-space round-trip exercised. |
| 6 | Simulation pretraining | OK | `test_simulation_pretrain` (12) green; `flybrain-py sim` is documented as Phase-6 stub. |
| 7 | Imitation learning | OK | `test_imitation` (9) green. |
| 8 | RL / Bandit (REINFORCE/PPO/Bandit/Offline) | OK | `test_rl` (15) green; ruff `N806`/`N812` exemption added for `flybrain/training/rl/*.py` (PyTorch / linear-algebra naming). |
| 9 | Baselines (9 entries) | OK | `test_baselines` (18) green; smoke run confirms 9 baselines × 4 benchmarks × 3 tasks aggregate cleanly. |
| 10 | Benchmarks + eval | OK | `flybrain-py bench --suite full_min --backend mock --tasks-per-benchmark 3` produces `comparison_{overall,bbh_mini,gsm8k,humaneval,synthetic_routing}.{md,json,csv}` plus `report.md` and `summaries.json`. |
| 11 | Final report + cherry-picks + dashboard notebook | OK after fix | `flybrain-py report --input <bench_dir>` previously crashed with `TypeError: must be real number, not NoneType` because strict-JSON serialises `float("inf")` as `null` and `_cost_quality_section` ran `math.isfinite(None)`. Fixed; new regression test added. Notebook `04_results_dashboard.ipynb` exercises all cells against the smoke run. Cherry-picked traces under `data/traces/sample/{degree_preserving,flybrain_imitation}/bbh_mini/` are still present. |

## Tooling baseline (post-cleanup)

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace --exclude flybrain-py --all-targets -- -D warnings` | clean |
| `cargo test --workspace --exclude flybrain-py` | 76 tests pass (12 + 26 + 14 + 24) |
| `ruff check flybrain tests` | clean |
| `ruff format --check flybrain tests` | clean (118 files) |
| `mypy flybrain` | Success: no issues found in 90 source files |
| `pytest tests/python` | 261 passed, 1 skipped (`FLYBRAIN_RUN_LIVE_LLM=1`-gated) |

## Cleanup summary (this branch)

* **SDK swap.** `yandex-cloud-ml-sdk` → `yandex-ai-studio-sdk>=0.20,<1` everywhere
  (`pyproject.toml`, `infra/Dockerfile`, `flybrain/llm/yandex_client.py`,
  `flybrain/embeddings/yandex_client.py`, `flybrain/llm/__init__.py`,
  `docs/yandex_setup.md`, `docs/rust_python_boundary.md`). The new SDK exposes
  the same `sdk.models.completions(...)` / `sdk.models.text_embeddings(...)`
  surface used by Phase 0 and Phase 4, so the wrapper logic is unchanged
  beyond the import + class rename.
* **Embeddings cleanup.** Dropped the legacy `sdk.models.embedding(uri)`
  fallback — the new SDK only exposes `text_embeddings`.
* **Lint config.** Added `[tool.ruff.lint.per-file-ignores]` for
  `flybrain/training/rl/*.py` allowing `N806`/`N812` (matches the original
  `ruff check . --exclude flybrain/training` policy), and re-formatted
  `tests/python/unit/test_eval.py` against the project's `ruff format`.
* **Phase-11 bug fix.** `_cost_quality_section` in `flybrain/eval/reports.py`
  now treats `None` (the JSON-round-trip representation of `float("inf")`) as
  ∞ instead of crashing on `math.isfinite(None)`. Regression test added in
  `tests/python/unit/test_eval.py::test_render_report_handles_none_cost_per_solved`.

## Known caveats

* `uv pip install -e ".[dev]" --no-build-isolation` requires
  `RUST_LOG=warn MATURIN_LOG=warn` to silence maturin 1.13's `INFO` traces
  on stdout (otherwise uv parses the log line as the wheel filename).
  CI already sets `NO_COLOR=1`; consider extending the env in
  `.github/workflows/python.yml` if the workflow ever upgrades to maturin
  1.13+.
* `terraform validate` is exercised by `ci.yml` (`terraform-validate`
  job) and not run locally during this verification (no `terraform`
  binary on the box).

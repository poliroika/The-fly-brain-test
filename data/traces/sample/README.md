# Cherry-picked execution traces

Selected by `flybrain.eval.reports.select_cherry_picks` from the YandexGPT
`full_min` run (2025-05-01, see `docs/final_report.md`). Two diagnostic
examples on the same task to make the comparison literal:

* `degree_preserving/bbh_mini/bbh_mini__boolean_expressions__0001.trace.json`
  — highest-scoring solved trace (verifier score 1.00, 5 steps,
  Planner → Retriever → Verifier → Finalizer → terminate). The
  reference shape of a healthy run on this benchmark.

* `flybrain_imitation/bbh_mini/bbh_mini__boolean_expressions__0001.trace.json`
  — the same task, run on the untrained `flybrain_imitation` baseline.
  Failed; verifier reports `failed_component=final_answer`. Useful as
  the ground truth for "what does an action-head failure look like in
  a trace" — the policy keeps cycling through agents and never returns
  `terminate`, so the verifier never sees a solution.

Browse interactively: <https://blessblissmari.github.io/flybrain-results/>
→ **Trace viewer** tab → pick the matching baseline / benchmark / task.

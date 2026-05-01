# Rust ↔ Python boundary

The single principle: **Rust hosts everything that does not depend on the LLM;
Python hosts everything that does.** This file explains where the line lives,
why, and how to extend it.

## What goes in Rust

* **Data types.** `AgentSpec`, `Trace*`, `GraphAction`, `VerificationResult`,
  `AgentGraph`, `FlyGraph`, `TaskSpec`. See
  [`docs/data_contracts.md`](data_contracts.md).
* **Connectome ops.** Loading 54M-edge FlyWire graphs and compressing them to
  K=32–256 via Louvain / Leiden / spectral / region / cell-type aggregation
  (Phase 1). Python would be too slow.
* **Runtime hot path.** Message bus, scheduler tick, trace JSONL writer,
  graph mutate API, deterministic graph hashing (Phase 2). Called every step.
* **Deterministic verifiers.** Schema, tool_use, budget, trace (loop / redundancy
  detection) (Phase 3). Hot path; must not flake.
* **CLI** for native-only commands (`flybrain build`, `flybrain sim` plumbing).

## What goes in Python

* **LLM client.** `yandex-cloud-ml-sdk` is Python-only.
* **Agent step.** Anything that talks to the LLM (prompts, tool dispatch, retry
  policy, function-calling).
* **Controller.** PyTorch + PyG; the Rust ML ecosystem (`tch-rs`, `burn`) is
  not yet competitive for GNN / GAT / attention research.
* **Embeddings.** `sentence-transformers` + Yandex `text-search-*`.
* **Training.** SSL, simulation, imitation, RL / bandit / PPO; PyTorch +
  Hydra.
* **LLM-judge verifiers.** Factual, reasoning. They call the LLM, so they
  belong in Python.
* **Benchmarks + eval + reporting.**

## How they talk

Maturin builds the workspace into a single Python extension module
`flybrain.flybrain_native` (configured via `[tool.maturin]` in
`pyproject.toml`). All cross-language calls go through this module.

Phase 0 only exposes JSON round-trip helpers and a few utilities; Phase 1+
adds typed handles (`AgentGraphHandle`, `RuntimeHandle`, etc.) so Python
code can hold references to long-lived Rust objects without re-deserializing
on every call.

### Crossing the boundary cheaply

* **Bulk traffic.** Pass a single JSON blob, not many small ones.
* **Long-lived objects.** Hold a Rust handle in Python; mutate via methods.
  Round-trip JSON only at boundaries (writing to disk, sending to LLM).
* **Numeric arrays.** When PyTorch tensors meet Rust, prefer NumPy-style
  zero-copy buffers (`numpy.ndarray` → `&[f32]`) over JSON.

## Adding a new shared type

1. Define it in `crates/flybrain-core/src/`. Derive `Serialize + Deserialize`.
2. Re-export from `lib.rs`.
3. Add a `*_round_trip` helper in `crates/flybrain-py/src/lib.rs`.
4. Add a unit test in `tests/python/unit/test_native_bindings.py`.

That's it — Python sees the type as a `dict`. When you need a typed handle,
add a `#[pyclass]` later, but only when the JSON cost actually matters.

## Adding a new Rust hot path

1. Land the function in the right crate (graph / runtime / verify).
2. Add a `#[pyfunction]` wrapper in `flybrain-py`.
3. Add `m.add_function(...)` in the `pymodule` block.
4. Run `make develop` to rebuild the .so file into the active venv.
5. Add a Python test that calls it.

//! PyO3 bindings exposing the FlyBrain Rust core to Python as `flybrain_native`.
//!
//! Phase 0 shipped JSON round-trip helpers + the budget verifier.
//! Phase 1 added the graph builder pipeline (`graph_py`).
//! Phase 2 adds the MAS runtime (`runtime_py`): `Scheduler`, `MessageBus`,
//! and `TraceWriter` are exposed as `#[pyclass]` types so Python can hold
//! a stateful handle across many ticks of the loop. Their methods still
//! exchange JSON-shaped dicts to keep the data contracts in sync with
//! `crates/flybrain-core` (see `docs/data_contracts.md`).

mod graph_py;
mod runtime_py;
mod verify_py;

use flybrain_core::action::GraphAction;
use flybrain_core::agent::AgentSpec;
use flybrain_core::graph::{AgentGraph, FlyGraph};
use flybrain_core::trace::Trace;
use flybrain_core::verify::VerificationResult;
use flybrain_graph::synthetic_fly_graph;
use flybrain_verify::budget::{BudgetState, BudgetVerifier};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    let s = serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let json_mod = py.import_bound("json")?;
    let parsed = json_mod.call_method1("loads", (s,))?;
    Ok(parsed.unbind())
}

pub(crate) fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let json_mod = py.import_bound("json")?;
    let s: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn round_trip<T>(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyObject>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let value = py_to_json(py, obj)?;
    let typed: T =
        serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let back = serde_json::to_value(&typed).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &back)
}

/// Validate and round-trip an `AgentSpec` dict via the Rust type. Useful for
/// catching schema drift in Python tests before we ship strongly-typed
/// bindings.
#[pyfunction]
fn agent_spec_round_trip(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    round_trip::<AgentSpec>(py, obj)
}

#[pyfunction]
fn graph_action_round_trip(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    round_trip::<GraphAction>(py, obj)
}

#[pyfunction]
fn verification_result_round_trip(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    round_trip::<VerificationResult>(py, obj)
}

#[pyfunction]
fn trace_round_trip(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    round_trip::<Trace>(py, obj)
}

#[pyfunction]
fn agent_graph_round_trip(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    round_trip::<AgentGraph>(py, obj)
}

#[pyfunction]
fn fly_graph_round_trip(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    round_trip::<FlyGraph>(py, obj)
}

/// Stable hash of an `AgentGraph` dict.
#[pyfunction]
fn agent_graph_hash(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let value = py_to_json(py, obj)?;
    let g: AgentGraph =
        serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(g.hash())
}

/// Build a deterministic fly-inspired graph (Phase 0 fallback for tests).
#[pyfunction]
#[pyo3(signature = (num_nodes, seed=42))]
fn build_synthetic_fly_graph(py: Python<'_>, num_nodes: usize, seed: u64) -> PyResult<PyObject> {
    let g = synthetic_fly_graph(num_nodes, seed);
    let value = serde_json::to_value(&g).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &value)
}

/// Run the budget verifier over a `(cost_rub, tokens_in, tokens_out, llm_calls)`
/// tuple and a hard cap. Returns a dict matching `VerificationResult`.
#[pyfunction]
fn budget_check(
    py: Python<'_>,
    hard_cap_rub: f32,
    cost_rub: f32,
    tokens_in: u64,
    tokens_out: u64,
    llm_calls: u32,
) -> PyResult<PyObject> {
    let v = BudgetVerifier::from_hard_cap_rub(hard_cap_rub);
    let state = BudgetState {
        tokens_in,
        tokens_out,
        llm_calls,
        cost_rub,
        latency_ms: 0,
    };
    let r = v.check(&state);
    let value = serde_json::to_value(&r).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &value)
}

#[pymodule]
fn flybrain_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let py = m.py();
    let modinfo = PyDict::new_bound(py);
    modinfo.set_item("crate", "flybrain-py")?;
    modinfo.set_item("phase", "4-embeddings")?;
    m.add("__modinfo__", modinfo)?;

    m.add_function(wrap_pyfunction!(agent_spec_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(graph_action_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(verification_result_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(trace_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(agent_graph_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(fly_graph_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(agent_graph_hash, m)?)?;
    m.add_function(wrap_pyfunction!(build_synthetic_fly_graph, m)?)?;
    m.add_function(wrap_pyfunction!(budget_check, m)?)?;

    graph_py::register(py, m)?;
    runtime_py::register(py, m)?;
    verify_py::register(py, m)?;

    Ok(())
}

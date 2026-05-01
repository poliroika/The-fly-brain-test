//! PyO3 bindings for the deterministic verifiers in `flybrain-verify`.
//!
//! All exposed functions take JSON-shaped Python dicts/lists, run the
//! corresponding Rust verifier, and return a dict matching
//! [`VerificationResult`](flybrain_core::verify::VerificationResult).

use flybrain_verify::{
    verify_schema, verify_tool_calls, verify_trace, verify_unit_tests, ToolUseSpec,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{json_to_py, py_to_json};

#[pyfunction]
fn schema_check(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload = py_to_json(py, payload)?;
    let schema = py_to_json(py, schema)?;
    let r = verify_schema(&payload, &schema);
    let value = serde_json::to_value(&r).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &value)
}

#[pyfunction]
#[pyo3(signature = (calls, allowed=None, requirements=None))]
fn tool_use_check(
    py: Python<'_>,
    calls: &Bound<'_, PyAny>,
    allowed: Option<&Bound<'_, PyAny>>,
    requirements: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let calls = py_to_json(py, calls)?;
    let calls_vec = match calls {
        serde_json::Value::Array(a) => a,
        _ => return Err(PyValueError::new_err("calls must be a list of dicts")),
    };

    let mut spec = ToolUseSpec::default();
    if let Some(allowed) = allowed {
        let v = py_to_json(py, allowed)?;
        spec.allowed =
            serde_json::from_value(v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(req) = requirements {
        let v = py_to_json(py, req)?;
        spec.requirements =
            serde_json::from_value(v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    let r = verify_tool_calls(&calls_vec, &spec);
    let value = serde_json::to_value(&r).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &value)
}

#[pyfunction]
fn trace_check(py: Python<'_>, trace: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let trace = py_to_json(py, trace)?;
    let r = verify_trace(&trace);
    let value = serde_json::to_value(&r).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &value)
}

#[pyfunction]
fn unit_test_check(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let payload = py_to_json(py, payload)?;
    let r = verify_unit_tests(&payload);
    let value = serde_json::to_value(&r).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &value)
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(schema_check, m)?)?;
    m.add_function(wrap_pyfunction!(tool_use_check, m)?)?;
    m.add_function(wrap_pyfunction!(trace_check, m)?)?;
    m.add_function(wrap_pyfunction!(unit_test_check, m)?)?;
    Ok(())
}

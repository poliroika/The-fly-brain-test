//! Phase-2 PyO3 bindings for the MAS runtime.
//!
//! Stateful runtime objects (`Scheduler`, `MessageBus`, `TraceWriter`)
//! are exposed as `#[pyclass]` because Python needs to hold a long-lived
//! handle across many `tick`s of the loop. Method arguments and return
//! values are still JSON-shaped (Python dicts) so we don't fork the
//! data contracts from `crates/flybrain-core` — see `docs/data_contracts.md`.

use crate::{json_to_py, py_to_json};
use flybrain_core::action::GraphAction;
use flybrain_core::graph::AgentGraph;
use flybrain_core::task::TaskType;
use flybrain_core::trace::{Totals, TraceStep};
use flybrain_core::verify::VerificationResult;
use flybrain_runtime::bus::MessageBus;
use flybrain_runtime::scheduler::{Scheduler, SchedulerStatus};
use flybrain_runtime::trace_writer::TraceWriter;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

fn parse_action(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<GraphAction> {
    let value = py_to_json(py, obj)?;
    serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_graph(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<AgentGraph> {
    let value = py_to_json(py, obj)?;
    serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_task_type(s: &str) -> PyResult<TaskType> {
    serde_json::from_value(serde_json::Value::String(s.to_string()))
        .map_err(|e| PyValueError::new_err(format!("invalid task_type {s}: {e}")))
}

fn status_to_dict(py: Python<'_>, status: SchedulerStatus) -> PyResult<PyObject> {
    let v = match status {
        SchedulerStatus::RunAgent(name) => serde_json::json!({
            "kind": "run_agent",
            "agent": name,
        }),
        SchedulerStatus::CallVerifier => serde_json::json!({"kind": "call_verifier"}),
        SchedulerStatus::CallMemory => serde_json::json!({"kind": "call_memory"}),
        SchedulerStatus::CallRetriever => serde_json::json!({"kind": "call_retriever"}),
        SchedulerStatus::CallToolExecutor => {
            serde_json::json!({"kind": "call_tool_executor"})
        }
        SchedulerStatus::GraphMutation => serde_json::json!({"kind": "graph_mutation"}),
        SchedulerStatus::Terminated => serde_json::json!({"kind": "terminated"}),
    };
    json_to_py(py, &v)
}

// ---------------------------------------------------------------------- Scheduler

#[pyclass(name = "Scheduler", module = "flybrain.flybrain_native")]
pub(crate) struct PyScheduler {
    inner: Scheduler,
}

#[pymethods]
impl PyScheduler {
    #[new]
    fn new(py: Python<'_>, agent_graph: &Bound<'_, PyAny>) -> PyResult<Self> {
        let g = parse_graph(py, agent_graph)?;
        Ok(Self {
            inner: Scheduler::new(g),
        })
    }

    /// Apply a `GraphAction` dict and return a status dict
    /// `{"kind": "run_agent" | "graph_mutation" | "call_verifier" | ...}`.
    fn apply(&mut self, py: Python<'_>, action: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let a = parse_action(py, action)?;
        let status = self
            .inner
            .apply(&a)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        status_to_dict(py, status)
    }

    /// Bump the step counter; returns the *new* step id.
    fn advance_step(&mut self) -> u64 {
        self.inner.advance_step()
    }

    #[getter]
    fn step_id(&self) -> u64 {
        self.inner.step_id()
    }

    #[getter]
    fn last_active_agent(&self) -> Option<String> {
        self.inner.last_active_agent().map(|s| s.to_string())
    }

    #[getter]
    fn is_terminated(&self) -> bool {
        self.inner.is_terminated()
    }

    #[getter]
    fn current_graph_hash(&self) -> String {
        self.inner.current_graph_hash()
    }

    /// Snapshot the current `AgentGraph` as a JSON-shaped dict.
    fn agent_graph(&self, py: Python<'_>) -> PyResult<PyObject> {
        let v = serde_json::to_value(self.inner.graph())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        json_to_py(py, &v)
    }
}

// ---------------------------------------------------------------------- MessageBus

#[pyclass(name = "MessageBus", module = "flybrain.flybrain_native")]
pub(crate) struct PyMessageBus {
    inner: MessageBus,
}

#[pymethods]
impl PyMessageBus {
    #[new]
    fn new() -> Self {
        Self {
            inner: MessageBus::new(),
        }
    }

    /// Send a message; returns the assigned message id.
    fn send(
        &mut self,
        py: Python<'_>,
        sender: &str,
        recipient: &str,
        content: &Bound<'_, PyAny>,
        step_id: u64,
    ) -> PyResult<u64> {
        let v = py_to_json(py, content)?;
        Ok(self.inner.send(sender, recipient, v, step_id))
    }

    /// Pop the next pending message for `recipient`. Returns `None` when
    /// the inbox is empty.
    fn pop(&mut self, py: Python<'_>, recipient: &str) -> PyResult<Option<PyObject>> {
        match self.inner.pop(recipient) {
            None => Ok(None),
            Some(msg) => {
                let v =
                    serde_json::to_value(&msg).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Some(json_to_py(py, &v)?))
            }
        }
    }

    fn pending(&self, recipient: &str) -> usize {
        self.inner.pending(recipient)
    }

    fn total(&self) -> usize {
        self.inner.total()
    }
}

// ---------------------------------------------------------------------- TraceWriter

#[pyclass(name = "TraceWriter", module = "flybrain.flybrain_native")]
pub(crate) struct PyTraceWriter {
    inner: Option<TraceWriter>,
}

#[pymethods]
impl PyTraceWriter {
    #[new]
    #[pyo3(signature = (task_id, task_type, path = None))]
    fn new(task_id: &str, task_type: &str, path: Option<String>) -> PyResult<Self> {
        let tt = parse_task_type(task_type)?;
        let p = path.map(PathBuf::from);
        let writer = TraceWriter::open(task_id, tt, p.as_deref())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Some(writer),
        })
    }

    /// Append a `TraceStep` dict (matching the Rust `TraceStep` schema).
    fn record_step(&mut self, py: Python<'_>, step: &Bound<'_, PyAny>) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("trace already finalised"))?;
        let v = py_to_json(py, step)?;
        let s: TraceStep =
            serde_json::from_value(v).map_err(|e| PyValueError::new_err(e.to_string()))?;
        writer
            .record_step(s)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Replace the running totals (used by the Python budget tracker).
    fn set_totals(&mut self, py: Python<'_>, totals: &Bound<'_, PyAny>) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("trace already finalised"))?;
        let v = py_to_json(py, totals)?;
        let t: Totals =
            serde_json::from_value(v).map_err(|e| PyValueError::new_err(e.to_string()))?;
        writer.set_totals(t);
        Ok(())
    }

    fn step_count(&self) -> PyResult<usize> {
        let writer = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("trace already finalised"))?;
        Ok(writer.step_count())
    }

    /// Snapshot the running trace as a dict without finalising.
    fn snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let writer = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("trace already finalised"))?;
        let snap = writer.snapshot();
        let v = serde_json::to_value(&snap).map_err(|e| PyValueError::new_err(e.to_string()))?;
        json_to_py(py, &v)
    }

    /// Finalise the trace and return the full `Trace` dict. Closes the
    /// JSONL sink. After this call any further method raises ValueError.
    #[pyo3(signature = (final_answer = None, verification = None, metadata = None))]
    fn finalize(
        &mut self,
        py: Python<'_>,
        final_answer: Option<String>,
        verification: Option<&Bound<'_, PyAny>>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("trace already finalised"))?;
        let v = match verification {
            None => None,
            Some(obj) => {
                let value = py_to_json(py, obj)?;
                let vr: VerificationResult = serde_json::from_value(value)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Some(vr)
            }
        };
        let m = match metadata {
            None => serde_json::Value::Null,
            Some(obj) => py_to_json(py, obj)?,
        };
        let trace = writer
            .finalize(final_answer, v, m)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let value =
            serde_json::to_value(&trace).map_err(|e| PyValueError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }
}

// ---------------------------------------------------------------------- registration

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScheduler>()?;
    m.add_class::<PyMessageBus>()?;
    m.add_class::<PyTraceWriter>()?;
    Ok(())
}

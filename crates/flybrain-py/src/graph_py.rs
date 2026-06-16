//! Phase-1 PyO3 bindings for the graph builder pipeline.
//!
//! Each function takes a Python dict and converts it to the typed Rust
//! structs via `serde_json::Value`. Returns are JSON-serialised dicts so
//! Python can `json.dumps` the result if it wants. We intentionally do *not*
//! expose `#[pyclass]` types yet — those churn during early phases. JSON
//! round-trip is plenty fast for graphs of size ≤ K=256.

use crate::{json_to_py, py_to_json};
use flybrain_core::graph::FlyGraph;
use flybrain_graph::builder::{BuildReport, BuildRequest};
use flybrain_graph::compression::{
    aggregate::aggregate, celltype_agg::celltype_aggregation, leiden::leiden, louvain::louvain,
    modularity, region_agg::region_aggregation, spectral::spectral, ClusterAssignment,
};
use flybrain_graph::format::{load_fbg, save_fbg, save_node_metadata_json};
use flybrain_graph::synthetic::synthetic_fly_graph;
use flybrain_graph::zenodo::{load_zenodo_csv, load_zenodo_dir};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

fn graph_to_py(py: Python<'_>, g: &FlyGraph) -> PyResult<PyObject> {
    let value = serde_json::to_value(g).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &value)
}

fn assignment_to_py(py: Python<'_>, a: &ClusterAssignment, q: f64) -> PyResult<PyObject> {
    let value = serde_json::json!({
        "assignment": a.assignment,
        "num_clusters": a.num_clusters,
        "labels": a.labels,
        "modularity": q,
    });
    json_to_py(py, &value)
}

#[pyfunction]
#[pyo3(signature = (num_nodes, seed=42))]
pub(crate) fn build_synthetic(py: Python<'_>, num_nodes: usize, seed: u64) -> PyResult<PyObject> {
    let g = synthetic_fly_graph(num_nodes, seed);
    graph_to_py(py, &g)
}

#[pyfunction]
pub(crate) fn load_zenodo(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let g = load_zenodo_dir(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    graph_to_py(py, &g)
}

#[pyfunction]
pub(crate) fn load_zenodo_pair(
    py: Python<'_>,
    neurons: &str,
    connections: &str,
) -> PyResult<PyObject> {
    let g = load_zenodo_csv(neurons, connections).map_err(|e| PyIOError::new_err(e.to_string()))?;
    graph_to_py(py, &g)
}

#[pyfunction]
pub(crate) fn save_graph(py: Python<'_>, graph: &Bound<'_, PyAny>, path: &str) -> PyResult<()> {
    let value = py_to_json(py, graph)?;
    let g: FlyGraph =
        serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    save_fbg(&g, path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
pub(crate) fn load_graph(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let g = load_fbg(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    graph_to_py(py, &g)
}

#[pyfunction]
pub(crate) fn write_node_metadata(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    path: &str,
) -> PyResult<()> {
    let value = py_to_json(py, graph)?;
    let g: FlyGraph =
        serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    save_node_metadata_json(&g, path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(())
}

fn run_compression(
    method: &str,
    graph: &FlyGraph,
    target_k: usize,
    seed: u64,
) -> PyResult<ClusterAssignment> {
    match method.to_lowercase().as_str() {
        "region_agg" | "region" => {
            region_aggregation(graph).map_err(|e| PyValueError::new_err(e.to_string()))
        }
        "celltype_agg" | "celltype" | "cell_type" => {
            celltype_aggregation(graph).map_err(|e| PyValueError::new_err(e.to_string()))
        }
        "louvain" => {
            louvain(graph, target_k, seed).map_err(|e| PyValueError::new_err(e.to_string()))
        }
        "leiden" => leiden(graph, target_k, seed).map_err(|e| PyValueError::new_err(e.to_string())),
        "spectral" => {
            spectral(graph, target_k, seed).map_err(|e| PyValueError::new_err(e.to_string()))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown compression method {other}"
        ))),
    }
}

/// Compress a graph dict with the named method. Returns a dict with
/// `assignment`, `num_clusters`, optional `labels`, and the directed
/// modularity `Q`.
#[pyfunction]
#[pyo3(signature = (graph, method, target_k, seed=42))]
pub(crate) fn compress(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    method: &str,
    target_k: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let value = py_to_json(py, graph)?;
    let g: FlyGraph =
        serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = run_compression(method, &g, target_k, seed)?;
    let q = modularity(&g, &a.assignment);
    assignment_to_py(py, &a, q)
}

/// One-shot compress-and-aggregate: returns the K-node compressed FlyGraph
/// dict.
#[pyfunction]
#[pyo3(signature = (graph, method, target_k, seed=42))]
pub(crate) fn compress_and_aggregate(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    method: &str,
    target_k: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let value = py_to_json(py, graph)?;
    let g: FlyGraph =
        serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = run_compression(method, &g, target_k, seed)?;
    let compressed = aggregate(&g, &a, method).map_err(|e| PyValueError::new_err(e.to_string()))?;
    graph_to_py(py, &compressed)
}

/// Full builder pipeline: source spec + method + K + output path.
/// `source_spec` is a dict like:
///   {"kind": "synthetic", "num_nodes": 2048, "seed": 42}
///   {"kind": "zenodo_dir", "dir": "data/flywire/"}
///   {"kind": "zenodo_csv", "neurons": "...", "connections": "..."}
#[pyfunction]
#[pyo3(signature = (source_spec, method, target_k, output, seed=42))]
pub(crate) fn build(
    py: Python<'_>,
    source_spec: &Bound<'_, PyAny>,
    method: &str,
    target_k: usize,
    output: &str,
    seed: u64,
) -> PyResult<PyObject> {
    let source_value = py_to_json(py, source_spec)?;
    let source = serde_json::from_value(source_value)
        .map_err(|e| PyValueError::new_err(format!("bad source spec: {e}")))?;
    let method = flybrain_graph::builder::CompressionMethod::parse(method)
        .map_err(|e| PyValueError::new_err(format!("unknown method {method}: {e}")))?;
    let req = BuildRequest {
        source,
        method,
        target_k,
        seed,
        output: PathBuf::from(output),
    };
    let report: BuildReport =
        flybrain_graph::builder::build(&req).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let value = serde_json::to_value(&report).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &value)
}

/// Compute directed modularity `Q` for a graph + assignment.
#[pyfunction]
pub(crate) fn graph_modularity(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    assignment: Vec<u32>,
) -> PyResult<f64> {
    let value = py_to_json(py, graph)?;
    let g: FlyGraph =
        serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(modularity(&g, &assignment))
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_synthetic, m)?)?;
    m.add_function(wrap_pyfunction!(load_zenodo, m)?)?;
    m.add_function(wrap_pyfunction!(load_zenodo_pair, m)?)?;
    m.add_function(wrap_pyfunction!(save_graph, m)?)?;
    m.add_function(wrap_pyfunction!(load_graph, m)?)?;
    m.add_function(wrap_pyfunction!(write_node_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(compress_and_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(graph_modularity, m)?)?;
    Ok(())
}

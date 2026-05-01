//! Local Zenodo / FlyWire CSV loader.
//!
//! Phase 1 ships a *file-based* loader. Downloading the Zenodo tar is left to
//! `scripts/download_flywire.sh` so we don't pull `reqwest` + native TLS into
//! the workspace just to invoke `curl`. The expected layout is:
//!
//! ```text
//! data/flywire/
//!   neurons.csv      # columns: id, cell_type, region
//!   connections.csv  # columns: pre_root_id, post_root_id, syn_count, [is_excitatory]
//! ```
//!
//! Column names follow the FlyWire / Codex CSV exports. Missing optional
//! columns degrade gracefully (`cell_type` / `region` default to empty
//! strings, `is_excitatory` defaults to `true`).
//!
//! Real research-grade loaders (Codex API, Zenodo tar) would augment this in
//! Phase 12; the contract `(neurons.csv + connections.csv) -> FlyGraph` is the
//! stable interface.

use crate::error::{GraphError, GraphResult};
use flybrain_core::graph::{FlyGraph, NodeMetadata};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct NeuronRow {
    /// Root id from FlyWire materialization.
    id: u64,
    #[serde(default)]
    cell_type: String,
    #[serde(default)]
    region: String,
}

#[derive(Debug, Deserialize)]
struct ConnectionRow {
    pre_root_id: u64,
    post_root_id: u64,
    #[serde(default)]
    syn_count: u32,
    #[serde(default)]
    is_excitatory: Option<bool>,
}

/// Load a FlyWire CSV pair from a directory containing `neurons.csv` and
/// `connections.csv`.
pub fn load_zenodo_dir(dir: impl AsRef<Path>) -> GraphResult<FlyGraph> {
    let dir = dir.as_ref();
    let neurons = dir.join("neurons.csv");
    let connections = dir.join("connections.csv");
    load_zenodo_csv(&neurons, &connections)
}

pub fn load_zenodo_csv(
    neurons_path: impl AsRef<Path>,
    connections_path: impl AsRef<Path>,
) -> GraphResult<FlyGraph> {
    // Pass 1: read neurons, assign dense local ids.
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(neurons_path.as_ref())?;
    let mut id_map: HashMap<u64, u32> = HashMap::new();
    let mut nodes: Vec<NodeMetadata> = Vec::new();
    for (i, rec) in rdr.deserialize::<NeuronRow>().enumerate() {
        let row = rec?;
        let local: u32 = i
            .try_into()
            .map_err(|_| GraphError::Zenodo("neuron count exceeds u32::MAX".to_string()))?;
        id_map.insert(row.id, local);
        nodes.push(NodeMetadata {
            id: local,
            node_type: row.cell_type,
            region: row.region,
            features: Vec::new(),
        });
    }
    if nodes.is_empty() {
        return Err(GraphError::Zenodo("neurons.csv is empty".to_string()));
    }

    // Pass 2: read connections, drop ones whose endpoints aren't in the
    // neurons table (FlyWire snapshots sometimes have orphan rows).
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(connections_path.as_ref())?;
    let mut edge_index = Vec::new();
    let mut edge_weight = Vec::new();
    let mut is_excitatory = Vec::new();
    let mut dropped = 0u64;
    for rec in rdr.deserialize::<ConnectionRow>() {
        let row = rec?;
        let (Some(&src), Some(&dst)) =
            (id_map.get(&row.pre_root_id), id_map.get(&row.post_root_id))
        else {
            dropped += 1;
            continue;
        };
        edge_index.push((src, dst));
        edge_weight.push(row.syn_count.max(1) as f32);
        is_excitatory.push(row.is_excitatory.unwrap_or(true));
    }

    let num_nodes = nodes.len();
    let mut provenance = BTreeMap::new();
    provenance.insert("source".into(), serde_json::json!("zenodo_csv"));
    provenance.insert(
        "neurons_path".into(),
        serde_json::json!(neurons_path.as_ref().to_string_lossy()),
    );
    provenance.insert(
        "connections_path".into(),
        serde_json::json!(connections_path.as_ref().to_string_lossy()),
    );
    provenance.insert("dropped_orphans".into(), serde_json::json!(dropped));

    Ok(FlyGraph {
        num_nodes,
        edge_index,
        edge_weight,
        is_excitatory,
        nodes,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn parses_minimal_csv_pair() {
        let n = write_temp(
            "fb_neurons.csv",
            "id,cell_type,region\n100,KC,mushroom_body\n101,MBON,mushroom_body\n102,DAN,central_complex\n",
        );
        let c = write_temp(
            "fb_connections.csv",
            "pre_root_id,post_root_id,syn_count,is_excitatory\n100,101,5,true\n101,102,2,false\n999,100,3,true\n",
        );
        let g = load_zenodo_csv(&n, &c).unwrap();
        assert_eq!(g.num_nodes, 3);
        assert_eq!(g.num_edges(), 2);
        // Orphan edge 999->100 should be dropped, recorded in provenance.
        assert_eq!(
            g.provenance.get("dropped_orphans").unwrap(),
            &serde_json::json!(1u64)
        );
        assert_eq!(g.nodes[0].region, "mushroom_body");
        let _ = std::fs::remove_file(&n);
        let _ = std::fs::remove_file(&c);
    }

    #[test]
    fn empty_neurons_errors() {
        let n = write_temp("fb_empty_neurons.csv", "id,cell_type,region\n");
        let c = write_temp("fb_empty_conn.csv", "pre_root_id,post_root_id,syn_count\n");
        let r = load_zenodo_csv(&n, &c);
        assert!(matches!(r, Err(GraphError::Zenodo(_))));
        let _ = std::fs::remove_file(&n);
        let _ = std::fs::remove_file(&c);
    }
}

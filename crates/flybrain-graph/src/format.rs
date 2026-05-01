//! `.fbg` file format: gzip-wrapped JSON of `FlyGraph`.
//!
//! Layout: `MAGIC(4) || gzip(json(FlyGraph))`. Gzip is self-describing so
//! `gunzip <file>.fbg | jq` works after dropping the magic bytes.
//!
//! We deliberately stay with JSON instead of bincode because `FlyGraph`'s
//! `provenance` field is a `BTreeMap<String, serde_json::Value>` —
//! bincode is non-self-describing and refuses to deserialize untyped JSON
//! values. A 256-node compressed graph round-trips at ~30 KB compressed,
//! which is fine for our use case.
//!
//! Companion file `<name>.node_metadata.json` is written alongside `.fbg`
//! so notebooks / Python tooling can read node info without decompressing.

use crate::error::{GraphError, GraphResult};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use flybrain_core::graph::FlyGraph;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8] = b"FBG1";

pub fn save_fbg(graph: &FlyGraph, path: impl AsRef<Path>) -> GraphResult<()> {
    let mut f = BufWriter::new(File::create(path.as_ref())?);
    f.write_all(MAGIC)?;
    let mut enc = GzEncoder::new(f, Compression::default());
    serde_json::to_writer(&mut enc, graph).map_err(|e| GraphError::Invalid(e.to_string()))?;
    enc.finish()?.flush()?;
    Ok(())
}

pub fn load_fbg(path: impl AsRef<Path>) -> GraphResult<FlyGraph> {
    let mut f = BufReader::new(File::open(path.as_ref())?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(GraphError::Invalid(format!(
            "bad magic: expected {:?}, got {:?}",
            MAGIC, magic
        )));
    }
    let dec = GzDecoder::new(f);
    let g: FlyGraph =
        serde_json::from_reader(dec).map_err(|e| GraphError::Invalid(e.to_string()))?;
    Ok(g)
}

pub fn save_node_metadata_json(graph: &FlyGraph, path: impl AsRef<Path>) -> GraphResult<()> {
    let value = serde_json::json!({
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges(),
        "nodes": graph.nodes,
        "provenance": graph.provenance,
    });
    let s = serde_json::to_string_pretty(&value).map_err(|e| GraphError::Invalid(e.to_string()))?;
    std::fs::write(path, s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::synthetic_fly_graph;

    #[test]
    fn round_trip() {
        let g = synthetic_fly_graph(64, 42);
        let tmp = std::env::temp_dir().join("fbg_round_trip.fbg");
        save_fbg(&g, &tmp).unwrap();
        let g2 = load_fbg(&tmp).unwrap();
        assert_eq!(g.num_nodes, g2.num_nodes);
        assert_eq!(g.edge_index, g2.edge_index);
        assert_eq!(g.edge_weight, g2.edge_weight);
        assert_eq!(g.is_excitatory, g2.is_excitatory);
        assert_eq!(g.provenance, g2.provenance);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn metadata_json() {
        let g = synthetic_fly_graph(16, 1);
        let tmp = std::env::temp_dir().join("fbg_meta.json");
        save_node_metadata_json(&g, &tmp).unwrap();
        let s = std::fs::read_to_string(&tmp).unwrap();
        assert!(s.contains("num_nodes"));
        assert!(s.contains("provenance"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn rejects_bad_magic() {
        let path = std::env::temp_dir().join("fbg_bad_magic.fbg");
        std::fs::write(&path, b"NOPE").unwrap();
        let r = load_fbg(&path);
        assert!(matches!(
            r,
            Err(GraphError::Invalid(_)) | Err(GraphError::Io(_))
        ));
        let _ = std::fs::remove_file(&path);
    }
}

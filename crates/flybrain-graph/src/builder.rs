//! Builder orchestrator: combines a `BuildSource` (synthetic / Zenodo CSV)
//! with a `CompressionMethod` and writes a `.fbg` (+ companion JSON) to disk.

use crate::compression::{
    aggregate::aggregate, celltype_agg::celltype_aggregation, leiden::leiden, louvain::louvain,
    region_agg::region_aggregation, spectral::spectral,
};
use crate::error::{GraphError, GraphResult};
use crate::format::{save_fbg, save_node_metadata_json};
use crate::synthetic::synthetic_fly_graph;
use crate::zenodo::{load_zenodo_csv, load_zenodo_dir};
use flybrain_core::graph::FlyGraph;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BuildSource {
    /// Synthetic fly-inspired graph generator. Use as default for CI and
    /// when no Zenodo snapshot is available.
    Synthetic { num_nodes: usize, seed: u64 },
    /// Pre-downloaded Zenodo / FlyWire CSV directory containing
    /// `neurons.csv` and `connections.csv`.
    ZenodoDir { dir: PathBuf },
    /// Explicit paths to the two CSV files.
    ZenodoCsv {
        neurons: PathBuf,
        connections: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompressionMethod {
    RegionAgg,
    CelltypeAgg,
    Louvain,
    Leiden,
    Spectral,
}

impl CompressionMethod {
    pub fn parse(name: &str) -> GraphResult<Self> {
        match name.to_lowercase().as_str() {
            "region_agg" | "region" => Ok(Self::RegionAgg),
            "celltype_agg" | "celltype" | "cell_type" => Ok(Self::CelltypeAgg),
            "louvain" => Ok(Self::Louvain),
            "leiden" => Ok(Self::Leiden),
            "spectral" => Ok(Self::Spectral),
            _ => Err(GraphError::UnknownMethod(name.to_string())),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RegionAgg => "region_agg",
            Self::CelltypeAgg => "celltype_agg",
            Self::Louvain => "louvain",
            Self::Leiden => "leiden",
            Self::Spectral => "spectral",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildRequest {
    pub source: BuildSource,
    pub method: CompressionMethod,
    pub target_k: usize,
    pub seed: u64,
    /// Output `.fbg` file path. Companion JSON is written next to it.
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildReport {
    pub source_num_nodes: usize,
    pub source_num_edges: usize,
    pub compressed_num_nodes: usize,
    pub compressed_num_edges: usize,
    pub method: String,
    pub target_k: usize,
    pub fbg_path: PathBuf,
    pub metadata_json_path: PathBuf,
    pub modularity_directed: f64,
}

pub fn build(req: &BuildRequest) -> GraphResult<BuildReport> {
    let source = load_source(&req.source)?;
    let target_k = req.target_k;

    let assignment = match req.method {
        CompressionMethod::RegionAgg => region_aggregation(&source)?,
        CompressionMethod::CelltypeAgg => celltype_aggregation(&source)?,
        CompressionMethod::Louvain => louvain(&source, target_k, req.seed)?,
        CompressionMethod::Leiden => leiden(&source, target_k, req.seed)?,
        CompressionMethod::Spectral => spectral(&source, target_k, req.seed)?,
    };

    let q = crate::compression::modularity(&source, &assignment.assignment);
    let compressed = aggregate(&source, &assignment, req.method.as_str())?;

    if let Some(parent) = req.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    save_fbg(&compressed, &req.output)?;
    let meta_path = with_suffix(&req.output, ".node_metadata.json");
    save_node_metadata_json(&compressed, &meta_path)?;

    Ok(BuildReport {
        source_num_nodes: source.num_nodes,
        source_num_edges: source.num_edges(),
        compressed_num_nodes: compressed.num_nodes,
        compressed_num_edges: compressed.num_edges(),
        method: req.method.as_str().into(),
        target_k,
        fbg_path: req.output.clone(),
        metadata_json_path: meta_path,
        modularity_directed: q,
    })
}

/// Convenience: produce a synthetic-source + Louvain build for the four
/// canonical K values used in the controller (32 / 64 / 128 / 256).
pub fn build_default(out_dir: impl Into<PathBuf>) -> GraphResult<Vec<BuildReport>> {
    build_default_with(out_dir, 2048, 42, CompressionMethod::Louvain)
}

/// Same as `build_default` but with explicit source size + seed + method.
pub fn build_default_with(
    out_dir: impl Into<PathBuf>,
    source_num_nodes: usize,
    seed: u64,
    method: CompressionMethod,
) -> GraphResult<Vec<BuildReport>> {
    let out_dir: PathBuf = out_dir.into();
    let mut reports = Vec::new();
    for k in [32usize, 64, 128, 256] {
        let req = BuildRequest {
            source: BuildSource::Synthetic {
                num_nodes: source_num_nodes,
                seed,
            },
            method,
            target_k: k,
            seed,
            output: out_dir.join(format!("fly_graph_{}.fbg", k)),
        };
        reports.push(build(&req)?);
    }
    Ok(reports)
}

fn load_source(src: &BuildSource) -> GraphResult<FlyGraph> {
    match src {
        BuildSource::Synthetic { num_nodes, seed } => Ok(synthetic_fly_graph(*num_nodes, *seed)),
        BuildSource::ZenodoDir { dir } => load_zenodo_dir(dir),
        BuildSource::ZenodoCsv {
            neurons,
            connections,
        } => load_zenodo_csv(neurons, connections),
    }
}

fn with_suffix(path: &std::path::Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn synthetic_louvain_round_trip() {
        let dir = tmp_dir("flybrain_build_louvain");
        let req = BuildRequest {
            source: BuildSource::Synthetic {
                num_nodes: 256,
                seed: 1,
            },
            method: CompressionMethod::Louvain,
            target_k: 16,
            seed: 1,
            output: dir.join("fly_graph_16.fbg"),
        };
        let r = build(&req).unwrap();
        assert_eq!(r.compressed_num_nodes, 16);
        assert_eq!(r.method, "louvain");
        assert!(r.fbg_path.exists());
        assert!(r.metadata_json_path.exists());

        let g = crate::format::load_fbg(&r.fbg_path).unwrap();
        assert_eq!(g.num_nodes, 16);
        assert!(g.provenance.contains_key("compression"));
        assert_eq!(g.provenance.get("K").unwrap(), &serde_json::json!(16));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_default_writes_four_files() {
        let dir = tmp_dir("flybrain_build_default");
        let reports = build_default(&dir).unwrap();
        assert_eq!(reports.len(), 4);
        for k in [32usize, 64, 128, 256] {
            assert!(dir.join(format!("fly_graph_{}.fbg", k)).exists());
            assert!(dir
                .join(format!("fly_graph_{}.fbg.node_metadata.json", k))
                .exists());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_method() {
        assert_eq!(
            CompressionMethod::parse("louvain").unwrap(),
            CompressionMethod::Louvain
        );
        assert_eq!(
            CompressionMethod::parse("CelltypeAgg")
                .err()
                .map(|e| e.to_string()),
            None.or(Some("unknown compression method: CelltypeAgg".to_string()))
        );
        // case-insensitive snake_case
        assert!(CompressionMethod::parse("celltype_agg").is_ok());
        assert!(CompressionMethod::parse("REGION").is_ok());
    }
}

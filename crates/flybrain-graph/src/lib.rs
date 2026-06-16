//! FlyWire connectome loaders and graph compression for FlyBrain Optimizer.
//!
//! Phase 1 ships:
//! - `synthetic`: deterministic fly-inspired generator (used as default source).
//! - `zenodo`: parse a Zenodo FlyWire CSV snapshot from a local path.
//! - `compression::{region_agg, celltype_agg, louvain, leiden, spectral}`:
//!   reduce a large graph (139k+ nodes) to a small K-node prior for the
//!   controller.
//! - `format`: compact bincode `.fbg` file format (gzip-wrapped) with a JSON
//!   sibling for `node_metadata.json`.
//! - `builder`: orchestrator that wires source → compression → file output.
//!
//! All compression methods are deterministic for a given (`graph`, `seed`,
//! `K`) tuple — that's required for the Phase 1 stability tests.

pub mod builder;
pub mod compression;
pub mod error;
pub mod format;
pub mod synthetic;
pub mod zenodo;

pub use builder::{build, build_default, BuildRequest, BuildSource, CompressionMethod};
pub use error::{GraphError, GraphResult};
pub use format::{load_fbg, save_fbg};
pub use synthetic::synthetic_fly_graph;

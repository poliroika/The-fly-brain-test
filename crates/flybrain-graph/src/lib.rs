//! FlyWire connectome loaders and graph compression.
//!
//! Phase 0 only ships a synthetic generator and the empty placeholders; real
//! Zenodo / Codex loaders and the Louvain / Leiden / spectral compressors land
//! in Phase 1.

pub mod synthetic;

pub use synthetic::synthetic_fly_graph;

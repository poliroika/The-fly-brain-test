//! Graph compression algorithms.
//!
//! Each method takes a `FlyGraph` and a target cluster count `K` and returns a
//! cluster assignment `node_id -> cluster_id` (`Vec<u32>` of length
//! `num_nodes`). The orchestrator (`builder.rs`) then aggregates the original
//! graph into a K-node compressed graph using `aggregate::aggregate`.
//!
//! All methods are deterministic for a given (`graph`, `seed`, `K`) tuple —
//! that's required by the Phase 1 stability tests.

pub mod aggregate;
pub mod celltype_agg;
pub mod leiden;
pub mod louvain;
pub mod region_agg;
pub mod spectral;

use crate::error::GraphResult;
use flybrain_core::graph::FlyGraph;

/// Result of a compression pass.
#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    /// `assignment[i]` = cluster id of original node `i`. `0..num_clusters`.
    pub assignment: Vec<u32>,
    pub num_clusters: usize,
    /// Optional human-readable label per cluster (e.g. region or cell type).
    pub labels: Option<Vec<String>>,
}

impl ClusterAssignment {
    pub fn check_valid(&self, expected_num_nodes: usize) -> GraphResult<()> {
        if self.assignment.len() != expected_num_nodes {
            return Err(crate::error::GraphError::Invalid(format!(
                "assignment length {} != num_nodes {}",
                self.assignment.len(),
                expected_num_nodes
            )));
        }
        for &c in &self.assignment {
            if (c as usize) >= self.num_clusters {
                return Err(crate::error::GraphError::Invalid(format!(
                    "cluster id {} >= num_clusters {}",
                    c, self.num_clusters
                )));
            }
        }
        Ok(())
    }
}

/// Compute *directed* modularity for a partition. Uses Leicht / Newman
/// (2008) formulation, ignoring edge sign:
///
/// `Q = (1 / m) Σ_c [ W_c − (k^{out}_c · k^{in}_c) / m ]`
///
/// where `W_c` is the total weight of intra-cluster edges, `k^{out}_c` and
/// `k^{in}_c` are the cluster's total out- and in-degree, and `m` is the
/// total edge weight. This is the directed analogue of `Q` for undirected
/// modularity, with `m` (not `2m`) in the normaliser.
pub fn modularity(graph: &FlyGraph, assignment: &[u32]) -> f64 {
    if graph.edge_index.is_empty() {
        return 0.0;
    }
    let n = graph.num_nodes;
    if assignment.len() != n {
        return 0.0;
    }

    let mut k_out = vec![0.0f64; n];
    let mut k_in = vec![0.0f64; n];
    let mut m = 0.0f64;
    for ((src, dst), &w) in graph.edge_index.iter().zip(graph.edge_weight.iter()) {
        let w = w as f64;
        k_out[*src as usize] += w;
        k_in[*dst as usize] += w;
        m += w;
    }
    if m == 0.0 {
        return 0.0;
    }

    let num_clusters = (assignment.iter().copied().max().unwrap_or(0) as usize) + 1;
    let mut intra_weight = vec![0.0f64; num_clusters];
    let mut sum_k_out = vec![0.0f64; num_clusters];
    let mut sum_k_in = vec![0.0f64; num_clusters];

    for (i, &c) in assignment.iter().enumerate() {
        let c = c as usize;
        sum_k_out[c] += k_out[i];
        sum_k_in[c] += k_in[i];
    }
    for ((src, dst), &w) in graph.edge_index.iter().zip(graph.edge_weight.iter()) {
        let cs = assignment[*src as usize] as usize;
        let cd = assignment[*dst as usize] as usize;
        if cs == cd {
            intra_weight[cs] += w as f64;
        }
    }
    let mut q = 0.0f64;
    for c in 0..num_clusters {
        q += intra_weight[c] - sum_k_out[c] * sum_k_in[c] / m;
    }
    q / m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::synthetic_fly_graph;

    #[test]
    fn modularity_singleton_partition_is_negative_or_zero() {
        // Every node in its own cluster — no intra-cluster edges, so
        // Q = -Σ k_out_i * k_in_i / m^2, which is non-positive.
        let g = synthetic_fly_graph(32, 7);
        let a: Vec<u32> = (0..g.num_nodes as u32).collect();
        let q = modularity(&g, &a);
        assert!(q <= 1e-9, "expected Q <= 0, got {}", q);
    }

    #[test]
    fn modularity_one_big_cluster_is_zero() {
        // One cluster — Σ A_ij = m, Σ expected = m → Q = 0.
        let g = synthetic_fly_graph(32, 7);
        let a = vec![0u32; g.num_nodes];
        let q = modularity(&g, &a);
        assert!(q.abs() < 1e-6, "expected Q ≈ 0, got {}", q);
    }
}

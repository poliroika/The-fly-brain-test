//! Aggregate an original graph into a compressed K-node graph using a
//! cluster assignment.

use super::ClusterAssignment;
use crate::error::GraphResult;
use flybrain_core::graph::{FlyGraph, NodeMetadata};
use std::collections::BTreeMap;

/// Build a K-node graph by collapsing each cluster into a super-node.
///
/// - Cluster super-edges are weighted by the sum of original edge weights.
/// - `is_excitatory` is set per super-edge by majority vote (excitatory if
///   ≥50% of the underlying edges are excitatory).
/// - Each super-node's `node_type` and `region` are filled with the modal
///   value of the cluster's members. If `assignment.labels` is provided, it
///   takes precedence for `region`.
///
/// `provenance` from the source graph is preserved and augmented with the
/// compression method + cluster sizes.
pub fn aggregate(
    src: &FlyGraph,
    assignment: &ClusterAssignment,
    method_label: &str,
) -> GraphResult<FlyGraph> {
    assignment.check_valid(src.num_nodes)?;
    let k = assignment.num_clusters;

    // Cluster sizes + modal cell type / region.
    let mut sizes = vec![0usize; k];
    let mut type_counts: Vec<BTreeMap<String, u32>> = vec![BTreeMap::new(); k];
    let mut region_counts: Vec<BTreeMap<String, u32>> = vec![BTreeMap::new(); k];
    for (i, c) in assignment.assignment.iter().enumerate() {
        let c = *c as usize;
        sizes[c] += 1;
        if let Some(node) = src.nodes.get(i) {
            *type_counts[c].entry(node.node_type.clone()).or_insert(0) += 1;
            *region_counts[c].entry(node.region.clone()).or_insert(0) += 1;
        }
    }

    let modal = |counts: &BTreeMap<String, u32>| -> String {
        counts
            .iter()
            .max_by_key(|(label, count)| (*count, std::cmp::Reverse((*label).clone())))
            .map(|(k, _)| k.clone())
            .unwrap_or_default()
    };

    let mut nodes = Vec::with_capacity(k);
    for c in 0..k {
        let region = if let Some(labels) = &assignment.labels {
            labels
                .get(c)
                .cloned()
                .unwrap_or_else(|| modal(&region_counts[c]))
        } else {
            modal(&region_counts[c])
        };
        nodes.push(NodeMetadata {
            id: c as u32,
            node_type: modal(&type_counts[c]),
            region,
            features: vec![sizes[c] as f32],
        });
    }

    // Aggregate edges. Using BTreeMap keeps determinism.
    let mut agg_w: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    let mut agg_exc: BTreeMap<(u32, u32), (u64, u64)> = BTreeMap::new(); // (excitatory, total)

    let assign = &assignment.assignment;
    for (i, ((src_node, dst_node), &w)) in src
        .edge_index
        .iter()
        .zip(src.edge_weight.iter())
        .enumerate()
    {
        let cs = assign[*src_node as usize];
        let cd = assign[*dst_node as usize];
        if cs == cd {
            // Intra-cluster edges become self-loops on the super-node.
            // Keep them — they represent within-region recurrence which is
            // structurally meaningful for the controller.
        }
        *agg_w.entry((cs, cd)).or_insert(0.0) += w as f64;
        let entry = agg_exc.entry((cs, cd)).or_insert((0, 0));
        entry.1 += 1;
        if let Some(true) = src.is_excitatory.get(i).copied() {
            entry.0 += 1;
        } else if src.is_excitatory.is_empty() {
            // Default: excitatory.
            entry.0 += 1;
        }
    }

    let mut edge_index = Vec::with_capacity(agg_w.len());
    let mut edge_weight = Vec::with_capacity(agg_w.len());
    let mut is_excitatory = Vec::with_capacity(agg_w.len());
    for ((s, d), w) in agg_w.iter() {
        edge_index.push((*s, *d));
        edge_weight.push(*w as f32);
        let (exc, total) = agg_exc.get(&(*s, *d)).copied().unwrap_or((0, 0));
        is_excitatory.push(total > 0 && (2 * exc) >= total);
    }

    let mut provenance = src.provenance.clone();
    provenance.insert("compression".into(), serde_json::json!(method_label));
    provenance.insert("cluster_sizes".into(), serde_json::json!(sizes));
    provenance.insert("K".into(), serde_json::json!(k));

    Ok(FlyGraph {
        num_nodes: k,
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
    use crate::synthetic::synthetic_fly_graph;

    #[test]
    fn aggregate_to_one_cluster_keeps_total_weight() {
        let g = synthetic_fly_graph(32, 7);
        let total: f64 = g.edge_weight.iter().map(|&w| w as f64).sum();
        let assn = ClusterAssignment {
            assignment: vec![0u32; g.num_nodes],
            num_clusters: 1,
            labels: None,
        };
        let c = aggregate(&g, &assn, "test").unwrap();
        assert_eq!(c.num_nodes, 1);
        let new_total: f64 = c.edge_weight.iter().map(|&w| w as f64).sum();
        assert!(
            (total - new_total).abs() < 1e-3,
            "{} vs {}",
            total,
            new_total
        );
    }

    #[test]
    fn aggregate_each_in_own_cluster_preserves_node_count() {
        let g = synthetic_fly_graph(16, 1);
        let assn = ClusterAssignment {
            assignment: (0..g.num_nodes as u32).collect(),
            num_clusters: g.num_nodes,
            labels: None,
        };
        let c = aggregate(&g, &assn, "identity").unwrap();
        assert_eq!(c.num_nodes, g.num_nodes);
    }
}

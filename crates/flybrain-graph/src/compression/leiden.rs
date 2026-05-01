//! Leiden community detection.
//!
//! Phase 1 ships a *minimal* Leiden: we run Louvain to get a base partition,
//! then perform a single refinement pass that splits each community into
//! well-connected sub-components. This addresses the main pathology Leiden
//! was designed to fix (Louvain producing internally disconnected
//! communities) without pulling in a full Leiden implementation, which is
//! ~600 LoC of careful refinement logic.
//!
//! Refinement: for each Louvain community, build the induced subgraph of
//! that community and run a connected-components labelling. Each component
//! becomes its own cluster. Then apply `clamp_to_k` to hit the target K.
//!
//! This is enough for Phase 1 stability tests; a full Leiden lands in
//! Phase 12 if profiling shows community quality matters for the
//! controller.

use super::louvain::louvain;
use super::ClusterAssignment;
use crate::compression::louvain;
use crate::error::GraphResult;
use flybrain_core::graph::FlyGraph;
use std::collections::BTreeMap;

pub fn leiden(graph: &FlyGraph, target_k: usize, seed: u64) -> GraphResult<ClusterAssignment> {
    let base = louvain(graph, target_k.max(1), seed)?;
    let refined = refine(graph, &base.assignment);
    let clamped = louvain::clamp_to_k(graph, &refined, target_k);
    let mut clamped = clamped;
    let num_clusters = renumber(&mut clamped);
    Ok(ClusterAssignment {
        assignment: clamped,
        num_clusters,
        labels: None,
    })
}

fn renumber(a: &mut [u32]) -> usize {
    let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
    for c in a.iter_mut() {
        let next = remap.len() as u32;
        let id = *remap.entry(*c).or_insert(next);
        *c = id;
    }
    remap.len()
}

/// Split each community into its connected components in the *induced*
/// subgraph (treating the graph as undirected for connectivity purposes).
fn refine(graph: &FlyGraph, base: &[u32]) -> Vec<u32> {
    let n = graph.num_nodes;
    // Per-community membership.
    let max_c = base.iter().copied().max().unwrap_or(0) as usize;
    let mut by_comm: Vec<Vec<usize>> = vec![Vec::new(); max_c + 1];
    for (i, &c) in base.iter().enumerate() {
        by_comm[c as usize].push(i);
    }

    // Adjacency restricted to within-community edges (undirected).
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (s, d) in &graph.edge_index {
        let s = *s as usize;
        let d = *d as usize;
        if s == d {
            continue;
        }
        if base.get(s) == base.get(d) {
            adj[s].push(d);
            adj[d].push(s);
        }
    }

    // Connected components via BFS.
    let mut new_label = vec![u32::MAX; n];
    let mut next_id: u32 = 0;
    for members in &by_comm {
        // Each community gets its own connected-component scan.
        for &start in members {
            if new_label[start] != u32::MAX {
                continue;
            }
            let cid = next_id;
            next_id += 1;
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                if new_label[v] != u32::MAX {
                    continue;
                }
                new_label[v] = cid;
                for &u in &adj[v] {
                    if new_label[u] == u32::MAX {
                        stack.push(u);
                    }
                }
            }
        }
    }
    // Any leftover nodes (no edges in their community at all) — assign their
    // own labels deterministically.
    let _ = n;
    for label in new_label.iter_mut() {
        if *label == u32::MAX {
            *label = next_id;
            next_id += 1;
        }
    }
    new_label
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::synthetic_fly_graph;

    #[test]
    fn produces_exact_k() {
        let g = synthetic_fly_graph(128, 7);
        for k in [4usize, 8, 16, 32] {
            let a = leiden(&g, k, 42).unwrap();
            assert_eq!(a.num_clusters, k);
            a.check_valid(g.num_nodes).unwrap();
        }
    }

    #[test]
    fn deterministic() {
        let g = synthetic_fly_graph(64, 1);
        let a = leiden(&g, 8, 42).unwrap();
        let b = leiden(&g, 8, 42).unwrap();
        assert_eq!(a.assignment, b.assignment);
    }
}

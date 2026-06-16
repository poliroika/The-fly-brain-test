//! Modularity-maximising Louvain community detection.
//!
//! Implementation follows Blondel et al. (2008). We use the *undirected*
//! formulation by symmetrising the graph (an edge `(s, d, w)` contributes `w`
//! to both `(s, d)` and `(d, s)` in the working adjacency); the directed
//! modularity used for evaluation in `mod.rs` still operates on the original
//! directed graph.
//!
//! Outer loop:
//!   1. Each node starts in its own community.
//!   2. Iterate over nodes in deterministic order; move each to the
//!      neighbouring community that maximises modularity gain.
//!   3. Repeat until no moves improve modularity.
//!   4. Build a meta-graph from the communities and recurse.
//!
//! Recursion stops once the number of communities is `<= target_k` (or when
//! a level produces no merges).
//!
//! After the algorithm converges we *renumber* communities so they are
//! `0..num_clusters` and (if too few) split the largest communities until we
//! hit `target_k` exactly. If too many, we merge smallest into nearest by
//! edge weight. This guarantees the returned `num_clusters == target_k`,
//! which the rest of the pipeline relies on.

use super::ClusterAssignment;
use crate::error::{GraphError, GraphResult};
use flybrain_core::graph::FlyGraph;
use std::collections::{BTreeMap, BTreeSet};

const MAX_OUTER_PASSES: usize = 8;
const MIN_GAIN: f64 = 1e-7;

/// Run Louvain and force the result to exactly `target_k` clusters.
pub fn louvain(graph: &FlyGraph, target_k: usize, seed: u64) -> GraphResult<ClusterAssignment> {
    if graph.num_nodes == 0 {
        return Err(GraphError::Invalid("empty graph".into()));
    }
    if target_k == 0 {
        return Err(GraphError::Invalid("target_k must be > 0".into()));
    }
    if target_k > graph.num_nodes {
        return Err(GraphError::KTooLarge {
            requested: target_k,
            available: graph.num_nodes,
        });
    }

    let mut working = build_undirected(graph);
    // node_to_community at the *current* level.
    let mut node_community: Vec<usize> = (0..working.num_nodes).collect();
    // mapping from level-0 node -> current super-node id.
    let mut leaf_to_super: Vec<usize> = (0..graph.num_nodes).collect();

    for _outer in 0..MAX_OUTER_PASSES {
        let moved = local_optimisation(&working, &mut node_community, seed);
        let num_communities = renumber(&mut node_community);
        if !moved || num_communities <= target_k.max(1) {
            // Propagate community ids back to the leaf level and stop.
            for super_id in leaf_to_super.iter_mut() {
                *super_id = node_community[*super_id];
            }
            let final_assignment: Vec<u32> = leaf_to_super.iter().map(|&c| c as u32).collect();
            let mut clamped = clamp_to_k(graph, &final_assignment, target_k);
            renumber_assignment(&mut clamped);
            let num_clusters = (clamped.iter().copied().max().unwrap_or(0) as usize) + 1;
            return Ok(ClusterAssignment {
                assignment: clamped,
                num_clusters,
                labels: None,
            });
        }
        // Build the meta-graph for the next pass.
        working = build_meta(&working, &node_community);
        // Update leaf_to_super to point to the new super-ids.
        for super_id in leaf_to_super.iter_mut() {
            *super_id = node_community[*super_id];
        }
        node_community = (0..working.num_nodes).collect();
    }

    // Hit the iteration cap — return whatever we have.
    let final_assignment: Vec<u32> = leaf_to_super.iter().map(|&c| c as u32).collect();
    let mut clamped = clamp_to_k(graph, &final_assignment, target_k);
    renumber_assignment(&mut clamped);
    let num_clusters = (clamped.iter().copied().max().unwrap_or(0) as usize) + 1;
    Ok(ClusterAssignment {
        assignment: clamped,
        num_clusters,
        labels: None,
    })
}

#[derive(Debug, Clone)]
struct UndirectedGraph {
    num_nodes: usize,
    /// For each node, list of (neighbour, weight). Self-loops counted once.
    adj: Vec<Vec<(usize, f64)>>,
    /// Total degree per node (sum of weights, with self-loops counted twice).
    k: Vec<f64>,
    /// Sum over all (i, j) of A_ij. Self-loops contribute their weight once
    /// since they appear once in adj[i] and we double-count below.
    two_m: f64,
}

fn build_undirected(graph: &FlyGraph) -> UndirectedGraph {
    let n = graph.num_nodes;
    let mut adj_w: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for ((s, d), &w) in graph.edge_index.iter().zip(graph.edge_weight.iter()) {
        let s = *s as usize;
        let d = *d as usize;
        if s >= n || d >= n {
            continue;
        }
        let w = w as f64;
        if s == d {
            *adj_w.entry((s, d)).or_insert(0.0) += w;
        } else {
            *adj_w.entry((s.min(d), s.max(d))).or_insert(0.0) += w;
        }
    }
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut k = vec![0.0f64; n];
    let mut two_m = 0.0f64;
    for ((a, b), w) in adj_w {
        if a == b {
            adj[a].push((b, w));
            k[a] += 2.0 * w;
            two_m += 2.0 * w;
        } else {
            adj[a].push((b, w));
            adj[b].push((a, w));
            k[a] += w;
            k[b] += w;
            two_m += 2.0 * w;
        }
    }
    UndirectedGraph {
        num_nodes: n,
        adj,
        k,
        two_m,
    }
}

fn local_optimisation(g: &UndirectedGraph, comm: &mut [usize], seed: u64) -> bool {
    if g.two_m == 0.0 {
        return false;
    }
    // Sigma_tot[c] = sum of degrees of nodes in community c.
    let mut sigma_tot = vec![0.0f64; g.num_nodes];
    for (i, &c) in comm.iter().enumerate() {
        sigma_tot[c] += g.k[i];
    }
    let two_m = g.two_m;
    let mut order: Vec<usize> = (0..g.num_nodes).collect();
    deterministic_shuffle(&mut order, seed);

    let mut moved_any = false;
    let mut improved = true;
    let mut passes = 0;
    while improved && passes < 16 {
        improved = false;
        passes += 1;
        for &i in &order {
            let ci = comm[i];
            let ki = g.k[i];

            // Sum of edge weights to each neighbour community.
            let mut k_to_comm: BTreeMap<usize, f64> = BTreeMap::new();
            let mut k_self_loop = 0.0f64;
            for &(j, w) in &g.adj[i] {
                if j == i {
                    k_self_loop += w;
                    continue;
                }
                *k_to_comm.entry(comm[j]).or_insert(0.0) += w;
            }
            // Remove i from its current community.
            sigma_tot[ci] -= ki;
            let k_in_old = *k_to_comm.get(&ci).unwrap_or(&0.0);

            let mut best_c = ci;
            let mut best_gain = 0.0f64;
            for (&c, &k_in_c) in &k_to_comm {
                let gain = k_in_c - sigma_tot[c] * ki / two_m;
                if gain > best_gain + MIN_GAIN || (gain > best_gain - MIN_GAIN && c < best_c) {
                    best_gain = gain;
                    best_c = c;
                }
            }
            // Always reconsider staying in `ci`.
            let stay_gain = k_in_old - sigma_tot[ci] * ki / two_m;
            if stay_gain > best_gain + MIN_GAIN {
                best_c = ci;
                let _ = stay_gain;
            }

            // Re-insert i into the chosen community.
            sigma_tot[best_c] += ki;
            if best_c != ci {
                comm[i] = best_c;
                improved = true;
                moved_any = true;
            }
            // Self-loop weight stays in the same community regardless.
            let _ = k_self_loop;
        }
    }
    moved_any
}

/// Renumber communities so they are 0..k contiguous. Returns k.
fn renumber(comm: &mut [usize]) -> usize {
    let mut remap: BTreeMap<usize, usize> = BTreeMap::new();
    for c in comm.iter_mut() {
        let next = remap.len();
        let id = *remap.entry(*c).or_insert(next);
        *c = id;
    }
    remap.len()
}

fn renumber_assignment(a: &mut [u32]) {
    let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
    for c in a.iter_mut() {
        let next = remap.len() as u32;
        let id = *remap.entry(*c).or_insert(next);
        *c = id;
    }
}

fn build_meta(g: &UndirectedGraph, comm: &[usize]) -> UndirectedGraph {
    let k = renumber_clone(comm);
    let num_communities = k.iter().copied().max().map(|x| x + 1).unwrap_or(0);
    let mut adj_w: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for i in 0..g.num_nodes {
        let ci = k[i];
        for &(j, w) in &g.adj[i] {
            let cj = k[j];
            if i <= j {
                let key = (ci.min(cj), ci.max(cj));
                *adj_w.entry(key).or_insert(0.0) += w;
            }
        }
    }
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_communities];
    let mut deg = vec![0.0f64; num_communities];
    let mut two_m = 0.0f64;
    for ((a, b), w) in adj_w {
        if a == b {
            adj[a].push((b, w));
            deg[a] += 2.0 * w;
            two_m += 2.0 * w;
        } else {
            adj[a].push((b, w));
            adj[b].push((a, w));
            deg[a] += w;
            deg[b] += w;
            two_m += 2.0 * w;
        }
    }
    UndirectedGraph {
        num_nodes: num_communities,
        adj,
        k: deg,
        two_m,
    }
}

fn renumber_clone(comm: &[usize]) -> Vec<usize> {
    let mut out = comm.to_vec();
    renumber(&mut out);
    out
}

fn deterministic_shuffle(slice: &mut [usize], seed: u64) {
    // Fisher-Yates with a simple LCG so order is reproducible.
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    for i in (1..slice.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (state >> 32) as usize;
        slice.swap(i, r % (i + 1));
    }
}

/// Force the assignment to have exactly `target_k` clusters by splitting the
/// largest cluster (round-robin) when too few, or by merging the smallest
/// into the largest neighbour when too many.
///
/// This is a deterministic post-processing step. It does NOT preserve
/// modularity optimality, but Phase 1 needs `K` to match the config exactly
/// so the controller has a known shape.
pub(crate) fn clamp_to_k(graph: &FlyGraph, assignment: &[u32], target_k: usize) -> Vec<u32> {
    let mut a = assignment.to_vec();
    if a.is_empty() {
        return a;
    }
    let mut current = (a.iter().copied().max().unwrap_or(0) as usize) + 1;
    while current < target_k {
        // Split the largest cluster into two halves (lower / upper id index).
        let mut sizes = vec![0usize; current];
        for &c in &a {
            sizes[c as usize] += 1;
        }
        let (split, &split_size) = sizes
            .iter()
            .enumerate()
            .max_by_key(|(idx, &s)| (s, std::cmp::Reverse(*idx)))
            .unwrap();
        if split_size <= 1 {
            break;
        }
        let mut indices: Vec<usize> = a
            .iter()
            .enumerate()
            .filter(|(_, &c)| c as usize == split)
            .map(|(i, _)| i)
            .collect();
        indices.sort();
        let half = indices.len() / 2;
        for &i in &indices[half..] {
            a[i] = current as u32;
        }
        current += 1;
    }
    while current > target_k {
        // Merge the smallest cluster into its strongest neighbour.
        let mut sizes = vec![0usize; current];
        for &c in &a {
            sizes[c as usize] += 1;
        }
        let (smallest, _) = sizes
            .iter()
            .enumerate()
            .min_by_key(|(idx, &s)| (s, *idx))
            .unwrap();
        // For each candidate target community, sum weights of edges crossing.
        let mut weight_to: BTreeMap<u32, f64> = BTreeMap::new();
        for ((s, d), &w) in graph.edge_index.iter().zip(graph.edge_weight.iter()) {
            let cs = a[*s as usize];
            let cd = a[*d as usize];
            if cs as usize == smallest && cd as usize != smallest {
                *weight_to.entry(cd).or_insert(0.0) += w as f64;
            } else if cd as usize == smallest && cs as usize != smallest {
                *weight_to.entry(cs).or_insert(0.0) += w as f64;
            }
        }
        let target_other = weight_to
            .into_iter()
            .max_by(|(c1, w1), (c2, w2)| {
                w1.partial_cmp(w2)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(c2.cmp(c1)) // stable: smaller id wins on tie
            })
            .map(|(c, _)| c as usize)
            .unwrap_or_else(|| {
                // Disconnected smallest: pick neighbour 0 (or 1 if smallest == 0).
                if smallest == 0 {
                    1
                } else {
                    0
                }
            });

        // Merge `smallest` into `target_other`.
        for c in a.iter_mut() {
            if (*c as usize) == smallest {
                *c = target_other as u32;
            }
        }
        // Renumber to keep ids contiguous.
        let _ = renumber_in_place(&mut a);
        current = (a.iter().copied().max().unwrap_or(0) as usize) + 1;
    }
    a
}

fn renumber_in_place(a: &mut [u32]) -> BTreeSet<u32> {
    let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
    for c in a.iter_mut() {
        let next = remap.len() as u32;
        let id = *remap.entry(*c).or_insert(next);
        *c = id;
    }
    remap.values().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::modularity;
    use crate::synthetic::synthetic_fly_graph;

    #[test]
    fn produces_exact_k() {
        let g = synthetic_fly_graph(128, 7);
        for k in [4usize, 8, 16, 32] {
            let a = louvain(&g, k, 42).unwrap();
            assert_eq!(a.num_clusters, k, "K={} failed", k);
            a.check_valid(g.num_nodes).unwrap();
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        let g = synthetic_fly_graph(128, 7);
        let a = louvain(&g, 8, 42).unwrap();
        let b = louvain(&g, 8, 42).unwrap();
        assert_eq!(a.assignment, b.assignment);
    }

    #[test]
    fn different_seed_gives_valid_partition() {
        let g = synthetic_fly_graph(64, 1);
        let a = louvain(&g, 4, 1).unwrap();
        let b = louvain(&g, 4, 99).unwrap();
        a.check_valid(g.num_nodes).unwrap();
        b.check_valid(g.num_nodes).unwrap();
    }

    #[test]
    fn modularity_better_than_random() {
        let g = synthetic_fly_graph(128, 7);
        let a = louvain(&g, 8, 42).unwrap();
        let q = modularity(&g, &a.assignment);

        // Random baseline: assign nodes round-robin.
        let random: Vec<u32> = (0..g.num_nodes as u32).map(|i| i % 8).collect();
        let q_rand = modularity(&g, &random);

        assert!(
            q > q_rand,
            "louvain Q={} should beat round-robin Q={}",
            q,
            q_rand
        );
    }
}

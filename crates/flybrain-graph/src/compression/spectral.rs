//! Spectral clustering via power iteration on the symmetric normalised
//! Laplacian, followed by deterministic k-means.
//!
//! For a graph with adjacency `W`, degree `D = diag(sum_j W_ij)`, define
//! `L_sym = I - D^{-1/2} W D^{-1/2}`. The K bottom eigenvectors of `L_sym`
//! (excluding the trivial eigenvector for connected graphs) embed the nodes
//! into ℝ^K-1 such that close points correspond to nodes in the same
//! cluster.
//!
//! We compute the K bottom eigenvectors by inverse power iteration:
//! equivalently, iterate `M = 2I - L_sym` and pick the top eigenvectors of
//! `M`. This avoids dealing with repeated zero eigenvalues directly.
//!
//! For Phase 1's typical target (K ≤ 256, original graphs ≤ 10k nodes
//! after pre-aggregation, or a synthetic graph), simultaneous iteration
//! with Gram-Schmidt is fast enough and adds zero deps.
//!
//! K-means then clusters the K-dim embedding into K clusters with
//! deterministic seeding (spread initialisation: pick the point furthest
//! from already-chosen centroids).

use super::louvain;
use super::ClusterAssignment;
use crate::error::{GraphError, GraphResult};
use flybrain_core::graph::FlyGraph;
use std::collections::BTreeMap;

const POWER_ITERS: usize = 60;
const KMEANS_ITERS: usize = 40;

pub fn spectral(graph: &FlyGraph, target_k: usize, seed: u64) -> GraphResult<ClusterAssignment> {
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
    if target_k == 1 {
        return Ok(ClusterAssignment {
            assignment: vec![0u32; graph.num_nodes],
            num_clusters: 1,
            labels: None,
        });
    }

    let adj = build_undirected_adj(graph);
    let n = graph.num_nodes;

    // Compute embedding as K largest eigenvectors of M = 2I - L_sym
    // Equivalently: y = D^{-1/2} W D^{-1/2} x is power-iter friendly.
    let deg_inv_sqrt: Vec<f64> = adj
        .iter()
        .map(|row| {
            let d: f64 = row.iter().map(|(_, w)| *w).sum();
            if d > 0.0 {
                1.0 / d.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // We want K eigenvectors. Initialise with deterministic seeded vectors.
    //
    // Power iteration on `M = 2I - L_sym`:
    //   tmp = D^{-1/2} W D^{-1/2} v        # = (I - L_sym) v
    //   v_new = v + tmp                    # = (2I - L_sym) v
    //
    // After each step we Gram-Schmidt orthonormalise so simultaneous
    // iteration converges to the top-K eigenvectors of M, i.e. the bottom-K
    // of L_sym — which is the spectral embedding we want.
    let dim = target_k;
    let mut basis: Vec<Vec<f64>> = (0..dim)
        .map(|j| init_vec(n, seed.wrapping_add(j as u64)))
        .collect();

    let mut tmp = vec![0.0f64; n];
    for _ in 0..POWER_ITERS {
        for vec_j in basis.iter_mut() {
            for v in tmp.iter_mut() {
                *v = 0.0;
            }
            for (i, neighbours) in adj.iter().enumerate() {
                let xi = deg_inv_sqrt[i] * vec_j[i];
                for &(j, w) in neighbours {
                    tmp[j] += deg_inv_sqrt[j] * w * xi;
                }
            }
            for (slot, t) in vec_j.iter_mut().zip(tmp.iter()).take(n) {
                *slot += t;
            }
        }
        for j in 0..dim {
            // Split mutable / immutable so we can read earlier basis vectors
            // while writing into basis[j].
            let (head, tail) = basis.split_at_mut(j);
            let target = &mut tail[0];
            for prev in head.iter() {
                let dot_kj = dot(target, prev);
                axpy(target, prev, -dot_kj);
            }
            let norm = dot(target, target).sqrt();
            if norm > 1e-12 {
                for v in target.iter_mut() {
                    *v /= norm;
                }
            }
        }
        // Suppress unused-variable hint when dim==0 path collapses to no-op.
        let _ = dim;
    }

    // Build embedding matrix: row i = (basis[0][i], basis[1][i], ...).
    let mut embedding = vec![vec![0.0f64; dim]; n];
    for (i, row) in embedding.iter_mut().enumerate().take(n) {
        for (j, b) in basis.iter().enumerate().take(dim) {
            row[j] = b[i];
        }
    }

    // K-means with deterministic spread initialisation.
    let mut assignment = kmeans_cluster(&embedding, target_k, seed);
    let assn_u32: Vec<u32> = assignment.drain(..).map(|c| c as u32).collect();
    let mut clamped = louvain::clamp_to_k(graph, &assn_u32, target_k);
    let num_clusters = renumber(&mut clamped);

    Ok(ClusterAssignment {
        assignment: clamped,
        num_clusters,
        labels: None,
    })
}

fn build_undirected_adj(graph: &FlyGraph) -> Vec<Vec<(usize, f64)>> {
    let n = graph.num_nodes;
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for ((s, d), &w) in graph.edge_index.iter().zip(graph.edge_weight.iter()) {
        let s = *s as usize;
        let d = *d as usize;
        if s >= n || d >= n {
            continue;
        }
        let w = w.abs() as f64;
        let key = if s == d { (s, d) } else { (s.min(d), s.max(d)) };
        *acc.entry(key).or_insert(0.0) += w;
    }
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for ((a, b), w) in acc {
        if a == b {
            adj[a].push((b, w));
        } else {
            adj[a].push((b, w));
            adj[b].push((a, w));
        }
    }
    adj
}

fn init_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_add(0xdead_beef_cafe_babe);
    let mut out = vec![0.0f64; n];
    for v in out.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (state >> 32) as u32;
        // Map to (-1, 1)
        *v = (r as f64 / u32::MAX as f64) * 2.0 - 1.0;
    }
    let norm = dot(&out, &out).sqrt();
    if norm > 0.0 {
        for v in out.iter_mut() {
            *v /= norm;
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn axpy(y: &mut [f64], x: &[f64], a: f64) {
    for (yv, xv) in y.iter_mut().zip(x.iter()) {
        *yv += a * xv;
    }
}

/// Deterministic k-means with spread initialisation.
fn kmeans_cluster(points: &[Vec<f64>], k: usize, seed: u64) -> Vec<usize> {
    let n = points.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let dim = points.first().map(|v| v.len()).unwrap_or(0);

    // Spread init: first centroid = point with id (seed % n); subsequent
    // centroids = point furthest from all already-chosen centroids.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = (seed as usize) % n;
    centroids.push(points[first].clone());
    let mut chosen_indices = vec![first];
    while centroids.len() < k {
        let mut best_idx = 0usize;
        let mut best_dist = -1.0f64;
        for (i, p) in points.iter().enumerate() {
            if chosen_indices.contains(&i) {
                continue;
            }
            let d = centroids
                .iter()
                .map(|c| sq_dist(p, c))
                .fold(f64::INFINITY, f64::min);
            // Tie-break on smaller index to keep determinism.
            if d > best_dist + 1e-12 || (d > best_dist - 1e-12 && i < best_idx) {
                best_dist = d;
                best_idx = i;
            }
        }
        centroids.push(points[best_idx].clone());
        chosen_indices.push(best_idx);
    }

    let mut assignment = vec![0usize; n];
    for _ in 0..KMEANS_ITERS {
        let mut moved = false;
        for (i, p) in points.iter().enumerate() {
            let mut best_c = 0usize;
            let mut best_d = f64::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = sq_dist(p, centroid);
                if d < best_d - 1e-12 || (d < best_d + 1e-12 && c < best_c) {
                    best_d = d;
                    best_c = c;
                }
            }
            if assignment[i] != best_c {
                assignment[i] = best_c;
                moved = true;
            }
        }
        // Recompute centroids.
        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![vec![0.0f64; dim]; k];
        for (i, &c) in assignment.iter().enumerate() {
            counts[c] += 1;
            for (slot, p) in new_centroids[c].iter_mut().zip(points[i].iter()) {
                *slot += p;
            }
        }
        for (c, count) in counts.iter().enumerate() {
            if *count > 0 {
                let denom = *count as f64;
                for v in new_centroids[c].iter_mut().take(dim) {
                    *v /= denom;
                }
                centroids[c] = std::mem::take(&mut new_centroids[c]);
            }
        }
        if !moved {
            break;
        }
    }
    assignment
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::synthetic_fly_graph;

    #[test]
    fn produces_exact_k() {
        let g = synthetic_fly_graph(128, 7);
        for k in [4usize, 8, 16] {
            let a = spectral(&g, k, 42).unwrap();
            assert_eq!(a.num_clusters, k);
            a.check_valid(g.num_nodes).unwrap();
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        let g = synthetic_fly_graph(64, 1);
        let a = spectral(&g, 8, 42).unwrap();
        let b = spectral(&g, 8, 42).unwrap();
        assert_eq!(a.assignment, b.assignment);
    }

    #[test]
    fn k_one_returns_single_cluster() {
        let g = synthetic_fly_graph(32, 1);
        let a = spectral(&g, 1, 0).unwrap();
        assert_eq!(a.num_clusters, 1);
        assert!(a.assignment.iter().all(|&c| c == 0));
    }
}

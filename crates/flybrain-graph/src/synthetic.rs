use flybrain_core::graph::{FlyGraph, NodeMetadata};
use std::collections::BTreeMap;

/// Deterministic fly-inspired graph generator used as a fallback when real
/// connectome data is unavailable.
///
/// Produces a small-world directed weighted graph with `K` nodes split into 4
/// pseudo-regions and a power-law-ish degree distribution. The output is
/// stable for a given (`K`, `seed`) pair.
pub fn synthetic_fly_graph(num_nodes: usize, seed: u64) -> FlyGraph {
    let mut nodes = Vec::with_capacity(num_nodes);
    let regions = [
        "lobula",
        "mushroom_body",
        "central_complex",
        "antennal_lobe",
    ];
    for i in 0..num_nodes {
        nodes.push(NodeMetadata {
            id: i as u32,
            node_type: format!("ct_{}", i % 8),
            region: regions[i % regions.len()].to_string(),
            features: vec![],
        });
    }

    // Simple LCG so we don't pull a heavy `rand` crate at this stage.
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut next_u32 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };

    // Each node gets ~ log2(N) outgoing edges, biased to in-region.
    let avg_deg = ((num_nodes as f32).log2().ceil() as usize).max(2);
    let mut edge_index = Vec::new();
    let mut edge_weight = Vec::new();
    let mut is_excitatory = Vec::new();

    for src in 0..num_nodes {
        let src_region = &nodes[src].region;
        for _ in 0..avg_deg {
            // 75% chance to pick a neighbour in the same region.
            let dst = if next_u32() % 4 != 0 {
                let same_region: Vec<u32> = nodes
                    .iter()
                    .filter(|n| n.region == *src_region && n.id as usize != src)
                    .map(|n| n.id)
                    .collect();
                if same_region.is_empty() {
                    (next_u32() as usize % num_nodes) as u32
                } else {
                    same_region[next_u32() as usize % same_region.len()]
                }
            } else {
                (next_u32() as usize % num_nodes) as u32
            };
            if dst as usize == src {
                continue;
            }
            edge_index.push((src as u32, dst));
            // Weights in (0.1, 1.0).
            let w = 0.1 + (next_u32() % 900) as f32 / 1000.0;
            edge_weight.push(w);
            // ~80% excitatory, like in fly central brain estimates.
            is_excitatory.push(next_u32() % 5 != 0);
        }
    }

    let mut provenance = BTreeMap::new();
    provenance.insert("source".into(), serde_json::json!("synthetic"));
    provenance.insert("seed".into(), serde_json::json!(seed));
    provenance.insert("avg_deg".into(), serde_json::json!(avg_deg));

    FlyGraph {
        num_nodes,
        edge_index,
        edge_weight,
        is_excitatory,
        nodes,
        provenance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_is_deterministic() {
        let a = synthetic_fly_graph(64, 42);
        let b = synthetic_fly_graph(64, 42);
        assert_eq!(a.num_nodes, b.num_nodes);
        assert_eq!(a.num_edges(), b.num_edges());
        assert_eq!(a.edge_index, b.edge_index);
        assert_eq!(a.edge_weight, b.edge_weight);
    }

    #[test]
    fn different_seed_gives_different_graph() {
        let a = synthetic_fly_graph(64, 1);
        let b = synthetic_fly_graph(64, 2);
        assert_ne!(a.edge_index, b.edge_index);
    }

    #[test]
    fn produces_some_edges() {
        let g = synthetic_fly_graph(32, 7);
        assert!(g.num_edges() > 0);
    }
}

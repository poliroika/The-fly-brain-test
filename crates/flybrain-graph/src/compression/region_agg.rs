//! Region-based aggregation: every node sharing a `region` field becomes one
//! cluster. Trivial, deterministic, and ideal as a sanity baseline because
//! it preserves anatomical priors.

use super::ClusterAssignment;
use crate::error::{GraphError, GraphResult};
use flybrain_core::graph::FlyGraph;
use std::collections::BTreeMap;

pub fn region_aggregation(graph: &FlyGraph) -> GraphResult<ClusterAssignment> {
    if graph.num_nodes == 0 {
        return Err(GraphError::Invalid("empty graph".into()));
    }
    let mut label_to_id: BTreeMap<String, u32> = BTreeMap::new();
    let mut labels: Vec<String> = Vec::new();
    let mut assignment = Vec::with_capacity(graph.num_nodes);
    for node in &graph.nodes {
        let next_id = label_to_id.len() as u32;
        let id = *label_to_id.entry(node.region.clone()).or_insert_with(|| {
            labels.push(node.region.clone());
            next_id
        });
        assignment.push(id);
    }
    let num_clusters = labels.len();
    Ok(ClusterAssignment {
        assignment,
        num_clusters,
        labels: Some(labels),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::synthetic_fly_graph;

    #[test]
    fn synthetic_has_four_regions() {
        let g = synthetic_fly_graph(64, 1);
        let a = region_aggregation(&g).unwrap();
        assert_eq!(a.num_clusters, 4);
        a.check_valid(g.num_nodes).unwrap();
        let labels = a.labels.unwrap();
        assert!(labels.contains(&"lobula".to_string()));
        assert!(labels.contains(&"mushroom_body".to_string()));
    }
}

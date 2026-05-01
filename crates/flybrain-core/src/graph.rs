use crate::action::GraphAction;
use crate::error::{FlybrainError, FlybrainResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Compressed FlyWire connectome graph used as a structural prior.
///
/// Edges are stored COO-style: `edge_index[i] = (src, dst)` with corresponding
/// `edge_weight[i]`. We keep `is_excitatory` as a parallel array so the
/// controller can use sign information when desired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlyGraph {
    pub num_nodes: usize,
    pub edge_index: Vec<(u32, u32)>,
    pub edge_weight: Vec<f32>,

    /// Optional per-edge sign (true = excitatory, false = inhibitory).
    #[serde(default)]
    pub is_excitatory: Vec<bool>,

    pub nodes: Vec<NodeMetadata>,

    /// Free-form metadata about the compression source/method.
    #[serde(default)]
    pub provenance: BTreeMap<String, serde_json::Value>,
}

impl FlyGraph {
    pub fn empty() -> Self {
        Self {
            num_nodes: 0,
            edge_index: Vec::new(),
            edge_weight: Vec::new(),
            is_excitatory: Vec::new(),
            nodes: Vec::new(),
            provenance: BTreeMap::new(),
        }
    }

    pub fn num_edges(&self) -> usize {
        self.edge_index.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub id: u32,

    /// e.g. "MBON", "KC", "DAN" for fly cell types; "Planner", "Coder" for agent graphs.
    #[serde(default)]
    pub node_type: String,

    /// Brain region or cluster id.
    #[serde(default)]
    pub region: String,

    #[serde(default)]
    pub features: Vec<f32>,
}

/// Dynamic agent graph mutated by the controller during MAS execution.
///
/// Lightweight wrapper over an adjacency map `from -> to -> weight`. Includes
/// stable hashing so traces can reference graph snapshots cheaply.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentGraph {
    pub nodes: BTreeSet<String>,
    /// Adjacency: outer key = source agent, inner key = destination agent, value = weight.
    pub edges: BTreeMap<String, BTreeMap<String, f32>>,
}

impl AgentGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_nodes<I, S>(nodes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut g = Self::default();
        for n in nodes {
            g.nodes.insert(n.into());
        }
        g
    }

    pub fn add_node(&mut self, name: impl Into<String>) {
        self.nodes.insert(name.into());
    }

    pub fn has_node(&self, name: &str) -> bool {
        self.nodes.contains(name)
    }

    pub fn edge_weight(&self, from: &str, to: &str) -> Option<f32> {
        self.edges.get(from).and_then(|m| m.get(to)).copied()
    }

    pub fn add_edge(&mut self, from: &str, to: &str, weight: f32) -> FlybrainResult<()> {
        if !self.nodes.contains(from) {
            return Err(FlybrainError::InvalidAction(format!(
                "unknown source agent {from}"
            )));
        }
        if !self.nodes.contains(to) {
            return Err(FlybrainError::InvalidAction(format!(
                "unknown dest agent {to}"
            )));
        }
        self.edges
            .entry(from.to_string())
            .or_default()
            .insert(to.to_string(), weight);
        Ok(())
    }

    pub fn remove_edge(&mut self, from: &str, to: &str) {
        if let Some(m) = self.edges.get_mut(from) {
            m.remove(to);
            if m.is_empty() {
                self.edges.remove(from);
            }
        }
    }

    pub fn scale_edge(&mut self, from: &str, to: &str, factor: f32) -> FlybrainResult<()> {
        let w = self
            .edges
            .get_mut(from)
            .and_then(|m| m.get_mut(to))
            .ok_or_else(|| FlybrainError::InvalidAction(format!("no edge {from}->{to}")))?;
        *w *= factor;
        Ok(())
    }

    /// Apply a `GraphAction` that mutates the graph itself. Non-mutating actions
    /// (CallMemory, CallRetriever, CallToolExecutor, CallVerifier, Terminate,
    /// ActivateAgent) are no-ops for the graph and return `Ok(())`; the runtime
    /// scheduler is responsible for handling them.
    pub fn apply(&mut self, action: &GraphAction) -> FlybrainResult<()> {
        match action {
            GraphAction::AddEdge { from, to, weight } => self.add_edge(from, to, *weight),
            GraphAction::RemoveEdge { from, to } => {
                self.remove_edge(from, to);
                Ok(())
            }
            GraphAction::ScaleEdge { from, to, factor } => self.scale_edge(from, to, *factor),
            GraphAction::ActivateAgent { agent } => {
                if !self.nodes.contains(agent) {
                    return Err(FlybrainError::InvalidAction(format!(
                        "unknown agent {agent}"
                    )));
                }
                Ok(())
            }
            GraphAction::CallMemory
            | GraphAction::CallRetriever
            | GraphAction::CallToolExecutor
            | GraphAction::CallVerifier
            | GraphAction::Terminate => Ok(()),
        }
    }

    /// Stable hash of the graph topology + weights, used as `current_graph_hash`
    /// in trace steps. Uses canonical JSON serialization (BTreeMap iteration is
    /// already deterministic) and the FxHash-style seahash for speed.
    pub fn hash(&self) -> String {
        // We use serde_json to canonicalize, then SHA-256-like via Blake3 in the
        // future. For Phase 0, a simple FNV-1a 64 over the JSON bytes is plenty.
        let bytes = serde_json::to_vec(self).expect("AgentGraph always serializes");
        let mut h: u64 = 0xcbf29ce484222325;
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_3() -> AgentGraph {
        AgentGraph::with_nodes(["Planner", "Coder", "Verifier"])
    }

    #[test]
    fn add_edge_requires_known_nodes() {
        let mut g = graph_3();
        assert!(g.add_edge("Planner", "Coder", 1.0).is_ok());
        assert!(g.add_edge("Planner", "Unknown", 1.0).is_err());
    }

    #[test]
    fn scale_edge_changes_weight() {
        let mut g = graph_3();
        g.add_edge("Planner", "Coder", 1.0).unwrap();
        g.scale_edge("Planner", "Coder", 2.5).unwrap();
        assert_eq!(g.edge_weight("Planner", "Coder"), Some(2.5));
    }

    #[test]
    fn hash_changes_after_mutation() {
        let mut g = graph_3();
        let h0 = g.hash();
        g.add_edge("Planner", "Coder", 1.0).unwrap();
        let h1 = g.hash();
        assert_ne!(h0, h1);
    }

    #[test]
    fn hash_is_stable_under_insert_order() {
        let mut a = graph_3();
        a.add_edge("Planner", "Coder", 1.0).unwrap();
        a.add_edge("Coder", "Verifier", 1.0).unwrap();

        let mut b = graph_3();
        b.add_edge("Coder", "Verifier", 1.0).unwrap();
        b.add_edge("Planner", "Coder", 1.0).unwrap();

        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn apply_terminate_is_no_op() {
        let mut g = graph_3();
        let h0 = g.hash();
        g.apply(&GraphAction::Terminate).unwrap();
        assert_eq!(g.hash(), h0);
    }
}

use serde::{Deserialize, Serialize};

/// Actions the FlyBrain controller can take to mutate the MAS execution.
///
/// Mirrors README §8 `Output actions` block. Action IDs are stable so that
/// trained policies can be loaded across versions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphAction {
    /// Schedule `agent` to run on the next tick.
    ActivateAgent { agent: String },

    /// Add a directed edge `from -> to` with optional weight.
    AddEdge {
        from: String,
        to: String,
        #[serde(default = "default_edge_weight")]
        weight: f32,
    },

    /// Remove a directed edge `from -> to` if it exists.
    RemoveEdge { from: String, to: String },

    /// Multiply edge weight by `factor` (>1 increases, <1 decreases).
    ScaleEdge {
        from: String,
        to: String,
        factor: f32,
    },

    /// Trigger memory read.
    CallMemory,

    /// Trigger retriever.
    CallRetriever,

    /// Trigger tool executor.
    CallToolExecutor,

    /// Trigger verifier.
    CallVerifier,

    /// Stop execution.
    Terminate,
}

fn default_edge_weight() -> f32 {
    1.0
}

impl GraphAction {
    /// Discrete action-space ID used by the controller's action head.
    pub fn discriminant(&self) -> u8 {
        match self {
            GraphAction::ActivateAgent { .. } => 0,
            GraphAction::AddEdge { .. } => 1,
            GraphAction::RemoveEdge { .. } => 2,
            GraphAction::ScaleEdge { .. } => 3,
            GraphAction::CallMemory => 4,
            GraphAction::CallRetriever => 5,
            GraphAction::CallToolExecutor => 6,
            GraphAction::CallVerifier => 7,
            GraphAction::Terminate => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_terminate() {
        let json = serde_json::to_string(&GraphAction::Terminate).unwrap();
        assert_eq!(json, r#"{"kind":"terminate"}"#);
        let parsed: GraphAction = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, GraphAction::Terminate);
    }

    #[test]
    fn round_trip_add_edge() {
        let action = GraphAction::AddEdge {
            from: "Planner".into(),
            to: "Coder".into(),
            weight: 0.5,
        };
        let json = serde_json::to_string(&action).unwrap();
        let parsed: GraphAction = serde_json::from_str(&json).unwrap();
        assert_eq!(action, parsed);
    }

    #[test]
    fn discriminants_are_stable() {
        assert_eq!(
            GraphAction::ActivateAgent { agent: "x".into() }.discriminant(),
            0
        );
        assert_eq!(GraphAction::Terminate.discriminant(), 8);
    }
}

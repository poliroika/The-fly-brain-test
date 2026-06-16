use crate::action::GraphAction;
use crate::task::TaskType;
use crate::verify::VerificationResult;
use serde::{Deserialize, Serialize};

/// One tick of MAS execution: the controller emits a `graph_action`, the
/// runtime applies it, the active agent runs (or doesn't), tools are called,
/// errors are collected, the verifier may emit a partial score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub step_id: u64,
    pub t_unix_ms: i64,

    /// Agent that ran on this tick (None for pure graph mutations).
    #[serde(default)]
    pub active_agent: Option<String>,

    #[serde(default)]
    pub input_msg_id: Option<String>,

    #[serde(default)]
    pub output_summary: Option<String>,

    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,

    #[serde(default)]
    pub errors: Vec<String>,

    #[serde(default)]
    pub tokens_in: u64,
    #[serde(default)]
    pub tokens_out: u64,

    #[serde(default)]
    pub latency_ms: u64,

    #[serde(default)]
    pub verifier_score: Option<f32>,

    /// Hash of the agent graph at the END of this step.
    pub current_graph_hash: String,

    /// Action emitted by the controller for this step.
    pub graph_action: GraphAction,

    #[serde(default)]
    pub cost_rub: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
    pub ok: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub latency_ms: u64,
}

/// Aggregate counters for a single MAS run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Totals {
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub llm_calls: u32,
    pub tool_calls: u32,
    pub failed_tool_calls: u32,
    pub latency_ms: u64,
    pub cost_rub: f32,
}

/// One full MAS execution trace. Persisted as a JSONL row in
/// `data/traces/{split}/...`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub task_id: String,
    pub task_type: TaskType,

    pub steps: Vec<TraceStep>,

    #[serde(default)]
    pub final_answer: Option<String>,

    #[serde(default)]
    pub verification: Option<VerificationResult>,

    pub totals: Totals,

    /// Free-form metadata (controller variant, model URI, seed, etc.).
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl Trace {
    pub fn new(task_id: impl Into<String>, task_type: TaskType) -> Self {
        Self {
            task_id: task_id.into(),
            task_type,
            steps: Vec::new(),
            final_answer: None,
            verification: None,
            totals: Totals::default(),
            metadata: serde_json::Value::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::GraphAction;

    #[test]
    fn empty_trace_round_trips() {
        let t = Trace::new("t1", TaskType::Coding);
        let s = serde_json::to_string(&t).unwrap();
        let parsed: Trace = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.task_id, "t1");
        assert_eq!(parsed.task_type, TaskType::Coding);
        assert!(parsed.steps.is_empty());
    }

    #[test]
    fn trace_step_with_action_round_trips() {
        let step = TraceStep {
            step_id: 0,
            t_unix_ms: 1_700_000_000_000,
            active_agent: Some("Planner".into()),
            input_msg_id: None,
            output_summary: Some("decomposed".into()),
            tool_calls: vec![],
            errors: vec![],
            tokens_in: 100,
            tokens_out: 50,
            latency_ms: 1234,
            verifier_score: None,
            current_graph_hash: "deadbeef".into(),
            graph_action: GraphAction::ActivateAgent {
                agent: "Coder".into(),
            },
            cost_rub: 0.06,
        };
        let s = serde_json::to_string(&step).unwrap();
        let parsed: TraceStep = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.step_id, 0);
        assert!(matches!(
            parsed.graph_action,
            GraphAction::ActivateAgent { .. }
        ));
    }
}

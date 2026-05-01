use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Coding,
    Math,
    Research,
    ToolUse,
    SyntheticRouting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub task_id: String,
    pub task_type: TaskType,
    pub prompt: String,

    /// Optional reference answer / unit tests / expected JSON.
    #[serde(default)]
    pub ground_truth: Option<serde_json::Value>,

    /// Per-task token / call / latency budget. None means "use global budget".
    #[serde(default)]
    pub budget: Option<TaskBudget>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskBudget {
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub max_calls: Option<u32>,
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
    pub max_cost_rub: Option<f32>,
}

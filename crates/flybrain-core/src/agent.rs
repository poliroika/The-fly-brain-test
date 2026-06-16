use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Which YandexGPT tier this agent should use.
///
/// We default to `Lite` for fast/cheap roles and `Pro` for reasoning/judgment
/// roles (Critic, Judge, Verifier, Planner). See `configs/llm/yandex.yaml`
/// for the canonical mapping.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    #[default]
    Lite,
    Pro,
}

/// Static description of one agent in the MAS.
///
/// Mirrors `class AgentSpec` from the README §6 plus a `model_tier` hint that
/// the LLM client uses to pick between `yandexgpt-lite/latest` and
/// `yandexgpt/latest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,
    pub role: String,
    pub system_prompt: String,

    #[serde(default)]
    pub tools: Vec<String>,

    /// Free-form JSON schema for the agent's input message.
    #[serde(default)]
    pub input_schema: serde_json::Value,

    /// Free-form JSON schema for the agent's output message.
    #[serde(default)]
    pub output_schema: serde_json::Value,

    /// Multiplicative weight applied to this agent's token cost when computing
    /// reward (see README §12.3 reward formula).
    #[serde(default = "default_cost_weight")]
    pub cost_weight: f32,

    #[serde(default)]
    pub model_tier: ModelTier,

    /// Optional metadata bag (e.g. role-tags, fly-region affinity).
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

fn default_cost_weight() -> f32 {
    1.0
}

impl AgentSpec {
    pub fn new(
        name: impl Into<String>,
        role: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            role: role.into(),
            system_prompt: system_prompt.into(),
            tools: Vec::new(),
            input_schema: serde_json::Value::Null,
            output_schema: serde_json::Value::Null,
            cost_weight: 1.0,
            model_tier: ModelTier::Lite,
            metadata: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_spec_round_trip() {
        let spec = AgentSpec::new("Planner", "planner/decomposer", "you plan");
        let json = serde_json::to_string(&spec).unwrap();
        let parsed: AgentSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, parsed);
    }

    #[test]
    fn model_tier_defaults_to_lite() {
        assert_eq!(ModelTier::default(), ModelTier::Lite);
    }
}

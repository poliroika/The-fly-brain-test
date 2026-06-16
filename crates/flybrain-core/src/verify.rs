use serde::{Deserialize, Serialize};

/// Mirrors `class VerificationResult` from README §10.
///
/// `passed` is the headline pass/fail; `score` is a graded version in [0, 1];
/// `failed_component` and `suggested_next_agent` let the controller decide
/// whom to call next; `reward_delta` is consumed directly by the trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub passed: bool,

    #[serde(default)]
    pub score: f32,

    #[serde(default)]
    pub errors: Vec<String>,

    #[serde(default)]
    pub warnings: Vec<String>,

    #[serde(default)]
    pub failed_component: Option<String>,

    #[serde(default)]
    pub suggested_next_agent: Option<String>,

    #[serde(default)]
    pub reward_delta: f32,
}

impl VerificationResult {
    pub fn pass(score: f32) -> Self {
        Self {
            passed: true,
            score,
            errors: vec![],
            warnings: vec![],
            failed_component: None,
            suggested_next_agent: None,
            reward_delta: 0.0,
        }
    }

    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            score: 0.0,
            errors: vec![reason.into()],
            warnings: vec![],
            failed_component: None,
            suggested_next_agent: None,
            reward_delta: -1.0,
        }
    }
}

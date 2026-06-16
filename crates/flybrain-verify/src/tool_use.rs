//! Verifies that the tool calls in a `TraceStep` are well-formed:
//!
//! 1. Every tool name is in the allow-list passed by the caller.
//! 2. Each tool call carries a non-empty `args` object.
//! 3. Optional per-tool required-argument lists are honoured.
//!
//! Operates on the `ToolCall` JSON shape so it stays in sync with
//! `flybrain-core::trace::ToolCall` without depending on it.

use std::collections::{HashMap, HashSet};

use flybrain_core::verify::VerificationResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolUseSpec {
    /// Allowed tool names. Empty list means "no constraint".
    #[serde(default)]
    pub allowed: Vec<String>,
    /// Per-tool required-argument names.
    #[serde(default)]
    pub requirements: HashMap<String, Vec<String>>,
}

pub fn verify_tool_calls(calls: &[Value], spec: &ToolUseSpec) -> VerificationResult {
    let allowed: HashSet<&str> = spec.allowed.iter().map(String::as_str).collect();
    let mut errors = Vec::new();

    for (i, call) in calls.iter().enumerate() {
        let Some(obj) = call.as_object() else {
            errors.push(format!("call[{i}]: not an object"));
            continue;
        };
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");

        if !spec.allowed.is_empty() && !allowed.contains(name) {
            errors.push(format!("call[{i}]: tool '{name}' not in allow-list"));
            continue;
        }
        let args = obj.get("args");
        let Some(Value::Object(arg_map)) = args else {
            errors.push(format!("call[{i}]: tool '{name}' missing 'args' object"));
            continue;
        };
        if let Some(required) = spec.requirements.get(name) {
            for req in required {
                if !arg_map.contains_key(req) {
                    errors.push(format!(
                        "call[{i}]: tool '{name}' missing required arg '{req}'"
                    ));
                }
            }
        }
    }

    if errors.is_empty() {
        VerificationResult::pass(1.0)
    } else {
        VerificationResult {
            passed: false,
            score: 0.0,
            errors,
            warnings: vec![],
            failed_component: Some("tool_use".into()),
            suggested_next_agent: Some("ToolExecutor".into()),
            reward_delta: -0.4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn allow_list_violation_fails() {
        let calls = vec![json!({"name": "rm_rf", "args": {}})];
        let spec = ToolUseSpec {
            allowed: vec!["python_exec".into()],
            ..Default::default()
        };
        let r = verify_tool_calls(&calls, &spec);
        assert!(!r.passed);
        assert!(r.errors[0].contains("not in allow-list"));
    }

    #[test]
    fn missing_required_arg_fails() {
        let calls = vec![json!({"name": "python_exec", "args": {}})];
        let mut req = HashMap::new();
        req.insert("python_exec".into(), vec!["code".into()]);
        let spec = ToolUseSpec {
            allowed: vec!["python_exec".into()],
            requirements: req,
        };
        let r = verify_tool_calls(&calls, &spec);
        assert!(!r.passed);
        assert!(r.errors[0].contains("missing required arg 'code'"));
    }

    #[test]
    fn no_calls_passes_trivially() {
        let r = verify_tool_calls(&[], &ToolUseSpec::default());
        assert!(r.passed);
    }

    #[test]
    fn empty_allow_list_means_no_constraint() {
        let calls = vec![json!({"name": "anything", "args": {"k": 1}})];
        let r = verify_tool_calls(&calls, &ToolUseSpec::default());
        assert!(r.passed);
    }

    #[test]
    fn missing_args_object_fails() {
        let calls = vec![json!({"name": "python_exec"})];
        let r = verify_tool_calls(&calls, &ToolUseSpec::default());
        assert!(!r.passed);
    }
}

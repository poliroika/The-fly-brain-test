//! Verifies a `unit_tester`-style payload (the `output` field of a
//! `python_exec` / `unit_tester` `ToolCall`).
//!
//! Expected shape (the same one `flybrain.runtime.tools.unit_tester`
//! emits):
//!
//! ```json
//! { "passed": <int>, "failed": <int>, "all_passed": <bool>,
//!   "stdout": "...", "stderr": "..." }
//! ```
//!
//! * If `failed > 0` or `all_passed == false`, the run failed.
//! * If the payload is missing the counters entirely we emit a warning
//!   instead of a hard error so a generic `python_exec` (no test
//!   counts) still produces a useful `VerificationResult`.

use flybrain_core::verify::VerificationResult;
use serde_json::Value;

pub fn verify_unit_tests(payload: &Value) -> VerificationResult {
    let Some(obj) = payload.as_object() else {
        return VerificationResult::fail("unit_test: payload is not an object");
    };

    let passed_count = obj.get("passed").and_then(Value::as_u64);
    let failed_count = obj.get("failed").and_then(Value::as_u64);
    let all_passed = obj.get("all_passed").and_then(Value::as_bool);

    if let Some(failed) = failed_count {
        if failed > 0 {
            return VerificationResult {
                passed: false,
                score: 0.0,
                errors: vec![format!(
                    "unit_test: {failed} test(s) failed (passed={})",
                    passed_count.unwrap_or(0)
                )],
                warnings: vec![],
                failed_component: Some("unit_test".into()),
                suggested_next_agent: Some("Debugger".into()),
                reward_delta: -0.5,
            };
        }
    }

    if all_passed == Some(false) {
        return VerificationResult {
            passed: false,
            score: 0.0,
            errors: vec!["unit_test: all_passed is false".into()],
            warnings: vec![],
            failed_component: Some("unit_test".into()),
            suggested_next_agent: Some("Debugger".into()),
            reward_delta: -0.5,
        };
    }

    let mut warnings = Vec::new();
    if passed_count.is_none() && all_passed.is_none() {
        warnings.push("unit_test: payload missing counters and all_passed".into());
    }

    VerificationResult {
        passed: true,
        score: 1.0,
        errors: vec![],
        warnings,
        failed_component: None,
        suggested_next_agent: None,
        reward_delta: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn all_passed_true_succeeds() {
        let p = json!({"passed": 3, "failed": 0, "all_passed": true});
        assert!(verify_unit_tests(&p).passed);
    }

    #[test]
    fn any_failed_count_fails() {
        let p = json!({"passed": 2, "failed": 1, "all_passed": false});
        let r = verify_unit_tests(&p);
        assert!(!r.passed);
        assert_eq!(r.suggested_next_agent.as_deref(), Some("Debugger"));
    }

    #[test]
    fn all_passed_false_fails() {
        let p = json!({"all_passed": false});
        assert!(!verify_unit_tests(&p).passed);
    }

    #[test]
    fn missing_counters_warn_but_pass() {
        let p = json!({"stdout": "ok"});
        let r = verify_unit_tests(&p);
        assert!(r.passed);
        assert!(!r.warnings.is_empty());
    }

    #[test]
    fn non_object_fails() {
        assert!(!verify_unit_tests(&json!("oops")).passed);
    }
}

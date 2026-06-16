//! Trace-level structural invariants (README §10 "trace verification"):
//!
//! 1. `task_id` non-empty.
//! 2. At least one step.
//! 3. `step_id`s are strictly monotonically increasing from 0.
//! 4. `totals.llm_calls` matches the number of steps with a non-empty
//!    `active_agent`.
//! 5. `totals.tokens_in` and `totals.tokens_out` agree with the sum
//!    over steps (within ±1 to allow for rounding).
//! 6. If `final_answer` is `Some(_)`, at least one step records the
//!    `final_answer` produced component (best-effort: we accept any
//!    non-empty `final_answer` since `produced_components` is in
//!    `trace.metadata`, not on `TraceStep`).

use flybrain_core::verify::VerificationResult;
use serde_json::Value;

pub fn verify_trace(trace: &Value) -> VerificationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    let Some(obj) = trace.as_object() else {
        return VerificationResult::fail("trace is not an object");
    };

    let task_id = obj.get("task_id").and_then(Value::as_str).unwrap_or("");
    if task_id.is_empty() {
        errors.push("trace: task_id missing or empty".into());
    }

    let empty_steps = Vec::new();
    let steps = obj
        .get("steps")
        .and_then(Value::as_array)
        .unwrap_or(&empty_steps);
    if steps.is_empty() {
        errors.push("trace: no steps recorded".into());
    }

    let mut expected_id: i64 = 0;
    let mut sum_in: u64 = 0;
    let mut sum_out: u64 = 0;
    let mut llm_calls: u64 = 0;
    for step in steps.iter() {
        let id = step.get("step_id").and_then(Value::as_i64).unwrap_or(-1);
        if id != expected_id {
            errors.push(format!(
                "trace: step_id discontinuity at expected={expected_id} actual={id}"
            ));
        }
        expected_id = id + 1;
        sum_in += step.get("tokens_in").and_then(Value::as_u64).unwrap_or(0);
        sum_out += step.get("tokens_out").and_then(Value::as_u64).unwrap_or(0);
        if step
            .get("active_agent")
            .and_then(Value::as_str)
            .map(|s| !s.is_empty())
            .unwrap_or(false)
        {
            llm_calls += 1;
        }
    }

    if let Some(totals) = obj.get("totals").and_then(Value::as_object) {
        let t_in = totals.get("tokens_in").and_then(Value::as_u64).unwrap_or(0);
        let t_out = totals
            .get("tokens_out")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let t_calls = totals.get("llm_calls").and_then(Value::as_u64).unwrap_or(0);
        if t_in.abs_diff(sum_in) > 1 {
            warnings.push(format!(
                "trace: totals.tokens_in mismatch (totals={t_in}, sum_steps={sum_in})"
            ));
        }
        if t_out.abs_diff(sum_out) > 1 {
            warnings.push(format!(
                "trace: totals.tokens_out mismatch (totals={t_out}, sum_steps={sum_out})"
            ));
        }
        if t_calls != llm_calls {
            warnings.push(format!(
                "trace: totals.llm_calls mismatch (totals={t_calls}, sum_steps={llm_calls})"
            ));
        }
    } else {
        warnings.push("trace: missing totals object".into());
    }

    let final_answer = obj.get("final_answer").and_then(Value::as_str);
    if final_answer.is_none_or(str::is_empty) {
        warnings.push("trace: final_answer missing or empty".into());
    }

    if errors.is_empty() {
        VerificationResult {
            passed: true,
            score: if warnings.is_empty() { 1.0 } else { 0.85 },
            errors,
            warnings,
            failed_component: None,
            suggested_next_agent: None,
            reward_delta: 0.0,
        }
    } else {
        VerificationResult {
            passed: false,
            score: 0.0,
            errors,
            warnings,
            failed_component: Some("trace".into()),
            suggested_next_agent: None,
            reward_delta: -0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_trace() -> Value {
        json!({
            "task_id": "t1",
            "task_type": "coding",
            "steps": [
                {"step_id": 0, "active_agent": "Coder", "tokens_in": 10, "tokens_out": 5,
                 "tool_calls": [], "errors": []},
                {"step_id": 1, "active_agent": "Verifier", "tokens_in": 5, "tokens_out": 2,
                 "tool_calls": [], "errors": []}
            ],
            "totals": {"tokens_in": 15, "tokens_out": 7, "llm_calls": 2,
                       "tool_calls_total": 0, "tool_call_failures": 0,
                       "latency_ms": 0, "cost_rub": 0.0},
            "final_answer": "done"
        })
    }

    #[test]
    fn good_trace_passes() {
        let r = verify_trace(&good_trace());
        assert!(r.passed, "errors: {:?}", r.errors);
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn missing_task_id_fails() {
        let mut t = good_trace();
        t.as_object_mut().unwrap().remove("task_id");
        let r = verify_trace(&t);
        assert!(!r.passed);
    }

    #[test]
    fn step_id_gap_fails() {
        let mut t = good_trace();
        t["steps"][1]["step_id"] = json!(5);
        let r = verify_trace(&t);
        assert!(!r.passed);
        assert!(r.errors.iter().any(|e| e.contains("discontinuity")));
    }

    #[test]
    fn totals_mismatch_warns_but_passes() {
        let mut t = good_trace();
        t["totals"]["llm_calls"] = json!(99);
        let r = verify_trace(&t);
        assert!(r.passed);
        assert!(!r.warnings.is_empty());
        assert!(r.score < 1.0);
    }

    #[test]
    fn empty_steps_fails() {
        let mut t = good_trace();
        t["steps"] = json!([]);
        let r = verify_trace(&t);
        assert!(!r.passed);
    }
}

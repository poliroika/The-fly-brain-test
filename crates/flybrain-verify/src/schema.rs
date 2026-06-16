//! Minimal JSON-schema validator covering the subset the agents emit.
//!
//! Phase 3 deliberately avoids the full `jsonschema-rs` crate (which
//! pulls a `getrandom` version that demands `edition2024`). The agents
//! only emit two shapes — final-answer envelopes and tool args — so a
//! hand-rolled subset is enough:
//!
//! * `type` ∈ {`"object"`, `"array"`, `"string"`, `"number"`,
//!   `"integer"`, `"boolean"`, `"null"`} (or a list of those)
//! * `required: [String]` for `"object"`s
//! * `properties: { name: <subschema> }`
//! * `items: <subschema>` for `"array"`s
//! * `enum: [Value]`
//! * `minLength` / `maxLength` for `"string"`s
//! * `minimum` / `maximum` for `"number"` / `"integer"`s
//! * `pattern: regex` (matched as a substring search via `String::contains` —
//!   no real regex engine is pulled in)
//!
//! Anything else in the schema is ignored. This is a deterministic,
//! dependency-free verifier and round-trips to `VerificationResult`.

use flybrain_core::verify::VerificationResult;
use serde_json::Value;

/// Validate `payload` against `schema`. Returns a `VerificationResult`
/// with `failed_component = Some("schema")` on failure.
pub fn verify_schema(payload: &Value, schema: &Value) -> VerificationResult {
    let mut errors = Vec::new();
    walk(payload, schema, "$", &mut errors);
    if errors.is_empty() {
        VerificationResult::pass(1.0)
    } else {
        VerificationResult {
            passed: false,
            score: 0.0,
            errors,
            warnings: vec![],
            failed_component: Some("schema".into()),
            suggested_next_agent: Some("SchemaValidator".into()),
            reward_delta: -0.5,
        }
    }
}

fn walk(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(schema_obj) = schema.as_object() else {
        return; // schema is not an object → treat as accept-all
    };

    if let Some(t) = schema_obj.get("type") {
        if !type_matches(value, t) {
            errors.push(format!(
                "{path}: expected type {}, got {}",
                describe(t),
                describe_value_type(value)
            ));
            return; // mismatch → no point walking deeper
        }
    }

    if let Some(enum_values) = schema_obj.get("enum").and_then(Value::as_array) {
        if !enum_values.iter().any(|allowed| allowed == value) {
            errors.push(format!("{path}: value not in enum"));
        }
    }

    match value {
        Value::Object(map) => {
            if let Some(required) = schema_obj.get("required").and_then(Value::as_array) {
                for r in required {
                    if let Some(name) = r.as_str() {
                        if !map.contains_key(name) {
                            errors.push(format!("{path}: missing required field '{name}'"));
                        }
                    }
                }
            }
            if let Some(props) = schema_obj.get("properties").and_then(Value::as_object) {
                for (name, sub_schema) in props {
                    if let Some(child) = map.get(name) {
                        walk(child, sub_schema, &format!("{path}.{name}"), errors);
                    }
                }
            }
        }
        Value::Array(items) => {
            if let Some(item_schema) = schema_obj.get("items") {
                for (i, item) in items.iter().enumerate() {
                    walk(item, item_schema, &format!("{path}[{i}]"), errors);
                }
            }
        }
        Value::String(s) => {
            if let Some(min) = schema_obj.get("minLength").and_then(Value::as_u64) {
                if (s.chars().count() as u64) < min {
                    errors.push(format!("{path}: string too short (<{min})"));
                }
            }
            if let Some(max) = schema_obj.get("maxLength").and_then(Value::as_u64) {
                if (s.chars().count() as u64) > max {
                    errors.push(format!("{path}: string too long (>{max})"));
                }
            }
            if let Some(pat) = schema_obj.get("pattern").and_then(Value::as_str) {
                if !s.contains(pat) {
                    errors.push(format!("{path}: string does not contain '{pat}'"));
                }
            }
        }
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(0.0);
            if let Some(min) = schema_obj.get("minimum").and_then(Value::as_f64) {
                if f < min {
                    errors.push(format!("{path}: number below minimum ({f} < {min})"));
                }
            }
            if let Some(max) = schema_obj.get("maximum").and_then(Value::as_f64) {
                if f > max {
                    errors.push(format!("{path}: number above maximum ({f} > {max})"));
                }
            }
        }
        _ => {}
    }
}

fn type_matches(value: &Value, t: &Value) -> bool {
    let kind = describe_value_type(value);
    match t {
        Value::String(s) => kind_matches(kind, s),
        Value::Array(arr) => arr
            .iter()
            .filter_map(Value::as_str)
            .any(|s| kind_matches(kind, s)),
        _ => true,
    }
}

fn kind_matches(actual: &str, expected: &str) -> bool {
    if actual == expected {
        return true;
    }
    matches!((actual, expected), ("integer", "number"))
}

fn describe(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn describe_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_required_field_fails() {
        let schema = json!({
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}
        });
        let payload = json!({"name": "Ada"});
        let r = verify_schema(&payload, &schema);
        assert!(!r.passed);
        assert_eq!(r.failed_component.as_deref(), Some("schema"));
        assert!(r.errors.iter().any(|e| e.contains("'age'")));
    }

    #[test]
    fn type_mismatch_fails() {
        let schema = json!({"type": "object", "properties": {"age": {"type": "integer"}}});
        let payload = json!({"age": "old"});
        let r = verify_schema(&payload, &schema);
        assert!(!r.passed);
    }

    #[test]
    fn enum_constraint_fails() {
        let schema = json!({"type": "string", "enum": ["lite", "pro"]});
        let r = verify_schema(&json!("xxl"), &schema);
        assert!(!r.passed);
    }

    #[test]
    fn min_max_length_enforced() {
        let schema = json!({"type": "string", "minLength": 3, "maxLength": 5});
        assert!(verify_schema(&json!("ab"), &schema).passed.not());
        assert!(verify_schema(&json!("abcdef"), &schema).passed.not());
        assert!(verify_schema(&json!("abc"), &schema).passed);
    }

    #[test]
    fn integer_satisfies_number_type() {
        let schema = json!({"type": "number"});
        assert!(verify_schema(&json!(3), &schema).passed);
    }

    #[test]
    fn nested_array_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string", "minLength": 1}}
            }
        });
        assert!(verify_schema(&json!({"tags": ["a", "b"]}), &schema).passed);
        assert!(!verify_schema(&json!({"tags": ["a", ""]}), &schema).passed);
    }

    trait BoolNot {
        fn not(self) -> Self;
    }
    impl BoolNot for bool {
        fn not(self) -> bool {
            !self
        }
    }
}

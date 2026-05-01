//! Deterministic verifier skeletons (schema / budget / trace / tool_use /
//! unit_test).
//!
//! Phase 0 shipped only `BudgetVerifier`; Phase 3 fills in the rest. All
//! verifiers project onto [`flybrain_core::verify::VerificationResult`]
//! so the controller and trainer never need to know which verifier
//! produced a given verdict.

pub mod budget;
pub mod schema;
pub mod tool_use;
pub mod trace;
pub mod unit_test;

pub use budget::{BudgetState, BudgetVerifier};
pub use schema::verify_schema;
pub use tool_use::{verify_tool_calls, ToolUseSpec};
pub use trace::verify_trace;
pub use unit_test::verify_unit_tests;

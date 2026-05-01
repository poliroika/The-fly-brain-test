//! Deterministic verifier skeletons (schema / budget / trace / tool_use).
//!
//! Phase 0 ships only `BudgetVerifier` so the LLM client and tests have
//! something to call; the rest land in Phase 3.

pub mod budget;

pub use budget::{BudgetState, BudgetVerifier};

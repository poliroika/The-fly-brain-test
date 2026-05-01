//! Core types and serialization for FlyBrain Optimizer.
//!
//! This crate is the source-of-truth for all data contracts that flow between
//! Rust and Python. Every other crate (and the Python package) depends on
//! these types.

pub mod action;
pub mod agent;
pub mod error;
pub mod graph;
pub mod task;
pub mod trace;
pub mod verify;

pub use action::GraphAction;
pub use agent::{AgentSpec, ModelTier};
pub use error::{FlybrainError, FlybrainResult};
pub use graph::{AgentGraph, FlyGraph, NodeMetadata};
pub use task::{TaskSpec, TaskType};
pub use trace::{ToolCall, Totals, Trace, TraceStep};
pub use verify::VerificationResult;

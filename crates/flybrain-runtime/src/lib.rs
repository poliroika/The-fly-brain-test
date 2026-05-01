//! MAS runtime for the FlyBrain Optimizer.
//!
//! The runtime owns the *mutable* state of one MAS execution:
//!
//! * the `AgentGraph` being mutated by the controller,
//! * a per-agent message bus (in-memory, deterministic ordering),
//! * a step counter + an optional pending activation,
//! * a [`BudgetState`] tally fed by the Python agent layer,
//! * a [`TraceWriter`] that persists per-step records to JSONL.
//!
//! The Python side drives the loop tick-by-tick: it calls the controller
//! to pick a [`GraphAction`], hands it to the [`Scheduler`], runs an
//! agent (via [`Agent.step`] in Python) when the scheduler asks for one,
//! and feeds tool / verifier outcomes back. PyO3 wrappers in
//! `flybrain-py` expose [`MessageBus`], [`Scheduler`], and
//! [`TraceWriter`] to Python.

pub mod bus;
pub mod scheduler;
pub mod trace_writer;

pub use bus::{Message, MessageBus};
pub use scheduler::{ActiveAgent, Scheduler, SchedulerStatus};
pub use trace_writer::TraceWriter;

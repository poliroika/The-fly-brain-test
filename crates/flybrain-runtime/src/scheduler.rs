//! Tick-based scheduler that owns the dynamic [`AgentGraph`].
//!
//! The Python side runs a loop:
//!
//! ```text
//! loop {
//!     action = controller.select_action(state);
//!     status = scheduler.apply(action);  // returns SchedulerStatus
//!     match status {
//!         RunAgent(name) => python.agent[name].step(...),
//!         CallVerifier   => python.verifier.run(...),
//!         CallMemory | CallRetriever | CallToolExecutor => python.handle(...),
//!         GraphMutation  => continue,
//!         Terminated     => break,
//!     }
//!     scheduler.advance_step();
//! }
//! ```
//!
//! All graph mutations happen inside the scheduler so that
//! [`Scheduler::current_graph_hash`] is always in sync with what the trace
//! is about to record.

use flybrain_core::action::GraphAction;
use flybrain_core::error::{FlybrainError, FlybrainResult};
use flybrain_core::graph::AgentGraph;

/// What the runtime should do as a side-effect of the controller's action.
///
/// The Python loop dispatches on this; the scheduler itself does not call
/// any Python code.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerStatus {
    /// Controller wants `agent` to take a turn.
    RunAgent(String),
    /// Verifier should be invoked. The verifier outcome is fed back via
    /// [`Scheduler::record_verification`] (Python helper, not on this struct).
    CallVerifier,
    /// Controller asked for memory read.
    CallMemory,
    /// Controller asked for retriever.
    CallRetriever,
    /// Controller asked for the tool executor (tools picked from the
    /// pending agent's metadata in Python).
    CallToolExecutor,
    /// Action only mutated the graph; no side-effect agent should run.
    GraphMutation,
    /// Controller asked to terminate. After this, [`Scheduler::is_terminated`]
    /// returns true and further `apply` calls are rejected.
    Terminated,
}

/// Convenience newtype for "currently active agent". Stored separately so
/// the trace writer can stamp `active_agent` consistently even when the
/// controller emits a chain of pure-graph mutations.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveAgent(pub String);

#[derive(Debug, Clone)]
pub struct Scheduler {
    graph: AgentGraph,
    step_id: u64,
    /// Most recently activated agent (carried until next ActivateAgent).
    last_active: Option<ActiveAgent>,
    terminated: bool,
}

impl Scheduler {
    pub fn new(graph: AgentGraph) -> Self {
        Self {
            graph,
            step_id: 0,
            last_active: None,
            terminated: false,
        }
    }

    pub fn graph(&self) -> &AgentGraph {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut AgentGraph {
        &mut self.graph
    }

    pub fn current_graph_hash(&self) -> String {
        self.graph.hash()
    }

    pub fn step_id(&self) -> u64 {
        self.step_id
    }

    pub fn last_active_agent(&self) -> Option<&str> {
        self.last_active.as_ref().map(|a| a.0.as_str())
    }

    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Apply a controller action and return the side-effect dispatch.
    ///
    /// Mutates the underlying [`AgentGraph`] for `AddEdge` / `RemoveEdge` /
    /// `ScaleEdge`. For `ActivateAgent`, the named agent must already
    /// exist in the graph. After [`SchedulerStatus::Terminated`], further
    /// calls return `FlybrainError::InvalidAction`.
    pub fn apply(&mut self, action: &GraphAction) -> FlybrainResult<SchedulerStatus> {
        if self.terminated {
            return Err(FlybrainError::InvalidAction(
                "scheduler is terminated".into(),
            ));
        }

        match action {
            GraphAction::ActivateAgent { agent } => {
                if !self.graph.has_node(agent) {
                    return Err(FlybrainError::InvalidAction(format!(
                        "unknown agent {agent}"
                    )));
                }
                self.last_active = Some(ActiveAgent(agent.clone()));
                Ok(SchedulerStatus::RunAgent(agent.clone()))
            }
            GraphAction::AddEdge { .. }
            | GraphAction::RemoveEdge { .. }
            | GraphAction::ScaleEdge { .. } => {
                self.graph.apply(action)?;
                Ok(SchedulerStatus::GraphMutation)
            }
            GraphAction::CallMemory => Ok(SchedulerStatus::CallMemory),
            GraphAction::CallRetriever => Ok(SchedulerStatus::CallRetriever),
            GraphAction::CallToolExecutor => Ok(SchedulerStatus::CallToolExecutor),
            GraphAction::CallVerifier => Ok(SchedulerStatus::CallVerifier),
            GraphAction::Terminate => {
                self.terminated = true;
                Ok(SchedulerStatus::Terminated)
            }
        }
    }

    /// Advance the step counter. The Python loop calls this after writing
    /// a [`TraceStep`](flybrain_core::TraceStep) so that the next step has a
    /// fresh `step_id`.
    pub fn advance_step(&mut self) -> u64 {
        self.step_id += 1;
        self.step_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_5() -> AgentGraph {
        AgentGraph::with_nodes(["Planner", "Coder", "Verifier", "Memory", "Tool"])
    }

    #[test]
    fn activate_unknown_agent_is_error() {
        let mut s = Scheduler::new(graph_5());
        let r = s.apply(&GraphAction::ActivateAgent {
            agent: "Ghost".into(),
        });
        assert!(matches!(r, Err(FlybrainError::InvalidAction(_))));
    }

    #[test]
    fn activate_known_agent_runs_it() {
        let mut s = Scheduler::new(graph_5());
        let r = s.apply(&GraphAction::ActivateAgent {
            agent: "Planner".into(),
        });
        assert_eq!(r.unwrap(), SchedulerStatus::RunAgent("Planner".into()));
        assert_eq!(s.last_active_agent(), Some("Planner"));
    }

    #[test]
    fn add_edge_mutates_graph_hash() {
        let mut s = Scheduler::new(graph_5());
        let h0 = s.current_graph_hash();
        let r = s
            .apply(&GraphAction::AddEdge {
                from: "Planner".into(),
                to: "Coder".into(),
                weight: 1.0,
            })
            .unwrap();
        assert_eq!(r, SchedulerStatus::GraphMutation);
        assert_ne!(s.current_graph_hash(), h0);
    }

    #[test]
    fn terminate_blocks_future_actions() {
        let mut s = Scheduler::new(graph_5());
        let r = s.apply(&GraphAction::Terminate).unwrap();
        assert_eq!(r, SchedulerStatus::Terminated);
        assert!(s.is_terminated());

        let again = s.apply(&GraphAction::CallVerifier);
        assert!(matches!(again, Err(FlybrainError::InvalidAction(_))));
    }

    #[test]
    fn advance_step_increments_counter() {
        let mut s = Scheduler::new(graph_5());
        assert_eq!(s.step_id(), 0);
        s.advance_step();
        s.advance_step();
        assert_eq!(s.step_id(), 2);
    }

    #[test]
    fn graph_mutation_does_not_change_active_agent() {
        let mut s = Scheduler::new(graph_5());
        s.apply(&GraphAction::ActivateAgent {
            agent: "Planner".into(),
        })
        .unwrap();
        s.apply(&GraphAction::AddEdge {
            from: "Planner".into(),
            to: "Coder".into(),
            weight: 1.0,
        })
        .unwrap();
        assert_eq!(s.last_active_agent(), Some("Planner"));
    }
}

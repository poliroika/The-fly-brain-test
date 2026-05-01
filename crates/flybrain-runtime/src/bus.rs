//! In-memory message bus for inter-agent communication.
//!
//! The bus is intentionally small: it is a `BTreeMap<recipient, VecDeque<Message>>`
//! plus a global ordered log used as `input_msg_id` references in
//! [`TraceStep`](flybrain_core::TraceStep). Deterministic for a fixed
//! insertion order.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// One message routed between agents (or from the runtime to an agent).
///
/// `id` is monotonic across the bus instance and is what
/// `TraceStep::input_msg_id` references.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: u64,
    pub sender: String,
    pub recipient: String,
    pub content: serde_json::Value,
    pub step_id: u64,
}

#[derive(Debug, Default, Clone)]
pub struct MessageBus {
    next_id: u64,
    queues: BTreeMap<String, VecDeque<Message>>,
    /// Append-only log of every message ever sent on this bus.
    log: Vec<Message>,
}

impl MessageBus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Send a message and return the assigned id. Idempotent (the same
    /// payload sent twice produces two independent messages).
    pub fn send(
        &mut self,
        sender: impl Into<String>,
        recipient: impl Into<String>,
        content: serde_json::Value,
        step_id: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let msg = Message {
            id,
            sender: sender.into(),
            recipient: recipient.into(),
            content,
            step_id,
        };
        self.queues
            .entry(msg.recipient.clone())
            .or_default()
            .push_back(msg.clone());
        self.log.push(msg);
        id
    }

    /// Pop the next pending message for `recipient` (FIFO). Returns
    /// `None` if the inbox is empty.
    pub fn pop(&mut self, recipient: &str) -> Option<Message> {
        self.queues.get_mut(recipient).and_then(|q| q.pop_front())
    }

    /// How many messages are waiting for `recipient`.
    pub fn pending(&self, recipient: &str) -> usize {
        self.queues.get(recipient).map(|q| q.len()).unwrap_or(0)
    }

    /// Total messages routed (including drained ones).
    pub fn total(&self) -> usize {
        self.log.len()
    }

    /// Read-only access to the entire log (useful for trace reconstruction).
    pub fn log(&self) -> &[Message] {
        &self.log
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fifo_ordering() {
        let mut bus = MessageBus::new();
        let a = bus.send("Planner", "Coder", json!({"text": "first"}), 0);
        let b = bus.send("Planner", "Coder", json!({"text": "second"}), 0);
        assert!(a < b);

        assert_eq!(bus.pop("Coder").unwrap().id, a);
        assert_eq!(bus.pop("Coder").unwrap().id, b);
        assert!(bus.pop("Coder").is_none());
    }

    #[test]
    fn pop_unknown_recipient_is_none() {
        let mut bus = MessageBus::new();
        assert!(bus.pop("ghost").is_none());
    }

    #[test]
    fn log_records_every_send() {
        let mut bus = MessageBus::new();
        bus.send("A", "B", json!(1), 0);
        bus.send("A", "C", json!(2), 0);
        assert_eq!(bus.total(), 2);
        assert_eq!(bus.log()[1].recipient, "C");
    }

    #[test]
    fn pending_count_reflects_drains() {
        let mut bus = MessageBus::new();
        bus.send("A", "B", json!(1), 0);
        bus.send("A", "B", json!(2), 0);
        assert_eq!(bus.pending("B"), 2);
        bus.pop("B");
        assert_eq!(bus.pending("B"), 1);
    }
}

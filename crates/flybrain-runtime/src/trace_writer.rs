//! Streaming JSONL trace writer.
//!
//! Each call to [`TraceWriter::record_step`] appends a [`TraceStep`] to the
//! in-memory trace and (if a sink path is configured) writes one
//! line of JSON per step to disk. [`TraceWriter::finalize`] sets the
//! optional `final_answer` / `verification`, drops the file handle, and
//! returns the fully-realised [`Trace`] which the caller can persist
//! separately as one JSON document.
//!
//! Two persisted artefacts coexist by design:
//!
//! * `<task_id>.steps.jsonl` — one [`TraceStep`] per line, written
//!   incrementally so a long run can be inspected mid-flight.
//! * `<task_id>.trace.json` — full [`Trace`] document, written by the
//!   caller after [`TraceWriter::finalize`].

use flybrain_core::error::FlybrainResult;
use flybrain_core::task::TaskType;
use flybrain_core::trace::{Totals, Trace, TraceStep};
use flybrain_core::verify::VerificationResult;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct TraceWriter {
    trace: Trace,
    sink: Option<BufWriter<File>>,
    sink_path: Option<PathBuf>,
}

impl TraceWriter {
    /// Open a trace writer for `task_id`. If `path` is `Some`, every
    /// `record_step` call appends a JSONL line to that file.
    pub fn open(
        task_id: impl Into<String>,
        task_type: TaskType,
        path: Option<&Path>,
    ) -> FlybrainResult<Self> {
        let trace = Trace::new(task_id, task_type);
        let (sink, sink_path) = match path {
            Some(p) => {
                if let Some(parent) = p.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                let file = OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(p)?;
                (Some(BufWriter::new(file)), Some(p.to_path_buf()))
            }
            None => (None, None),
        };
        Ok(Self {
            trace,
            sink,
            sink_path,
        })
    }

    pub fn task_id(&self) -> &str {
        &self.trace.task_id
    }

    pub fn task_type(&self) -> TaskType {
        self.trace.task_type
    }

    pub fn step_count(&self) -> usize {
        self.trace.steps.len()
    }

    pub fn sink_path(&self) -> Option<&Path> {
        self.sink_path.as_deref()
    }

    /// Append a step to the in-memory trace and (if a sink is configured)
    /// flush a JSONL line to disk.
    pub fn record_step(&mut self, step: TraceStep) -> FlybrainResult<()> {
        // Aggregate totals as we go so callers don't have to do a final pass.
        self.trace.totals.tokens_in += step.tokens_in;
        self.trace.totals.tokens_out += step.tokens_out;
        self.trace.totals.latency_ms += step.latency_ms;
        self.trace.totals.cost_rub += step.cost_rub;
        if !step.tool_calls.is_empty() {
            self.trace.totals.tool_calls += step.tool_calls.len() as u32;
            self.trace.totals.failed_tool_calls +=
                step.tool_calls.iter().filter(|t| !t.ok).count() as u32;
        }
        if step.tokens_in > 0 || step.tokens_out > 0 {
            self.trace.totals.llm_calls += 1;
        }

        if let Some(sink) = self.sink.as_mut() {
            let line = serde_json::to_string(&step)?;
            sink.write_all(line.as_bytes())?;
            sink.write_all(b"\n")?;
            sink.flush()?;
        }

        self.trace.steps.push(step);
        Ok(())
    }

    /// Replace the running [`Totals`] with `totals`. Useful when the
    /// caller is tracking budget elsewhere and wants the trace to mirror it.
    pub fn set_totals(&mut self, totals: Totals) {
        self.trace.totals = totals;
    }

    pub fn totals(&self) -> &Totals {
        &self.trace.totals
    }

    /// Finalise the trace. Closes the JSONL sink if any. The returned
    /// [`Trace`] is owned by the caller, who should typically persist it
    /// as `<task_id>.trace.json`.
    pub fn finalize(
        mut self,
        final_answer: Option<String>,
        verification: Option<VerificationResult>,
        metadata: serde_json::Value,
    ) -> FlybrainResult<Trace> {
        if let Some(mut sink) = self.sink.take() {
            sink.flush()?;
        }
        self.trace.final_answer = final_answer;
        self.trace.verification = verification;
        self.trace.metadata = metadata;
        Ok(self.trace)
    }

    /// Snapshot the in-memory trace without finalising. Useful for
    /// debugging and tests.
    pub fn snapshot(&self) -> Trace {
        self.trace.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flybrain_core::action::GraphAction;
    use flybrain_core::task::TaskType;
    use flybrain_core::trace::{ToolCall, TraceStep};
    use serde_json::json;

    fn step(step_id: u64, tokens_in: u64, tokens_out: u64) -> TraceStep {
        TraceStep {
            step_id,
            t_unix_ms: 0,
            active_agent: Some("Coder".into()),
            input_msg_id: None,
            output_summary: None,
            tool_calls: vec![],
            errors: vec![],
            tokens_in,
            tokens_out,
            latency_ms: 5,
            verifier_score: None,
            current_graph_hash: "deadbeef".into(),
            graph_action: GraphAction::ActivateAgent {
                agent: "Coder".into(),
            },
            cost_rub: 0.01,
        }
    }

    #[test]
    fn record_step_aggregates_totals() {
        let mut w = TraceWriter::open("t1", TaskType::Coding, None).unwrap();
        w.record_step(step(0, 100, 50)).unwrap();
        w.record_step(step(1, 30, 20)).unwrap();
        let totals = w.totals();
        assert_eq!(totals.tokens_in, 130);
        assert_eq!(totals.tokens_out, 70);
        assert_eq!(totals.llm_calls, 2);
    }

    #[test]
    fn record_step_persists_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traces").join("t1.steps.jsonl");
        let mut w = TraceWriter::open("t1", TaskType::Coding, Some(&path)).unwrap();
        w.record_step(step(0, 10, 5)).unwrap();
        w.record_step(step(1, 0, 0)).unwrap();
        let _ = w.finalize(Some("done".into()), None, json!({})).unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let parsed: TraceStep = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.step_id, 0);
    }

    #[test]
    fn finalize_attaches_final_answer_and_metadata() {
        let mut w = TraceWriter::open("t1", TaskType::Math, None).unwrap();
        w.record_step(step(0, 10, 5)).unwrap();
        let t = w
            .finalize(Some("42".into()), None, json!({"controller": "manual"}))
            .unwrap();
        assert_eq!(t.final_answer.as_deref(), Some("42"));
        assert_eq!(t.metadata["controller"], "manual");
        assert_eq!(t.steps.len(), 1);
    }

    #[test]
    fn tool_call_failures_count_separately() {
        let mut w = TraceWriter::open("t1", TaskType::Coding, None).unwrap();
        let mut s = step(0, 0, 0);
        s.tool_calls = vec![
            ToolCall {
                name: "python_exec".into(),
                args: json!({}),
                ok: true,
                error: None,
                latency_ms: 1,
            },
            ToolCall {
                name: "web_search".into(),
                args: json!({}),
                ok: false,
                error: Some("timeout".into()),
                latency_ms: 30000,
            },
        ];
        w.record_step(s).unwrap();
        assert_eq!(w.totals().tool_calls, 2);
        assert_eq!(w.totals().failed_tool_calls, 1);
    }
}

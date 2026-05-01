"""Verification layer.

Deterministic verifiers (schema, tool_use, budget, trace) live in Rust under
`flybrain_native.verify`. LLM-judge verifiers (factual, reasoning) live here
under `flybrain.verification.llm`.

Implementations land in Phase 3."""

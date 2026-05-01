"""Validate that every Phase-2 `AgentSpec` round-trips through the Rust
`AgentSpec` schema and that the lite/Pro tier mapping matches
`configs/llm/yandex.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flybrain.agents import EXTENDED_25, MINIMAL_15, load_minimal_15

native = pytest.importorskip("flybrain.flybrain_native")

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_minimal_15_has_15_specs() -> None:
    assert len(MINIMAL_15) == 15
    names = [s.name for s in MINIMAL_15]
    expected = [
        "Planner",
        "TaskDecomposer",
        "Coder",
        "Debugger",
        "TestRunner",
        "MathSolver",
        "Retriever",
        "MemoryReader",
        "MemoryWriter",
        "ToolExecutor",
        "SchemaValidator",
        "Verifier",
        "Critic",
        "Judge",
        "Finalizer",
    ]
    assert names == expected


def test_extended_25_has_25_specs() -> None:
    assert len(EXTENDED_25) == 25
    extras = {s.name for s in EXTENDED_25} - {s.name for s in MINIMAL_15}
    assert extras == {
        "Refiner",
        "SearchAgent",
        "Researcher",
        "ContextCompressor",
        "CitationChecker",
        "ConstraintChecker",
        "FailureRecovery",
        "BudgetController",
        "ProofChecker",
        "SafetyFilter",
    }


def test_specs_have_non_empty_prompts() -> None:
    for spec in EXTENDED_25:
        assert spec.system_prompt.strip(), f"{spec.name} has empty prompt"
        assert spec.role, f"{spec.name} has empty role"


def test_specs_round_trip_through_native() -> None:
    for spec in EXTENDED_25:
        out = native.agent_spec_round_trip(spec.to_dict())
        assert out["name"] == spec.name
        assert out["model_tier"] == spec.model_tier
        assert out["system_prompt"] == spec.system_prompt
        assert out["tools"] == spec.tools


def test_tier_mapping_matches_yandex_config() -> None:
    cfg_path = _REPO_ROOT / "configs" / "llm" / "yandex.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    expected = cfg["agent_to_model"]
    for spec in EXTENDED_25:
        if spec.name in expected:
            assert spec.model_tier == expected[spec.name], (
                f"{spec.name} tier mismatch: spec={spec.model_tier} cfg={expected[spec.name]}"
            )


def test_load_minimal_15_returns_a_copy() -> None:
    a = load_minimal_15()
    b = load_minimal_15()
    a.append(b[0])
    assert len(b) == 15  # mutating one copy must not affect the other

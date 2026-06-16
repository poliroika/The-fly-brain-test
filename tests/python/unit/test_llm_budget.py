"""Tests for the Python BudgetTracker."""

from __future__ import annotations

import pytest

from flybrain.llm import BudgetTracker
from flybrain.llm.budget import BudgetExceededError


def test_initial_state() -> None:
    b = BudgetTracker(hard_cap_rub=2000.0)
    assert b.cost_rub == 0
    assert b.remaining_rub == 2000.0


def test_record_increments_counters() -> None:
    b = BudgetTracker(hard_cap_rub=2000.0)
    b.record(tokens_in=100, tokens_out=50, cost_rub=5.0)
    assert b.tokens_in == 100
    assert b.tokens_out == 50
    assert b.llm_calls == 1
    assert b.cost_rub == pytest.approx(5.0)


def test_reserve_raises_over_hard_cap() -> None:
    b = BudgetTracker(hard_cap_rub=10.0)
    b.record(tokens_in=100, tokens_out=50, cost_rub=8.0)
    with pytest.raises(BudgetExceededError):
        b.reserve(5.0)


def test_soft_cap_emits_warning() -> None:
    b = BudgetTracker(hard_cap_rub=100.0)
    b.record(tokens_in=1000, tokens_out=500, cost_rub=85.0)
    assert any("soft cap" in w for w in b.warnings)


def test_will_exceed_does_not_mutate() -> None:
    b = BudgetTracker(hard_cap_rub=10.0)
    assert b.will_exceed(20.0)
    assert b.cost_rub == 0
    assert b.llm_calls == 0


def test_snapshot_keys() -> None:
    b = BudgetTracker(hard_cap_rub=2000.0)
    b.record(tokens_in=100, tokens_out=50, cost_rub=5.0)
    snap = b.snapshot()
    assert set(snap.keys()) == {"tokens_in", "tokens_out", "llm_calls", "cost_rub", "remaining_rub"}

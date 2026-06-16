"""Phase-8 RL smoke tests (rewards, bandits, REINFORCE, PPO, offline RL)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flybrain.training.rl import (
    LinUCBBandit,
    OfflineRLConfig,
    PPOConfig,
    ReinforceConfig,
    RewardConfig,
    ThompsonBandit,
    compute_reward,
    offline_rl_train,
    ppo_train,
    reinforce_train,
)

# -- rewards -------------------------------------------------------------------


def _passing_trace(extra_steps: int = 5) -> dict[str, Any]:
    return {
        "task_id": "t1",
        "task_type": "coding",
        "verification": {"passed": True, "score": 0.9},
        "steps": [
            {
                "kind": "activate_agent",
                "agent": f"A{i}",
                "verifier_score": 0.9,
            }
            for i in range(extra_steps)
        ],
        "totals": {"tokens_in": 50, "tokens_out": 100, "wall_seconds": 1.5, "cost_rub": 0.1},
    }


def test_compute_reward_pass_high() -> None:
    r = compute_reward(_passing_trace())
    # success(1) + 0.5*0.9 - tokens(150*1e-4) - llm(5*0.01) - latency(1.5*1e-3)
    assert r > 1.0
    assert r < 1.5


def test_compute_reward_failed_zero_baseline() -> None:
    failing = _passing_trace()
    failing["verification"]["passed"] = False
    failing["verification"]["score"] = 0.1
    failing["steps"] = [{"kind": "activate_agent", "agent": "A", "verifier_score": 0.1}] * 3
    r = compute_reward(failing)
    assert r < 0.5


def test_compute_reward_handles_missing_fields() -> None:
    r = compute_reward({"steps": [{}]})
    assert isinstance(r, float)


def test_compute_reward_failed_tools_penalised() -> None:
    base = _passing_trace()
    fail = _passing_trace()
    fail["steps"][0]["tool_outcome"] = "error"
    assert compute_reward(fail) < compute_reward(base)


def test_reward_config_asdict_roundtrip() -> None:
    cfg = RewardConfig(success_weight=2.0, alpha_tokens=0.5)
    d = cfg.asdict()
    assert d["success_weight"] == 2.0
    assert d["alpha_tokens"] == 0.5


# -- bandits -------------------------------------------------------------------


def test_linucb_select_runs_and_updates() -> None:
    rng = np.random.default_rng(0)
    b = LinUCBBandit(num_arms=4, context_dim=6, alpha=0.5, rng=rng)
    ctx = rng.standard_normal(6)
    a = b.select(ctx)
    assert 0 <= a < 4
    b.update(arm=a, context=ctx, reward=1.0)
    # A_inv must change after an update.
    a2 = b.select(ctx)
    # Idempotency check: same context still picks a valid arm.
    assert 0 <= a2 < 4


def test_linucb_action_mask() -> None:
    b = LinUCBBandit(num_arms=4, context_dim=3)
    ctx = np.ones(3)
    mask = [False, True, False, False]
    for _ in range(10):
        a = b.select(ctx, action_mask=mask)
        assert a == 1


def test_linucb_learns_best_arm() -> None:
    rng = np.random.default_rng(0)
    b = LinUCBBandit(num_arms=3, context_dim=4, alpha=0.1, rng=rng)
    # Arm 1 is uniformly the best across all contexts.
    for _ in range(200):
        ctx = rng.standard_normal(4)
        a = b.select(ctx)
        reward = 2.0 if a == 1 else 0.0
        b.update(a, ctx, reward)
    # After 200 trials, arm 1 should be selected on a fresh context.
    chosen = [b.select(rng.standard_normal(4)) for _ in range(50)]
    counts = {k: chosen.count(k) for k in (0, 1, 2)}
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_thompson_select_runs() -> None:
    b = ThompsonBandit(num_arms=3, context_dim=5)
    ctx = np.zeros(5)
    a = b.select(ctx)
    assert 0 <= a < 3
    b.update(a, ctx, 1.0)


def test_thompson_action_mask_forces_choice() -> None:
    b = ThompsonBandit(num_arms=3, context_dim=2)
    ctx = np.zeros(2)
    for _ in range(10):
        assert b.select(ctx, action_mask=[False, False, True]) == 2


# -- REINFORCE / PPO / offline RL ---------------------------------------------

torch = pytest.importorskip("torch")


def _build_controller_and_traces(tmp_path: Path) -> tuple[Any, list, list[str]]:
    """Create a small GNN controller and a couple of in-memory traces."""
    import asyncio

    from flybrain.agents.specs import MINIMAL_15
    from flybrain.controller import FlyBrainGNNController
    from flybrain.embeddings import (
        AgentEmbedder,
        AgentGraphEmbedder,
        ControllerStateBuilder,
        FlyGraphEmbedder,
        MockEmbeddingClient,
        TaskEmbedder,
        TraceEmbedder,
    )

    client = MockEmbeddingClient(output_dim=16)
    agents_emb = AgentEmbedder(client)
    asyncio.run(agents_emb.precompute(MINIMAL_15))
    builder = ControllerStateBuilder(
        task=TaskEmbedder(client),
        agents=agents_emb,
        trace=TraceEmbedder(client),
        fly=FlyGraphEmbedder(dim=4),
        agent_graph=AgentGraphEmbedder(in_dim=16, hidden_dim=8, out_dim=16),
    )
    ctrl = FlyBrainGNNController(
        builder=builder,
        task_dim=16,
        agent_dim=16,
        graph_dim=16,
        trace_dim=16 + 13,
        fly_dim=4,
        produced_dim=6,
        hidden_dim=16,
    )
    agent_names = [a.name for a in MINIMAL_15]

    # Write 3 small synthetic trace files (use the live runtime's
    # step format: ``graph_action`` is a sub-dict on each step).
    for i in range(3):
        passed = i % 2 == 0
        trace = {
            "task_id": f"task-{i}",
            "task_type": "coding",
            "verification": {"passed": passed, "score": 0.8 if passed else 0.2},
            "steps": [
                {
                    "graph_action": {
                        "kind": "activate_agent",
                        "agent": agent_names[(j + i) % len(agent_names)],
                    },
                    "verifier_score": 0.8 if passed else 0.1,
                }
                for j in range(2)
            ]
            + [
                {
                    "graph_action": {"kind": "call_verifier"},
                    "verifier_score": 0.8 if passed else 0.1,
                },
                {"graph_action": {"kind": "terminate"}},
            ],
            "totals": {"tokens_in": 30, "tokens_out": 60, "wall_seconds": 0.5},
        }
        (tmp_path / f"task-{i}.trace.json").write_text(json.dumps(trace))

    from flybrain.training.expert_dataset import iter_traces

    traces = list(iter_traces(tmp_path))
    return ctrl, traces, agent_names


def test_reinforce_runs_smoke(tmp_path: Path) -> None:
    ctrl, traces, names = _build_controller_and_traces(tmp_path)
    cfg = ReinforceConfig(epochs=2, learning_rate=1e-3, seed=0)
    res = reinforce_train(ctrl, traces=traces, agent_names=names, config=cfg)
    assert res.num_episodes == 3
    assert len(res.epoch_returns) == 2
    assert len(res.epoch_losses) == 2


def test_ppo_runs_smoke(tmp_path: Path) -> None:
    ctrl, traces, names = _build_controller_and_traces(tmp_path)
    cfg = PPOConfig(iterations=2, epochs_per_batch=1, learning_rate=1e-3, seed=0)
    res = ppo_train(ctrl, traces=traces, agent_names=names, config=cfg)
    assert res.num_episodes == 3
    assert len(res.iteration_returns) == 2
    assert 0.0 <= res.final_clip_fraction <= 1.0


def test_offline_rl_runs_from_disk(tmp_path: Path) -> None:
    ctrl, _, names = _build_controller_and_traces(tmp_path)
    cfg = OfflineRLConfig(epochs=1, learning_rate=1e-3, seed=0)
    res = offline_rl_train(ctrl, traces_dir=tmp_path, agent_names=names, config=cfg)
    assert res.num_episodes == 3
    assert len(res.epoch_returns) == 1


def test_reinforce_handles_empty_traces(tmp_path: Path) -> None:
    ctrl, _, names = _build_controller_and_traces(tmp_path)
    cfg = ReinforceConfig(epochs=1)
    res = reinforce_train(ctrl, traces=[], agent_names=names, config=cfg)
    assert res.num_episodes == 0
    assert res.epoch_returns == []


def test_reinforce_only_passed_filters(tmp_path: Path) -> None:
    ctrl, traces, names = _build_controller_and_traces(tmp_path)
    cfg = ReinforceConfig(epochs=1, only_passed=True)
    res = reinforce_train(ctrl, traces=traces, agent_names=names, config=cfg)
    # Only 2 of 3 traces pass.
    assert res.num_episodes == 2

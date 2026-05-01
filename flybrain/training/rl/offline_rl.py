"""Offline RL re-training from saved traces (README §12.4).

This is a thin wrapper around :func:`reinforce_train` that loads
traces from disk and runs REINFORCE without ever touching the LLM
runtime. The difference from :func:`flybrain.training.imitation_train`
is the *signal*: imitation maximises log-likelihood of the expert's
action; offline RL maximises ``log_pi * reward`` so the controller
learns to mimic *good* expert traces while down-weighting bad ones.

Usage::

    cfg = OfflineRLConfig(epochs=8, only_passed=False)
    result = offline_rl_train(
        controller, traces_dir=Path("data/traces/v1"),
        agent_names=names, config=cfg,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flybrain.training.rl.reinforce import (
    ReinforceConfig,
    ReinforceResult,
    reinforce_train,
)
from flybrain.training.rl.rewards import RewardConfig


@dataclass(slots=True)
class OfflineRLConfig:
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0
    only_passed: bool = False
    reward: RewardConfig = field(default_factory=RewardConfig)
    use_value_baseline: bool = True
    entropy_bonus: float = 1e-3


@dataclass(slots=True)
class OfflineRLResult:
    epoch_returns: list[float]
    epoch_losses: list[float]
    epoch_entropy: list[float]
    num_episodes: int
    traces_dir: str


def offline_rl_train(
    controller: Any,
    *,
    traces_dir: Path,
    agent_names: list[str],
    config: OfflineRLConfig | None = None,
) -> OfflineRLResult:
    """Run offline REINFORCE over every ``*.trace.json`` under
    ``traces_dir``. Cheap (no LLM calls) and idempotent — safe to
    re-run after each Phase-7 collection."""
    from flybrain.training.expert_dataset import iter_traces

    cfg = config or OfflineRLConfig()
    rcfg = ReinforceConfig(
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        use_value_baseline=cfg.use_value_baseline,
        entropy_bonus=cfg.entropy_bonus,
        reward=cfg.reward,
        only_passed=cfg.only_passed,
    )
    traces = list(iter_traces(Path(traces_dir)))
    res: ReinforceResult = reinforce_train(
        controller,
        traces=traces,
        agent_names=agent_names,
        config=rcfg,
    )
    return OfflineRLResult(
        epoch_returns=res.epoch_returns,
        epoch_losses=res.epoch_losses,
        epoch_entropy=res.epoch_entropy,
        num_episodes=res.num_episodes,
        traces_dir=str(traces_dir),
    )


__all__ = ["OfflineRLConfig", "OfflineRLResult", "offline_rl_train"]

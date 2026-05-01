"""Clipped-objective PPO over the Phase-5 controllers.

We treat each saved trace as one episode (same as REINFORCE) but
cache the *old* policy's per-step log-probabilities up front so the
surrogate objective::

    L^{CLIP} = E_t [ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]

with ``r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)`` matches the
standard PPO formulation. Multiple optimisation epochs over the same
batch of trajectories are supported (``epochs_per_batch``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from flybrain.training.expert_dataset import TraceFile

from flybrain.training.rl.rewards import RewardConfig, compute_reward


@dataclass(slots=True)
class PPOConfig:
    iterations: int = 5
    """Number of (collect → optimise) iterations."""
    epochs_per_batch: int = 4
    """Inner-loop optimiser epochs per collected batch."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    clip_eps: float = 0.2
    value_loss_weight: float = 0.5
    entropy_bonus: float = 0.01
    seed: int = 0
    only_passed: bool = False
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(slots=True)
class PPOResult:
    iteration_returns: list[float]
    iteration_losses: list[float]
    final_clip_fraction: float
    num_episodes: int


def _collect_old_logps(
    controller: Any,
    bundles: list[tuple],
) -> list[list]:
    """For each example in each episode, compute and cache the
    *current* policy's log-prob and value. These become the "old"
    references for the PPO ratio."""
    import torch

    out: list[list] = []
    for _, examples in bundles:
        episode = []
        with torch.no_grad():
            for ex in examples:
                cs = controller.builder.from_runtime_sync(ex.runtime_state)
                head = controller(cs)
                kind_logp = torch.log_softmax(head.kind_logits, dim=-1)[ex.label_kind]
                step_logp = float(kind_logp.item())
                if (
                    ex.label_agent is not None
                    and head.agent_logits.numel() > 0
                    and 0 <= ex.label_agent < int(head.agent_logits.shape[-1])
                ):
                    agent_logp = torch.log_softmax(head.agent_logits, dim=-1)[ex.label_agent]
                    step_logp += float(agent_logp.item())
                episode.append((step_logp, float(head.value.item())))
        out.append(episode)
    return out


def ppo_train(
    controller: Any,
    *,
    traces: Iterable[TraceFile],
    agent_names: list[str],
    config: PPOConfig | None = None,
) -> PPOResult:
    """Run clipped-PPO on a fixed offline corpus of traces. Each
    iteration re-uses the same ``traces`` (we don't re-collect from
    LLM calls — that's offline RL by construction).

    Returns per-iteration return + loss curves."""
    import torch
    import torch.nn.functional as F

    cfg = config or PPOConfig()
    torch.manual_seed(cfg.seed)

    from flybrain.training.expert_dataset import trace_to_examples

    bundles = []
    for tr in traces:
        if cfg.only_passed and not tr.verification_passed:
            continue
        episode_examples = trace_to_examples(tr, agent_names=agent_names)
        if episode_examples:
            bundles.append((tr, episode_examples))
    if not bundles:
        return PPOResult(
            iteration_returns=[],
            iteration_losses=[],
            final_clip_fraction=0.0,
            num_episodes=0,
        )

    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    iter_returns: list[float] = []
    iter_losses: list[float] = []
    last_clip_fraction = 0.0

    for _ in range(cfg.iterations):
        old_logps = _collect_old_logps(controller, bundles)

        avg_return = 0.0
        for trace, _ in bundles:
            avg_return += compute_reward(trace.raw, cfg.reward)
        avg_return /= len(bundles)
        iter_returns.append(avg_return)

        iter_loss_total = 0.0
        clipped_steps = 0
        total_steps = 0

        for _ in range(cfg.epochs_per_batch):
            for episode_idx, (trace, examples) in enumerate(bundles):
                episode_return = compute_reward(trace.raw, cfg.reward)

                policy_loss = torch.zeros((), dtype=torch.float32)
                value_loss = torch.zeros((), dtype=torch.float32)
                entropy_term = torch.zeros((), dtype=torch.float32)

                for step_idx, ex in enumerate(examples):
                    cs = controller.builder.from_runtime_sync(ex.runtime_state)
                    head = controller(cs)
                    kind_logp_full = F.log_softmax(head.kind_logits, dim=-1)
                    step_logp = kind_logp_full[ex.label_kind]
                    step_entropy = -(F.softmax(head.kind_logits, dim=-1) * kind_logp_full).sum()
                    if (
                        ex.label_agent is not None
                        and head.agent_logits.numel() > 0
                        and 0 <= ex.label_agent < int(head.agent_logits.shape[-1])
                    ):
                        agent_logp_full = F.log_softmax(head.agent_logits, dim=-1)
                        step_logp = step_logp + agent_logp_full[ex.label_agent]
                        step_entropy = (
                            step_entropy
                            - (F.softmax(head.agent_logits, dim=-1) * agent_logp_full).sum()
                        )

                    old_logp, old_value = old_logps[episode_idx][step_idx]
                    ratio = torch.exp(step_logp - old_logp)
                    advantage = episode_return - old_value

                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage
                    surrogate = -torch.min(unclipped, clipped)
                    policy_loss = policy_loss + surrogate

                    if abs(float(ratio.detach()) - 1.0) > cfg.clip_eps:
                        clipped_steps += 1
                    total_steps += 1

                    value_loss = value_loss + F.mse_loss(
                        head.value,
                        torch.tensor(episode_return, dtype=head.value.dtype),
                    )
                    entropy_term = entropy_term + step_entropy

                n = max(1, len(examples))
                policy_loss = policy_loss / n
                value_loss = value_loss / n
                entropy_term = entropy_term / n
                loss = (
                    policy_loss
                    + cfg.value_loss_weight * value_loss
                    - cfg.entropy_bonus * entropy_term
                )

                optimizer.zero_grad()
                if loss.requires_grad and not (math.isnan(loss.item()) or math.isinf(loss.item())):
                    loss.backward()
                    optimizer.step()
                iter_loss_total += float(loss.detach())

        iter_losses.append(iter_loss_total / max(1, len(bundles)))
        last_clip_fraction = clipped_steps / max(1, total_steps)

    return PPOResult(
        iteration_returns=iter_returns,
        iteration_losses=iter_losses,
        final_clip_fraction=last_clip_fraction,
        num_episodes=len(bundles),
    )


__all__ = ["PPOConfig", "PPOResult", "ppo_train"]

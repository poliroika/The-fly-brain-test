"""Vanilla policy-gradient (REINFORCE) for the Phase-5 controllers.

We treat each saved trace as one *episode*: the controller's
sequence of step-level actions is the action trajectory, the
trace-level :func:`compute_reward` is the (single, terminal) reward.

The episode return ``G`` is plugged into the per-step REINFORCE
update::

    L = - sum_t  log pi(a_t | s_t) * (G - V(s_t))

where ``V`` is the controller's existing value head, used as a
state-dependent baseline. The aux verifier head is left untouched
here — it's already supervised in Phase 7.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from flybrain.training.expert_dataset import ImitationExample, TraceFile

from flybrain.training.rl.rewards import RewardConfig, compute_reward


@dataclass(slots=True)
class ReinforceConfig:
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0
    use_value_baseline: bool = True
    value_loss_weight: float = 0.5
    entropy_bonus: float = 1e-3
    """Coefficient on the per-step entropy regulariser."""
    reward: RewardConfig = field(default_factory=RewardConfig)
    only_passed: bool = False


@dataclass(slots=True)
class ReinforceResult:
    epoch_returns: list[float]
    """Mean trace-level reward per epoch."""
    epoch_losses: list[float]
    epoch_entropy: list[float]
    num_episodes: int


def _trace_examples(
    traces: list[TraceFile],
    agent_names: list[str],
    only_passed: bool,
) -> list[tuple[TraceFile, list[ImitationExample]]]:
    from flybrain.training.expert_dataset import trace_to_examples

    out: list[tuple[TraceFile, list[ImitationExample]]] = []
    for trace in traces:
        if only_passed and not trace.verification_passed:
            continue
        examples = trace_to_examples(trace, agent_names=agent_names)
        if examples:
            out.append((trace, examples))
    return out


def reinforce_train(
    controller: Any,
    *,
    traces: Iterable[TraceFile],
    agent_names: list[str],
    config: ReinforceConfig | None = None,
) -> ReinforceResult:
    """Optimise ``controller`` with REINFORCE on a corpus of traces.

    ``traces`` is any iterable of :class:`TraceFile` instances (from
    :func:`flybrain.training.expert_dataset.iter_traces`). The
    function never makes any LLM calls — the rewards come from the
    trace's recorded ``totals`` + ``verification`` blocks.
    """
    import torch
    import torch.nn.functional as F

    cfg = config or ReinforceConfig()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    trace_list = list(traces)
    bundles = _trace_examples(trace_list, agent_names, cfg.only_passed)
    if not bundles:
        return ReinforceResult(epoch_returns=[], epoch_losses=[], epoch_entropy=[], num_episodes=0)

    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    epoch_returns: list[float] = []
    epoch_losses: list[float] = []
    epoch_entropies: list[float] = []

    for _ in range(cfg.epochs):
        rng.shuffle(bundles)
        epoch_loss = 0.0
        epoch_entropy = 0.0
        epoch_return = 0.0

        for trace, examples in bundles:
            reward = compute_reward(trace.raw, cfg.reward)
            episode_return = float(reward)

            policy_loss = torch.zeros((), dtype=torch.float32)
            value_loss = torch.zeros((), dtype=torch.float32)
            entropy = torch.zeros((), dtype=torch.float32)

            for ex in examples:
                cs = controller.builder.from_runtime_sync(ex.runtime_state)
                out = controller(cs)
                # log pi(kind) at expert kind label
                kind_logp = F.log_softmax(out.kind_logits, dim=-1)
                step_logp = kind_logp[ex.label_kind]
                step_entropy = -(F.softmax(out.kind_logits, dim=-1) * kind_logp).sum()

                # If kind is activate_agent and we have agent logits, add agent term.
                if ex.label_agent is not None and out.agent_logits.numel() > 0:
                    agent_logp = F.log_softmax(out.agent_logits, dim=-1)
                    if 0 <= ex.label_agent < int(agent_logp.shape[-1]):
                        step_logp = step_logp + agent_logp[ex.label_agent]
                        step_entropy = (
                            step_entropy - (F.softmax(out.agent_logits, dim=-1) * agent_logp).sum()
                        )

                advantage = episode_return
                if cfg.use_value_baseline:
                    advantage = episode_return - float(out.value.detach())
                    value_loss = value_loss + F.mse_loss(
                        out.value,
                        torch.tensor(episode_return, dtype=out.value.dtype),
                    )

                policy_loss = policy_loss - step_logp * advantage
                entropy = entropy + step_entropy

            n = max(1, len(examples))
            policy_loss = policy_loss / n
            entropy = entropy / n
            loss = (
                policy_loss + cfg.value_loss_weight * (value_loss / n) - cfg.entropy_bonus * entropy
            )

            optimizer.zero_grad()
            if loss.requires_grad and not (math.isnan(loss.item()) or math.isinf(loss.item())):
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_entropy += float(entropy.detach())
            epoch_return += episode_return

        epoch_returns.append(epoch_return / len(bundles))
        epoch_losses.append(epoch_loss / len(bundles))
        epoch_entropies.append(epoch_entropy / len(bundles))

    return ReinforceResult(
        epoch_returns=epoch_returns,
        epoch_losses=epoch_losses,
        epoch_entropy=epoch_entropies,
        num_episodes=len(bundles),
    )


__all__ = ["ReinforceConfig", "ReinforceResult", "reinforce_train"]
